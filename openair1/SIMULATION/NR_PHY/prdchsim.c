/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <string.h>
#include <math.h>
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "common/utils/LOG/log.h"
#include "common/utils/nr/nr_common.h"
#include "common/ran_context.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_common.h"

#include "PHY_AIOT/defs_aiot_gNB.h"
#include "PHY_AIOT/defs_aiot_UE.h"
#include "PHY_AIOT/defs_aiot_r2d.h"

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, not finished yet */
  return "detect_leaks=0";
}

PHY_VARS_gNB_AIOT *gNB_AIOT;
PHY_VARS_NR_UE_AIOT *UE;
RAN_CONTEXT_t RC;
NR_DL_FRAME_PARMS *frame_parms;

double cpuf;

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void) {
  return &softmodem_params;
}

void lineEncoding(uint8_t *in, int inLenBits, uint8_t *out, int outLenBits)
{
  memset(out, 0, (outLenBits + 7) / 8);

  for (int i = 0; i < inLenBits; i++) {
    int bit = (in[i / 8] >> (7 - (i % 8))) & 0x01;

    // Line encoding: 0 -> 01, 1 -> 10
    int outBitPos = 2 * i;
    int outByteIdx = outBitPos / 8;
    int outBitIdx = 7 - (outBitPos % 8);

    if (bit == 0) {
      // 0 -> 01
      out[outByteIdx] |= (1 << (outBitIdx - 1));
    } else {
      // 1 -> 10
      out[outByteIdx] |= (1 << outBitIdx);
    }
  }
}

void headersInsertion(uint8_t *in, int inLenBits, uint8_t *out, int outLenBits)
{
  memset(out, 0, (outLenBits + 7) / 8);

  out[0] = R_TAS_SIP; // R_TAS_SIP
  out[1] = R_TAS_CAP << 4; // R_TAS_CAP

  for (int i = 0; i < inLenBits / 8; i++) {
    out[i + 1] |= (in[i] >> 4) & 0x0F;
    out[i + 2] = (in[i] & 0x0F) << 4;
  }

  int postambleBit = N_R_TAS_SIP + N_R_TAS_CAP + inLenBits;
  out[postambleBit / 8] |= R2D_POSTAMBLE << (4 - (postambleBit % 8));
}

char filename[30];

configmodule_interface_t *uniqCfg = NULL;
int main(int argc, char **argv)
{
  stop = false;
  __attribute__((unused)) struct sigaction oldaction;
  sigaction(SIGINT, &sigint_action, &oldaction);

  int loglvl = OAILOG_WARNING;

  cpuf = get_cpu_freq_GHz();

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_AIOT_PRDCHSIM] Error, configuration module init failed\n");
  }

  int c;
  while ((c = getopt(argc, argv, "--:O:h:")) != -1) {
    /* ignore long options starting with '--', option '-O' and their arguments that are handled by configmodule */
    /* with this opstring getopt returns 1 for non-option arguments, refer to 'man 3 getopt' */
    if (c == 1 || c == '-' || c == 'O')
      continue;

    printf("handling optarg %c\n", c);
    switch (c) {
      default:
      case 'h':
        printf("%s -h(elp)\n", argv[0]);
        // printf("%s -h(elp) -p(extended_prefix) -N cell_id -f output_filename -F input_filename -g channel_model -n n_frames -t
        // Delayspread -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant -i Intefrence0 -j Interference1 -A interpolation_file
        // -C(alibration offset dB) -N CellId\n", argv[0]);
        printf("-h This message\n");
        exit(-1);
        break;
    }
  }

  logInit();
  set_glog(loglvl);

  get_softmodem_params()->phy_test = 1;
  get_softmodem_params()->do_ra = 0;
  IS_SOFTMODEM_DLSIM = true;

  randominit(0);
  load_dftslib();
  crcTableInit();
  InitSinLUT();

  // ---------------------------------------------------------------

  int RBs = 10;
  c16_t freqFrame[14*12*RBs];

  memset(freqFrame, 0, sizeof(freqFrame));

  int n_antennas = 1;

  NR_DL_FRAME_PARMS frame_parms_storage;
  frame_parms = &frame_parms_storage;
  memset(frame_parms, 0, sizeof(NR_DL_FRAME_PARMS));

  frame_parms->N_RB_DL = RBs;
  frame_parms->N_RB_UL = RBs;
  frame_parms->Ncp = 0; // normal CP
  frame_parms->nb_antennas_tx = n_antennas;
  frame_parms->nb_antennas_rx = n_antennas;
  frame_parms->subcarrier_spacing = 15e3;
  frame_parms->symbols_per_slot = 14;
  frame_parms->slots_per_subframe = 1;
  frame_parms->slots_per_frame = 10;
  frame_parms->ofdm_symbol_size = 2048;
  frame_parms->nb_prefix_samples = 144;
  frame_parms->nb_prefix_samples0 = 160;
  frame_parms->numerology_index = 0;
  frame_parms->samples_per_frame_wCP = frame_parms->slots_per_frame *
                             frame_parms->symbols_per_slot *
                             frame_parms->ofdm_symbol_size;
  frame_parms->samples_per_subframe = (frame_parms->nb_prefix_samples0 + frame_parms->ofdm_symbol_size) * 2 + 
                             (frame_parms->nb_prefix_samples + frame_parms->ofdm_symbol_size) * (frame_parms->symbols_per_slot * frame_parms->slots_per_subframe - 2); 

  int slot = 1;

  int frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  //frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  int slot_offset = frame_parms->samples_per_subframe * slot;
  int slot_length = slot_offset - frame_parms->samples_per_subframe * (slot - 1);
  printf("frame_length_complex_samples %d, slot_offset %d, slot_length %d\n",frame_length_complex_samples,slot_offset,slot_length);

  c16_t **txDataF, **txData, **rxData;

  txData = malloc(n_antennas * sizeof(int*));
  txDataF = malloc(n_antennas * sizeof(int*));
  rxData = malloc(n_antennas * sizeof(int*));

  for (int i = 0; i < n_antennas; i++) {
    printf("Allocating %d samples for txdata\n", frame_length_complex_samples);
    txData[i] = calloc(1, frame_length_complex_samples * sizeof(int));
    txDataF[i] = calloc(1, frame_length_complex_samples * sizeof(int));
    rxData[i] = calloc(1, frame_length_complex_samples * sizeof(int));
  }

  int ofdm_symbol_size = frame_parms->ofdm_symbol_size;
  int num_symbols = 14;
  int points_per_symbol = 12 * RBs;

  memset(freqFrame, 0, sizeof(freqFrame));
  memcpy(freqFrame, SIP_symbol_Ones, sizeof(SIP_symbol_Ones));
  memset(txDataF[0], 0, frame_length_complex_samples * sizeof(int));

  for (int sym = 0; sym < num_symbols; sym++) {
    int outMod_offset = sym * points_per_symbol;
    int txData_offset = sym * ofdm_symbol_size;

    for (int n = 0; n < points_per_symbol; n++) {
      txDataF[0][txData_offset + n] = freqFrame[outMod_offset + n];
    }
  }

  bool was_symbol_used[NR_NUMBER_OF_SYMBOLS_PER_SLOT];
  for (int i = 0; i < 14; i++) {
    was_symbol_used[i] = true;
  }

  nr_normal_prefix_mod(txDataF[0],
                      txData[0],
                      14,
                      frame_parms,
                      1,
                      was_symbol_used);

  sprintf(filename,"ofdm_SIP_2RBS_Ones.m");
  LOG_M(filename,"ofdm", txData[0], slot_length, 1, 1);


  // ZC
  memset(freqFrame, 0, sizeof(freqFrame));
  memcpy(freqFrame, SIP_symbol_ZC, sizeof(SIP_symbol_ZC));
  memset(txDataF[0], 0, frame_length_complex_samples * sizeof(int));

  for (int sym = 0; sym < num_symbols; sym++) {
    int outMod_offset = sym * points_per_symbol;
    int txData_offset = sym * ofdm_symbol_size;

    for (int n = 0; n < points_per_symbol; n++) {
      txDataF[0][txData_offset + n] = freqFrame[outMod_offset + n];
    }
  }

  for (int i = 0; i < 14; i++) {
    was_symbol_used[i] = true;
  }

  nr_normal_prefix_mod(txDataF[0],
                      txData[0],
                      14,
                      frame_parms,
                      1,
                      was_symbol_used);

  sprintf(filename,"ofdm_SIP_2RBS_ZC.m");
  LOG_M(filename,"ofdm", txData[0], slot_length, 1, 1);

  /*double **s_re,**s_im,**r_re,**r_im;

  s_re = malloc(n_antennas * sizeof(double *));
  s_im = malloc(n_antennas * sizeof(double *));
  r_re = malloc(n_antennas * sizeof(double *));
  r_im = malloc(n_antennas * sizeof(double *));

  for (int i = 0; i < n_antennas; i++) {
    s_re[i] = calloc(1, slot_length * sizeof(double));
    s_im[i] = calloc(1, slot_length * sizeof(double));
  }

  for (int i = 0; i < n_antennas; i++) {
    r_re[i] = calloc(1, slot_length * sizeof(double));
    r_im[i] = calloc(1, slot_length * sizeof(double));
  }

  bzero(r_re[0], slot_length * sizeof(double));
  bzero(r_im[0], slot_length * sizeof(double));

  printf("frame_length_complex_samples %d, slot_offset %d, slot_length %d\n",frame_length_complex_samples,slot_offset,slot_length);

  uint8_t nb_antennas = 1;
  SCM_t channel_model = AWGN;
  uint64_t fc = 0; // Carrier frequency n8 band, #50 RB
  double DS_TDL = .03;
  int delay = 0;
  int N_RB_DL = 6;
  double samples = N_RB2sampling_rate(N_RB_DL);
  double rxbw = N_RB2channel_bandwidth(N_RB_DL);

  channel_desc_t *channel = new_channel_desc_scm(nb_antennas,
                                  nb_antennas,
                                  channel_model,
                                  samples,//sampling frequency in MHz
                                  fc,
                                  rxbw,
                                  DS_TDL,
                                  0.0,
                                  CORR_LEVEL_LOW,
                                  0,
                                  delay,
                                  0,
                                  0);
  
  int txlev;
  int l_ofdm = 6;
  double SNR = 10.0;
  txlev = signal_energy((int32_t *)txData[0],
  frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples);
  printf("txlev = %d (%f dB)\n",txlev,10*log10((double)txlev));

  for (int i = 0; i < slot_length; i++) {
    s_re[0][i] = (double) txData[0][0].r;
    s_im[0][i] = (double) txData[0][0].i;
  }

  double ts = 1.0/(frame_parms->subcarrier_spacing * frame_parms->ofdm_symbol_size); 
  //Compute AWGN variance
  double sigma2_dB = 10 * log10((double)txlev * ((double)frame_parms->ofdm_symbol_size/(12*1))) - SNR;
  double sigma2    = pow(10, sigma2_dB/10);
  printf("sigma2 %f (%f dB), txlev %f (factor %f)\n",sigma2,sigma2_dB,10*log10((double)txlev),(double)(double)frame_parms->ofdm_symbol_size/(12*1));

  multipath_channel(channel, s_re, s_im, r_re, r_im, slot_length, 0, 1);
  add_noise(rxData,
            (const double **)r_re,
            (const double **)r_im,
            sigma2,
            slot_length,
            slot_offset,
            ts,
            delay,
            0x0,
            0x1,
            frame_parms->nb_antennas_rx);

  for (int i = 0; i < n_antennas; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }
  for (int i = 0; i < n_antennas; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);

  free_channel_desc_scm(channel);*/

  return 0;
}
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
softmodem_params_t *get_softmodem_params(void)
{
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

// Butterworth 3rd-order LPF coefficients (fs=1.92 MHz, fc=0.625 MHz)
static const double b[4] = {0.0976, 0.2929, 0.2929, 0.0976};
static const double a[4] = {1.0000, -0.0000, 0.1716, -0.0000};

// Filter state
typedef struct {
  double xr[4]; // past real inputs
  double xi[4]; // past imag inputs
  double yr[4]; // past real outputs
  double yi[4]; // past imag outputs
} iir_state_t;

void butterworth_filter(c16_t *in, c16_t *out, int length, iir_state_t *st)
{
  for (int n = 0; n < length; n++) {
    // shift history
    for (int k = 3; k > 0; k--) {
      st->xr[k] = st->xr[k - 1];
      st->xi[k] = st->xi[k - 1];
      st->yr[k] = st->yr[k - 1];
      st->yi[k] = st->yi[k - 1];
    }

    // new input
    st->xr[0] = (double)in[n].r;
    st->xi[0] = (double)in[n].i;

    // apply filter
    double yr = 0.0, yi = 0.0;

    // feedforward
    for (int k = 0; k < 4; k++) {
      yr += b[k] * st->xr[k];
      yi += b[k] * st->xi[k];
    }

    // feedback (skip a[0]=1)
    for (int k = 1; k < 4; k++) {
      yr -= a[k] * st->yr[k];
      yi -= a[k] * st->yi[k];
    }

    // store outputs
    st->yr[0] = yr;
    st->yi[0] = yi;

    // clamp to int16_t
    if (yr > 32767)
      yr = 32767;
    if (yr < -32768)
      yr = -32768;
    if (yi > 32767)
      yi = 32767;
    if (yi < -32768)
      yi = -32768;

    out[n].r = (int16_t)yr;
    out[n].i = (int16_t)yi;
  }
}

char filename[30];
char foldername[] = "results/";

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

  int RBs = 6;
  c16_t freqFrame[NR_NUMBER_OF_SYMBOLS_PER_SLOT * NR_NB_SC_PER_RB * RBs];

  memset(freqFrame, 0, sizeof(freqFrame));

  const int n_antennas = 1;

  NR_DL_FRAME_PARMS frame_parms_storage;
  frame_parms = &frame_parms_storage;
  memset(frame_parms, 0, sizeof(NR_DL_FRAME_PARMS));

  frame_parms->N_RB_DL = RBs;
  frame_parms->N_RB_UL = RBs;
  frame_parms->N_RB_SL = 0;
  frame_parms->Ncp = 0; // normal CP
  frame_parms->frame_type = TDD;
  frame_parms->nb_antennas_tx = n_antennas;
  frame_parms->nb_antennas_rx = n_antennas;
  frame_parms->subcarrier_spacing = 15e3;
  frame_parms->symbols_per_slot = NR_NUMBER_OF_SYMBOLS_PER_SLOT;
  frame_parms->slots_per_subframe = 1;
  frame_parms->slots_per_frame = 10;
  frame_parms->ofdm_symbol_size = 2048;
  frame_parms->nb_prefix_samples = 144;
  frame_parms->nb_prefix_samples0 = 160;
  frame_parms->numerology_index = 0;
  frame_parms->samples_per_frame_wCP = frame_parms->slots_per_frame * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;
  frame_parms->samples_per_subframe = (frame_parms->nb_prefix_samples0 + frame_parms->ofdm_symbol_size) * 2
                                      + (frame_parms->nb_prefix_samples + frame_parms->ofdm_symbol_size)
                                            * (frame_parms->symbols_per_slot * frame_parms->slots_per_subframe - 2);

  int slot = 1;

  int frame_length_complex_samples = frame_parms->samples_per_subframe * NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  // frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  int slot_offset = frame_parms->samples_per_subframe * slot;
  int slot_length = slot_offset - frame_parms->samples_per_subframe * (slot - 1);
  printf("frame_length_complex_samples %d, slot_offset %d, slot_length %d\n",
         frame_length_complex_samples,
         slot_offset,
         slot_length);

  c16_t **txDataF, **txData, **rxData, *filteredData;

  txData = malloc(n_antennas * sizeof(int *));
  txDataF = malloc(n_antennas * sizeof(int *));
  rxData = malloc(n_antennas * sizeof(int *));
  filteredData = calloc(1, frame_length_complex_samples * sizeof(int));

  for (int i = 0; i < n_antennas; i++) {
    printf("Allocating %d samples for txdata\n", frame_length_complex_samples);
    txData[i] = calloc(1, frame_length_complex_samples * sizeof(int));
    txDataF[i] = calloc(1, frame_length_complex_samples * sizeof(int));
    rxData[i] = calloc(1, frame_length_complex_samples * sizeof(int));
  }

  int ofdm_symbol_size = frame_parms->ofdm_symbol_size;
  int num_symbols = NR_NUMBER_OF_SYMBOLS_PER_SLOT;
  int points_per_symbol = NR_NB_SC_PER_RB * RBs;
  bool was_symbol_used[NR_NUMBER_OF_SYMBOLS_PER_SLOT];

  memset(freqFrame, 0, sizeof(freqFrame));
  memcpy(freqFrame, SIP_SCs_ZC_6RBs_M4, sizeof(SIP_SCs_ZC_6RBs_M4));
  memset(txDataF[0], 0, frame_length_complex_samples * sizeof(int));

  for (int sym = 0; sym < num_symbols; sym++) {
    int outMod_offset = sym * points_per_symbol;
    int txData_offset = sym * ofdm_symbol_size;

    for (int n = 0; n < points_per_symbol; n++) {
      txDataF[0][txData_offset + n] = freqFrame[outMod_offset + n];
    }
  }

  for (int i = 0; i < NR_NUMBER_OF_SYMBOLS_PER_SLOT; i++) {
    was_symbol_used[i] = true;
  }

  nr_normal_prefix_mod(txDataF[0], txData[0], NR_NUMBER_OF_SYMBOLS_PER_SLOT, frame_parms, 1, was_symbol_used);

  sprintf(filename, "%s/ofdm_SIP_%dRBS_ZC.m", foldername, RBs);
  LOG_M(filename, "ofdm", txData[0], slot_length, 1, 1);

  double **s_re, **s_im, **r_re, **r_im;

  s_re = malloc(NB_ANTENNAS_TX * sizeof(double *));
  s_im = malloc(NB_ANTENNAS_TX * sizeof(double *));
  r_re = malloc(NB_ANTENNAS_TX * sizeof(double *));
  r_im = malloc(NB_ANTENNAS_TX * sizeof(double *));

  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    s_re[i] = calloc(1, slot_length * sizeof(double));
    s_im[i] = calloc(1, slot_length * sizeof(double));
  }

  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    r_re[i] = calloc(1, slot_length * sizeof(double));
    r_im[i] = calloc(1, slot_length * sizeof(double));
  }

  bzero(s_re[0], slot_length * sizeof(double));
  bzero(s_im[0], slot_length * sizeof(double));
  bzero(r_re[0], slot_length * sizeof(double));
  bzero(r_im[0], slot_length * sizeof(double));

  SCM_t channel_model = AWGN;
  uint64_t fc = 897500000; // Carrier frequency n8 band, #50 RB
  double DS_TDL = 0; //.03;
  double SNR = 0; // 30.0;
  double path_loss_dB = 0; //-20;
  double noise_power_dB = -160.0;
  int delay = 0;
  double samples = N_RB2sampling_rate(RBs);
  double rxbw = N_RB2channel_bandwidth(RBs);

  channel_desc_t *channel = new_channel_desc_scm(n_antennas,
                                                 n_antennas,
                                                 channel_model,
                                                 samples, // sampling frequency in MHz
                                                 fc,
                                                 rxbw,
                                                 DS_TDL,
                                                 0.0,
                                                 CORR_LEVEL_LOW,
                                                 0,
                                                 delay,
                                                 path_loss_dB,
                                                 noise_power_dB);

  int txlev = signal_energy((int32_t *)&txData[0][0], frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples);
  double txlev_dB = 10 * log10((double)txlev);
  printf("txlev = %d (%f dB)\n", txlev, txlev_dB);

  for (int i = 0; i < slot_length; i++) {
    s_re[0][i] = (double)txData[0][i].r;
    s_im[0][i] = (double)txData[0][i].i;
  }

  for (int i = 0; i < 16; i++) {
    printf("s_re[0][%d] = %f, s_im[0][%d] = %f\n", i, s_re[0][i], i, s_im[0][i]);
  }

  double *output = malloc(2 * sizeof(double) * slot_length);
  for (int i = 0; i < slot_length; i++) {
    output[2 * i] = s_re[0][i];
    output[2 * i + 1] = s_im[0][i];
  }

  sprintf(filename, "%s/channel.m", foldername);
  LOG_M(filename, "channelx", output, slot_length, 1, 8);

  double ts = 1.0 / (frame_parms->subcarrier_spacing * frame_parms->ofdm_symbol_size);
  // Compute AWGN variance
  double sigma2_dB = 10 * log10((double)txlev * ((double)frame_parms->ofdm_symbol_size / points_per_symbol)) + path_loss_dB - SNR;
  double sigma2 = pow(10, sigma2_dB / 10);
  printf("sigma2 %f (%f dB), txlev %f (factor %f)\n",
         sigma2,
         sigma2_dB,
         10 * log10((double)txlev),
         (double)(double)frame_parms->ofdm_symbol_size / points_per_symbol);

  sigma2 = 0;
  delay = 0;

  multipath_channel(channel, s_re, s_im, r_re, r_im, slot_length, 0, 1);
  add_noise(rxData,
            (const double **)r_re,
            (const double **)r_im,
            sigma2,
            slot_length,
            0,
            ts,
            delay,
            0x0,
            0x1,
            frame_parms->nb_antennas_rx);

  for (int i = 0; i < slot_length; i++) {
    output[2 * i] = r_re[0][i];
    output[2 * i + 1] = r_im[0][i];
  }

  sprintf(filename, "%s/channel_out.m", foldername);
  LOG_M(filename, "channelx_out", output, slot_length, 1, 8);

  free(output);

  sprintf(filename, "%s/rxdata_6RBs.m", foldername);
  LOG_M(filename, "rxdata", rxData[0], slot_length, 1, 1);

  // Envelope detector (squared)
  for (int i = 0; i < slot_length; i++) {
    rxData[0][i] = c16MulConjShift(rxData[0][i], rxData[0][i], 15);
  }

  sprintf(filename, "%s/envelope.m", foldername);
  LOG_M(filename, "envelopex", rxData[0], slot_length, 1, 1);

  // Butterworth filter 3rd order
  iir_state_t st = {0};

  // fill input[] with complex samples
  butterworth_filter(rxData[0], filteredData, slot_length, &st);

  sprintf(filename, "%s/filtered.m", foldername);
  LOG_M(filename, "filteredx", filteredData, slot_length, 1, 1);

  // Downsample the signal by N = 16
  int N = 16;
  int downSampled_length = slot_length / N;
  c16_t *downSampled = malloc(downSampled_length * sizeof(c16_t));
  for (int i = 0; i < downSampled_length; i++) {
    downSampled[i] = filteredData[i * N];
  }

  sprintf(filename, "%s/downsampled.m", foldername);
  LOG_M(filename, "downsampledx", downSampled, downSampled_length, 1, 1);

  // Correlate the received signal with known SIP sequence
  int SIP_downsampled_length = 2*ofdm_symbol_size/N + frame_parms->nb_prefix_samples/N;
  c16_t SIP_ideal[SIP_downsampled_length];
  c16_t value0 = (c16_t){0, 0};
  c16_t value1 = (c16_t){16384, 0};

  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < ofdm_symbol_size/N/4; k++)
    {
      SIP_ideal[i*ofdm_symbol_size/N/4 + k] = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;
    }
  }

  for (int k = 0; k < frame_parms->nb_prefix_samples/N; k++)
  {
    SIP_ideal[4*ofdm_symbol_size/N/4 + k] = value0;
  }

  for (int i = 4; i < 8; i++) {
    for (int k = 0; k < ofdm_symbol_size/N/4; k++)
    {
      SIP_ideal[frame_parms->nb_prefix_samples/N + i*ofdm_symbol_size/N/4 + k] = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;
    }
  }

  sprintf(filename, "%s/SIP_ideal.m", foldername);
  LOG_M(filename, "SIP_idealx", SIP_ideal, SIP_downsampled_length, 1, 1);

  double *correlation = malloc(downSampled_length * sizeof(double));
  memset(correlation, 0, downSampled_length * sizeof(double));

  int corr_len = sizeof(SIP_ideal) / sizeof(SIP_ideal[0]);
  for (int i = 0; i < downSampled_length - corr_len; i++) {
    double sum = 0.0;
    for (int j = 0; j < corr_len; j++) {
      sum += downSampled[i + j].r * SIP_ideal[j].r + downSampled[i + j].i * SIP_ideal[j].i;
    }
    correlation[i] = sum;
  }

  sprintf(filename, "%s/correlation.m", foldername);
  LOG_M(filename, "correlationx", correlation, downSampled_length, 1, 7);

  double max_value = 0.0;
  int SIP_offset = 0;
  for (int i = 0; i < downSampled_length - corr_len; i++)
  {
    if(max_value < correlation[i]) {
      max_value = correlation[i];
      SIP_offset = i;
    }
  }

  printf("SIP offset: %d\n", SIP_offset);

  free(correlation);

  // Remove CP (CP0 size = 10, CP size = 9, OFDM symbol size = 128)
  /*int cleared_length = num_symbols * (frame_parms->ofdm_symbol_size/N);
  c16_t *cleared = malloc(cleared_length * sizeof(c16_t));
  for (int i = 0, cp = 0; i < num_symbols; i++)
  {
    cp += ((i%7) == 0) ? frame_parms->nb_prefix_samples0/N : frame_parms->nb_prefix_samples/N;
    for (int j = 0; j < frame_parms->ofdm_symbol_size/N; j++) {
      cleared[i*(frame_parms->ofdm_symbol_size/N) + j] = downSampled[i*frame_parms->ofdm_symbol_size/N + j + cp];
    }
  }

  sprintf(filename, "%s/cleared.m", foldername);
  LOG_M(filename, "clearedx", cleared, cleared_length, 1, 1);

  // Calculate energy of chips
  double *energy = malloc(4 * num_symbols * sizeof(double));
  memset(energy, 0, 4 * num_symbols * sizeof(double));
  for (int i = 0; i < num_symbols*4; i++) {
    for (int j = 0; j < (frame_parms->ofdm_symbol_size/N/4); j++)
    {
      energy[i] += (double)cleared[i*(frame_parms->ofdm_symbol_size/N/4) + j].r;
    }
  }

  sprintf(filename, "%s/energy.m", foldername);
  LOG_M(filename, "energyx", energy, num_symbols*4, 1, 7);

  // Decode line encoding
  uint8_t payload = 0;
  for (int i = 0; i < 8; i+=2) {
    int bit0 = (energy[i] > energy[i+1]) ? 1 : 0;
    payload = (payload << 1) | bit0;
  }

  printf("Received payload: 0x%02X\n", payload);

  free(energy);
  free(cleared);*/
  free(downSampled);

  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }
  for (int i = 0; i < NB_ANTENNAS_RX; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);

  free_channel_desc_scm(channel);

  return 0;
}
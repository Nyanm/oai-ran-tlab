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

NR_AIOT_DL_FRAME_PARMS *frame_parms;

double cpuf;
char filename[50];
char foldername[] = "results/";

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void)
{
  return &softmodem_params;
}

// Default parametrs
#define RBs_DEFAULT 6
#define M_DEFAULT 4
#define ZC_Ones_DEFAULT true
#define PAYLOAD_SIZE_DEFAULT 32 // in bits
#define ANTENNAS_DEFAULT 1
#define SUBCARRIER_SPACING_DEFAULT 15e3
#define OFDM_SYMBOL_SIZE_DEFAULT 2048
#define CP0_SYMBOL_SIZE_DEFAULT 160
#define CP_SYMBOL_SIZE_DEFAULT 144

void AIOT_R2D_PHY_calc_packet_sizes(int payloadSize, NR_AIOT_DL_FRAME_PARMS *frame)
{
  frame->packet_encoded_size = 2 * payloadSize;
  frame->packet_symbols = R_TAS_SIP_N / R_TAS_SIP_M
                          + ((R_TAS_CAP_N + frame->packet_encoded_size + R2D_POSTAMBLE_N) + (frame->M - 1)) / frame->M; // ceil
  frame->packet_slots = (frame->packet_symbols + (NR_NUMBER_OF_SYMBOLS_PER_SLOT - 1)) / NR_NUMBER_OF_SYMBOLS_PER_SLOT; // ceil
  frame->packet_subcarriers = NR_NB_SC_PER_RB * frame->nr_frame_parms.N_RB_DL;
  frame->packet_samples =
      frame->nr_frame_parms.samples_per_subframe / frame->nr_frame_parms.slots_per_subframe * frame->packet_slots;

  printf("Calculated R2D packet size: %d symbols, each %d SCs\n", frame->packet_symbols, frame->packet_subcarriers);
}

c16_t *AIOT_R2D_PHY_REs(uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame)
{
  printf("Creating R2D packet\n");

  // Allocate memory for the whole R2D packet
  c16_t *REsPacket = malloc(frame->packet_symbols * frame->packet_subcarriers * sizeof(c16_t));
  if (REsPacket == NULL) {
    printf("Memory allocation failed\n");
    return NULL;
  } else {
    printf("Allocated %d symbols (each %d SCs) for R2D packet\n", frame->packet_symbols, frame->packet_subcarriers);
  }

  // Copy R-TAS SIP preamble to the packet
  const c16_t *SIP_SCs_selection = SIP_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->Zadoff_Chu);
  int SIP_length = R_TAS_SIP_N / R_TAS_SIP_M * frame->packet_subcarriers;
  memcpy(REsPacket, SIP_SCs_selection, SIP_length * sizeof(c16_t)); // SIP
  printf("Inserted R-TAS SIP preamble\n");

  // Copy R-TAS CAS preamble to the packet
  const c16_t *CAS_SCs_selection = CAS_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int CAS_length = R_TAS_CAP_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length, CAS_SCs_selection, CAS_length * sizeof(c16_t)); // CAS
  printf("Inserted R-TAS CAS\n");

  // Copy payload to the packet (using Manchester line encoding embedded in the symbol mapping)
  int Payload_length = (frame->packet_encoded_size + (frame->M - 1)) / frame->M * frame->packet_subcarriers;
  for (int i = 0, j = 0; i < frame->packet_encoded_size / 2; i += frame->M / 2, j++) {
    uint8_t symbol = 0;
    for (int k = 0; k < frame->M / 2; k++) {
      if (i + k < frame->packet_encoded_size / 2) {
        int bit = (payload[(i + k) / 8] >> (7 - ((i + k) % 8))) & 0x01;
        symbol = (symbol << 1) | bit;
      } else {
        symbol = (symbol << 1);
      }
    }

    const c16_t *Payload_SCs_selection = Payload_SCs_select(frame->nr_frame_parms.N_RB_DL, symbol, frame->Zadoff_Chu);
    memcpy(REsPacket + SIP_length + CAS_length + j * frame->packet_subcarriers,
           Payload_SCs_selection,
           frame->packet_subcarriers * sizeof(c16_t));
  }
  printf("Inserted payload of %d bits\n", frame->packet_encoded_size);

  // Copy R-TAS postamble
  const c16_t *Postamble_SCs_selection = Postamble_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int Postamble_length = R2D_POSTAMBLE_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length + CAS_length + Payload_length,
         Postamble_SCs_selection,
         Postamble_length * sizeof(c16_t)); // Postamble
  printf("Inserted postamble\n");

  sprintf(filename, "%s/1_REs_Packet.m", foldername);
  LOG_M(filename, "REs_Packet_sig", REsPacket, frame->packet_symbols * frame->packet_subcarriers, 1, 1);

  return REsPacket;
}

void AIOT_R2D_PHY_REs_free(c16_t *REsPacket)
{
  if (REsPacket != NULL)
    free(REsPacket);
}

c16_t *AIOT_R2D_PHY_Signal(c16_t *REsPacket, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Allocate TX buffers
  c16_t *txData = calloc(1, frame->packet_samples * sizeof(c16_t));
  memset(txData, 0, frame->packet_samples * sizeof(c16_t));

  c16_t *txDataF = calloc(1, frame->packet_samples * sizeof(c16_t));
  memset(txDataF, 0, frame->packet_samples * sizeof(c16_t));

  if(txData == NULL || txDataF == NULL) {
    printf("Memory allocation failed\n");
    return NULL;
  } else {
    printf("Allocated %d samples for txdata\n", frame->packet_samples);
  }

  for (int sym = 0; sym < frame->packet_symbols; sym++) {
    int in_offset = sym * frame->packet_subcarriers;
    int out_offset = sym * frame->nr_frame_parms.ofdm_symbol_size;

    for (int n = 0; n < frame->packet_subcarriers; n++) {
      txDataF[out_offset + n] = REsPacket[in_offset + n];
    }
  }

  bool was_symbol_used[NR_NUMBER_OF_SYMBOLS_PER_SLOT];
  for (int i = 0; i < NR_NUMBER_OF_SYMBOLS_PER_SLOT; i++) {
    was_symbol_used[i] = true;
  }

  for (int slot = 0; slot < frame->packet_slots; slot++) {
    nr_normal_prefix_mod(txDataF + slot * frame->nr_frame_parms.samples_per_slot_wCP,
                         txData + slot * (frame->nr_frame_parms.samples_per_subframe / frame->nr_frame_parms.slots_per_subframe),
                         0,
                         (const NR_DL_FRAME_PARMS *)&frame->nr_frame_parms,
                         0,
                         was_symbol_used);
  }

  free(txDataF);

  sprintf(filename, "%s/2_TX_Frame.m", foldername);
  LOG_M(filename, "TX_Frame_sig", txData, frame->packet_samples, 1, 1);

  return txData;
}


void AIOT_R2D_PHY_Signal_free(c16_t *txData)
{
  if (txData != NULL)
    free(txData);
}


void AIOT_R2D_PHY_IQ(double ***s_re, double ***s_im, c16_t *txData, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Initialization of channel signals
  *s_re = malloc(NB_ANTENNAS_TX * sizeof(double *));
  *s_im = malloc(NB_ANTENNAS_TX * sizeof(double *));

  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    (*s_re)[i] = calloc(1, frame->packet_samples * sizeof(double));
    (*s_im)[i] = calloc(1, frame->packet_samples * sizeof(double));
  }

  for (int i = 0; i < frame->packet_samples; i++) {
    (*s_re)[0][i] = (double)txData[i].r;
    (*s_im)[0][i] = (double)txData[i].i;
  }
}

void AIOT_R2D_PHY_IQ_free(double **s_re, double **s_im)
{
  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }
  free(s_re);
  free(s_im);
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

  // Prepare variables for parameters
  SCM_t channel_model = AWGN;
  uint64_t fc = 897500000; // Carrier frequency n8 band, #50 RB
  double DS_TDL = .03;
  double SNR = 20.0;
  double path_loss_dB = -40.0;
  double noise_power_dB = -140.0;
  int delay = 1290;

  uint8_t payload[MAX_AIOT_R2D_PAYLOAD_SIZE] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
                                                0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
  uint16_t payloadSize = PAYLOAD_SIZE_DEFAULT; // payload size in bits

  // Allocate memory for frame parameters
  NR_AIOT_DL_FRAME_PARMS frame_parms_storage;
  frame_parms = &frame_parms_storage;
  memset(frame_parms, 0, sizeof(NR_AIOT_DL_FRAME_PARMS));

  // Common NR frame parameters
  frame_parms->nr_frame_parms.freq_range = FR1;
  frame_parms->nr_frame_parms.N_RB_DL = RBs_DEFAULT;
  frame_parms->nr_frame_parms.N_RB_UL = RBs_DEFAULT;
  frame_parms->nr_frame_parms.N_RB_SL = 0;
  frame_parms->nr_frame_parms.Ncp = 0; // normal CP
  frame_parms->nr_frame_parms.nb_antennas_tx = ANTENNAS_DEFAULT;
  frame_parms->nr_frame_parms.nb_antennas_rx = ANTENNAS_DEFAULT;
  frame_parms->nr_frame_parms.subcarrier_spacing = SUBCARRIER_SPACING_DEFAULT;
  frame_parms->nr_frame_parms.symbols_per_slot = NR_NUMBER_OF_SYMBOLS_PER_SLOT;
  frame_parms->nr_frame_parms.slots_per_subframe = 1;
  frame_parms->nr_frame_parms.slots_per_frame = 10;
  frame_parms->nr_frame_parms.ofdm_symbol_size = OFDM_SYMBOL_SIZE_DEFAULT;
  frame_parms->nr_frame_parms.nb_prefix_samples = CP_SYMBOL_SIZE_DEFAULT;
  frame_parms->nr_frame_parms.nb_prefix_samples0 = CP0_SYMBOL_SIZE_DEFAULT;
  frame_parms->nr_frame_parms.numerology_index = 0;

  frame_parms->nr_frame_parms.samples_per_frame_wCP = frame_parms->nr_frame_parms.slots_per_frame
                                                      * frame_parms->nr_frame_parms.symbols_per_slot
                                                      * frame_parms->nr_frame_parms.ofdm_symbol_size;

  frame_parms->nr_frame_parms.samples_per_slot_wCP =
      frame_parms->nr_frame_parms.ofdm_symbol_size * frame_parms->nr_frame_parms.symbols_per_slot;

  frame_parms->nr_frame_parms.samples_per_subframe_wCP =
      frame_parms->nr_frame_parms.samples_per_slot_wCP * frame_parms->nr_frame_parms.slots_per_subframe;

  frame_parms->nr_frame_parms.samples_per_subframe =
      ((frame_parms->nr_frame_parms.nb_prefix_samples0 + frame_parms->nr_frame_parms.ofdm_symbol_size) * 2
       + (frame_parms->nr_frame_parms.nb_prefix_samples + frame_parms->nr_frame_parms.ofdm_symbol_size)
             * (frame_parms->nr_frame_parms.symbols_per_slot - 2))
      * frame_parms->nr_frame_parms.slots_per_subframe;

  // AIoT-specific frame parameters
  frame_parms->Zadoff_Chu = ZC_Ones_DEFAULT;
  frame_parms->M = M_DEFAULT;

  int c;
  while ((c = getopt(argc, argv, "--:O:h:L:R:P:p:M:Z:S:N:D:")) != -1) {
    /* ignore long options starting with '--', option '-O' and their arguments that are handled by configmodule */
    /* with this opstring getopt returns 1 for non-option arguments, refer to 'man 3 getopt' */
    if (c == 1 || c == '-' || c == 'O')
      continue;

    printf("handling optarg %c\n", c);
    switch (c) {
      default:
      case 'h':
        printf("%s <options>\n", argv[0]);
        printf("-h This message\n");
        printf("-L <log level, 0(errors), 1(warning), 2(analysis), 3(info), 4(debug), 5(trace)>\n");
        printf("-R Number of RBs (supported: 1, 6, 25, 50, 100)\n");
        printf("-P Payload in hex string (e.g. 1234ABCD)\n");
        printf("-p Payload size in bits (max %d)\n", MAX_AIOT_R2D_PAYLOAD_SIZE * 8);
        printf("-M Chips in symbol (supported: 1, 2, 4)\n");
        printf("-Z Zadoff-Chu (1) or Ones (0)\n");
        printf("-S SNR in dB\n");
        printf("-N Path loss in dB\n");
        printf("-D Delay in samples\n");
        exit(-1);
        break;
      case 'L':
        loglvl = atoi(optarg);
        break;
      case 'R':
        int RBs = atoi(optarg);
        if (RBs != 1 && RBs != 6 && RBs != 25 && RBs != 50 && RBs != 100) {
          printf("Error: number of RBs %d not supported, use 1, 6, 25, 50 or 100\n", RBs);
          exit(-1);
        } else {
          printf("Using %d RBs\n", RBs);
          frame_parms->nr_frame_parms.N_RB_DL = RBs;
          frame_parms->nr_frame_parms.N_RB_UL = RBs;
        }
        break;

      case 'P':
        // Load hex string from optarg into payload array
        int hex_len = strlen(optarg);
        if (hex_len > 250)
          hex_len = 250; // 2 chars per byte, max 125 bytes

        for (int i = 0; i < hex_len; i++) {
          sscanf(optarg + 2 * i, "%2hhx", &payload[i]);
        }
        printf("Loaded payload %s of %d bytes\n", payload, hex_len);
        break;

      case 'p':
        payloadSize = atoi(optarg);
        if (payloadSize > MAX_AIOT_R2D_PAYLOAD_SIZE * 8) {
          printf("Error: maximum payload bit size is %d\n", MAX_AIOT_R2D_PAYLOAD_SIZE * 8);
          exit(-1);
        } else {
          printf("Using payload size %d bits\n", payloadSize);
        }
        break;

      case 'M':
        int M = atoi(optarg);
        if (M != 1 && M != 2 && M != 3 && M != 4) {
          printf("Error: M must be between 1 and 4\n");
          exit(-1);
        } else {
          printf("Using M=%d\n", M);
          frame_parms->M = M;
        }
        break;

      case 'Z':
        int ZC_Ones = atoi(optarg);
        if (ZC_Ones != 0 && ZC_Ones != 1) {
          printf("Error: ZC_Ones must be 0 for Ones or 1 for Zadoff-Chu\n");
          exit(-1);
        } else {
          printf("Using %s\n", ZC_Ones ? "Zadoff-Chu" : "Ones");
          frame_parms->Zadoff_Chu = ZC_Ones;
        }
        break;

      case 'S':
        SNR = atof(optarg);
        printf("Using SNR=%f dB\n", SNR);
        break;

      case 'N':
        path_loss_dB = atof(optarg);
        printf("Using path_loss_dB=%f dB\n", path_loss_dB);
        break;

      case 'D':
        delay = atoi(optarg);
        printf("Using delay=%d samples\n", delay);
        break;
    }
  }

  logInit();
  set_glog(loglvl);

  get_softmodem_params()->phy_test = 1;
  get_softmodem_params()->do_ra = 0;
  IS_SOFTMODEM_RFSIM = true;

  randominit(0);
  load_dftslib();
  crcTableInit();
  InitSinLUT();

  // ---------------------------------------------------------------

  // Packet constants calculation
  /*
    int r2d_line_encoded_bits = 2 * payloadSize;
    int r2d_symbols = R_TAS_SIP_N / R_TAS_SIP_M + ((R_TAS_CAP_N + r2d_line_encoded_bits + R2D_POSTAMBLE_N) + (M-1)) / M; // ceil
    int r2d_slots = (r2d_symbols + (NR_NUMBER_OF_SYMBOLS_PER_SLOT-1)) / NR_NUMBER_OF_SYMBOLS_PER_SLOT; // ceil
    int r2d_subcarriers = NR_NB_SC_PER_RB * RBs;
    int r2d_samples = frame_parms->samples_per_subframe / frame_parms->slots_per_subframe * r2d_slots; // packet samples aligned to
    slots
  */

  AIOT_R2D_PHY_calc_packet_sizes(payloadSize, frame_parms);

  printf("R2D packet parameters:\n");
  printf("  RBs: %d\n", frame_parms->nr_frame_parms.N_RB_DL);
  printf("  M: %d\n", frame_parms->M);
  printf("  ZC_Ones: %s\n", frame_parms->Zadoff_Chu ? "Zadoff-Chu" : "Ones");
  printf("  Payload size: %d bits\n", payloadSize);
  printf("  Packet symbols: %d\n", frame_parms->packet_symbols);
  printf("  Packet slots: %d\n", frame_parms->packet_slots);
  printf("  Packet subcarriers: %d\n", frame_parms->packet_subcarriers);
  printf("  Packet samples (aligned to slots): %d\n", frame_parms->packet_samples);

  printf("Original payload (%d bits):\n", payloadSize);
  for (int i = 0; i < (payloadSize + 7) / 8; i++) {
    printf("%02X ", payload[i]);
  }
  printf("\n");

    // Setup radio channel
  double samples = N_RB2sampling_rate(frame_parms->nr_frame_parms.N_RB_DL); // in MHz
  double rxbw = N_RB2channel_bandwidth(frame_parms->nr_frame_parms.N_RB_DL);
  double txbw = N_RB2channel_bandwidth(frame_parms->nr_frame_parms.N_RB_DL);

  printf("Channel parameters:\n");
  printf("  Channel model: %s\n", channel_model == AWGN ? "AWGN" : "TDL");
  printf("  Sampling rate: %f MHz\n", samples);
  printf("  RX bandwidth: %f MHz\n", rxbw);
  printf("  TX bandwidth: %f MHz\n", txbw);
  printf("  Carrier frequency: %f MHz\n", ((double)fc) / 1e6);
  printf("  Delay spread: %f us\n", DS_TDL);
  printf("  SNR: %f dB\n", SNR);
  printf("  Delay: %d samples\n", delay);
  printf("  Path loss: %f dB\n", path_loss_dB);
  printf("  Noise power: %f dB\n", noise_power_dB);

  c16_t *REsPacket = NULL, *txData = NULL;
  double **s_re, **s_im, **r_re, **r_im;

  REsPacket = AIOT_R2D_PHY_REs(payload, frame_parms);
  txData = AIOT_R2D_PHY_Signal(REsPacket, frame_parms);
  AIOT_R2D_PHY_IQ(&s_re, &s_im, txData, frame_parms);

  // Initialization of channel signals
  r_re = malloc(NB_ANTENNAS_TX * sizeof(double *));
  r_im = malloc(NB_ANTENNAS_TX * sizeof(double *));

  int rx_size = frame_parms->packet_samples * 2;
  for (int i = 0; i < NB_ANTENNAS_TX; i++) {
    r_re[i] = calloc(1, rx_size * sizeof(double));
    r_im[i] = calloc(1, rx_size * sizeof(double));
  }

  bzero(r_re[0], rx_size * sizeof(double));
  bzero(r_im[0], rx_size * sizeof(double));

  channel_desc_t *channel = new_channel_desc_scm(frame_parms->nr_frame_parms.nb_antennas_tx,
                                                 frame_parms->nr_frame_parms.nb_antennas_rx,
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

  int txlev = signal_energy((int32_t *) txData, frame_parms->nr_frame_parms.ofdm_symbol_size + frame_parms->nr_frame_parms.nb_prefix_samples0);
  double txlev_dB = 10 * log10((double)txlev);
  printf("Signal energy: %d (%f dB)\n", txlev, txlev_dB);

  for (int i = 0; i < frame_parms->packet_samples; i++) {
    s_re[0][i] = (double)txData[i].r;
    s_im[0][i] = (double)txData[i].i;
  }

  double ts = 1.0 / (frame_parms->nr_frame_parms.subcarrier_spacing * frame_parms->nr_frame_parms.ofdm_symbol_size);
  // Compute AWGN variance
  double sigma2_dB = 10 * log10((double)txlev) + path_loss_dB - SNR; // *((double)frame_parms->ofdm_symbol_size / r2d_subcarriers)
  double sigma2 = pow(10, sigma2_dB / 10);
  printf("Noise sigma2: %f (%f dB)\n", sigma2, sigma2_dB);

  c16_t **rxData = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    printf("Allocating %d samples for rxdata\n", frame_parms->packet_samples * 2);
    rxData[i] = calloc(1, frame_parms->packet_samples * 2 * sizeof(c16_t));
    memset(rxData[i], 0, frame_parms->packet_samples * 2 * sizeof(c16_t));
  }

  multipath_channel(channel, s_re, s_im, r_re, r_im, frame_parms->packet_samples, 0, 1);
  add_noise(rxData,
            (const double **)r_re,
            (const double **)r_im,
            sigma2,
            rx_size,
            0,
            ts,
            0,
            0x0,
            0x1,
            frame_parms->nr_frame_parms.nb_antennas_rx);

  // Save channel output
  double *output = malloc(rx_size * 2 * sizeof(double));

  for (int i = 0; i < rx_size; i++) {
    output[2 * i] = r_re[0][i];
    output[2 * i + 1] = r_im[0][i];
  }

  sprintf(filename, "%s/Channel_Signal_IQ.m", foldername);
  LOG_M(filename, "Channel_IQ", output, rx_size, 1, 8);

  free(output);

  // Save AWGN received signal
  sprintf(filename, "%s/RX_Signal_IQ.m", foldername);
  LOG_M(filename, "RX_IQ", rxData[0], rx_size, 1, 1);

  // Envelope detector (squared)
  for (int i = 0; i < rx_size; i++) {
    rxData[0][i] = c16MulConjShift(rxData[0][i], rxData[0][i], 15);
  }

  sprintf(filename, "%s/Envelope_Signal.m", foldername);
  LOG_M(filename, "Envelope", rxData[0], rx_size, 1, 1);

  // Butterworth filter 3rd order
  iir_state_t st = {0};
  c16_t *filteredData = calloc(1, rx_size * sizeof(c16_t));

  // fill input[] with complex samples
  butterworth_filter(rxData[0], filteredData, rx_size, &st);

  sprintf(filename, "%s/Filtered_Signal.m", foldername);
  LOG_M(filename, "Filtered", filteredData, rx_size, 1, 1);

  // Downsample the signal by N = 16
  int N = 16;
  int downSampled_length = rx_size / N;
  c16_t *downSampled = malloc(downSampled_length * sizeof(c16_t));
  for (int i = 0; i < downSampled_length; i++) {
    downSampled[i] = filteredData[i * N];
  }

  sprintf(filename, "%s/Downsampled_Signal.m", foldername);
  LOG_M(filename, "Downsampled", downSampled, downSampled_length, 1, 1);

  // Correlate the received signal with known SIP sequence
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / N;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / N;
  int SIP_downsampled_length = 2 * downsampled_OFDM_size + downsampled_CP_size;
  c16_t SIP_ideal[SIP_downsampled_length];
  c16_t value0 = (c16_t){0, 0};
  c16_t value1 = (c16_t){16384, 0};

  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < downsampled_OFDM_size / 4; k++) {
      SIP_ideal[i * downsampled_OFDM_size / 4 + k] = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;
    }
  }

  for (int k = 0; k < downsampled_CP_size; k++) {
    SIP_ideal[4 * downsampled_OFDM_size / 4 + k] = value0;
  }

  for (int i = 4; i < 8; i++) {
    for (int k = 0; k < downsampled_OFDM_size / 4; k++) {
      SIP_ideal[downsampled_CP_size + i * downsampled_OFDM_size / 4 + k] =
          R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;
    }
  }

  sprintf(filename, "%s/SIP_Signal.m", foldername);
  LOG_M(filename, "SIP", SIP_ideal, SIP_downsampled_length, 1, 1);

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

  sprintf(filename, "%s/Correlation_Signal.m", foldername);
  LOG_M(filename, "Correlation", correlation, downSampled_length, 1, 7);

  double max_value = 0.0;
  int SIP_offset = 0;
  for (int i = 0; i < downSampled_length - corr_len; i++) {
    if (max_value < correlation[i]) {
      max_value = correlation[i];
      SIP_offset = i;
    }
  }

  printf("SIP offset: %d\n", SIP_offset);

  // Move the start and remove CP (CP0 size = 10, CP size = 9, OFDM symbol size = 128)
  int cleared_length = downSampled_length - SIP_offset;
  int rx_symbols = 0;
  c16_t *cleared = malloc(cleared_length * sizeof(c16_t));
  memset(cleared, 0, cleared_length * sizeof(c16_t));
  for (int i = 0, cp = 0; SIP_offset + (i+1)*downsampled_OFDM_size + cp < downSampled_length; i++)
  {
    if(i != 0) cp += ((i%7) == 0) ? downsampled_CP0_size : downsampled_CP_size;
    for (int j = 0; j < downsampled_OFDM_size; j++) {
      cleared[i*downsampled_OFDM_size + j] = downSampled[SIP_offset + i*downsampled_OFDM_size + j + cp];
    }
    rx_symbols++;
  }

  sprintf(filename, "%s/Cleared_Signal.m", foldername);
  LOG_M(filename, "Cleared", cleared, cleared_length, 1, 1);

  // Calculate energy of chips
  int energy_length = (rx_symbols-2) * frame_parms->M + 2*4; // first 2 symbols are SIP (M=4), rest is decoded with M
  int *energy = malloc(energy_length * sizeof(int));
  memset(energy, 0, energy_length * sizeof(int));
  for (int i = 0; i < energy_length; i++) {
    for (int j = 0; j < (downsampled_OFDM_size / frame_parms->M); j++)
    {
      energy[i] += cleared[i*(downsampled_OFDM_size / frame_parms->M) + j].r;
    }
  }

  sprintf(filename, "%s/Energy_Signal.m", foldername);
  LOG_M(filename, "Energy", energy, energy_length, 1, 2);

  // Find threshold
  int threshold = 0;
  for (int i = R_TAS_SIP_N; i < R_TAS_SIP_N + R_TAS_CAP_N; i++)
  {
    threshold += energy[i];
  }
  threshold /= R_TAS_CAP_N;

  // Decode line encoding
  uint8_t rx_payload[MAX_AIOT_R2D_PACKET_SIZE] = {0};
  uint8_t end = 0;
  printf("Energy threshold: %d\n", threshold);
  printf("Decoding payload\n");

  for (int i = R_TAS_SIP_N + R_TAS_CAP_N, j=0; i < energy_length; i++)
  {
    end = (end << 2) | ((energy[i] > threshold) ? 1 : 0);

    if((end & 0x0F) == 0x0F) {
      printf("\nEnd of payload detected, length: %d\n", j);
      payloadSize = j;
      break;
    }

    if(i%2 != 0) continue;

    //threshold = (threshold*3 + energy[i] + energy[i+1]) / 5;
    int bit0 = (energy[i] > energy[i+1]) ? 1 : 0;
    rx_payload[j/8] = (rx_payload[j/8] << 1) | bit0;
    j++;
  }

  printf("Received payload (%d bits):\n", payloadSize);
  for(int i = 0; i < (payloadSize+7)/8; i++) {
    printf("%02X ", rx_payload[i]);
  }
  printf("\n");

  int BER_errors = 0;
  double BER = 0.0;
  for (int i = 0; i < payloadSize/8; i++) {
    int diff = payload[i] ^ rx_payload[i];

    // Count ones in diff
    for (int b = 0; b < 8; b++) {
      BER_errors += (diff >> b) & 0x01;
    }
  }

  BER = (double)BER_errors / (double)payloadSize;
  printf("BER: %f, Error bits: %d, Payload size: %d\n", 100.0 * BER, BER_errors, payloadSize);

  AIOT_R2D_PHY_REs_free(REsPacket);
  AIOT_R2D_PHY_Signal_free(txData);
  AIOT_R2D_PHY_IQ_free(s_re, s_im);

  free(filteredData);
  free(correlation);
  free(rxData[0]);
  free(rxData);
  free(energy);
  free(cleared);
  free(downSampled);

  for (int i = 0; i < NB_ANTENNAS_RX; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

  free(r_re);
  free(r_im);
  free_channel_desc_scm(channel);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
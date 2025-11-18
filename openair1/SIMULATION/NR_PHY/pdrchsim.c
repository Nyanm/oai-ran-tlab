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
#include "common/utils/time_meas.h"
#include "common/ran_context.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_common.h"

#include <time.h>

#include "PHY_AIOT/defs_aiot_d2r.h"

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, not finished yet */
  return "detect_leaks=0";
}

NR_AIOT_UL_FRAME_PARMS *frame_parms;

double cpuf;
char filename[50];
char foldername[] = "./D2R_results";

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void)
{
  return &softmodem_params;
}

// Default parametrs
#define RBs_DEFAULT 1
#define M_DEFAULT 4
#define ZC_Ones_DEFAULT true
#define PAYLOAD_SIZE_DEFAULT 20 // in bits
#define ANTENNAS_DEFAULT 1
#define SUBCARRIER_SPACING_DEFAULT 15e3
#define OFDM_SYMBOL_SIZE_DEFAULT 2048
#define CP0_SYMBOL_SIZE_DEFAULT 160
#define CP_SYMBOL_SIZE_DEFAULT 144

// Channel model structure
typedef struct {
  SCM_t channel_model;
  uint64_t fc;
  double DS_TDL;
  double SNR;
  double path_loss_dB;
  double noise_power_dB;
  int delay;
  double sampling_rate;
  double bw;
  double tx_pwr_dBm;
} channel_model_t;

#define MIN_SNR_DB (-15)
#define MAX_SNR_DB 10
#define SNR_STEP_DB 1
#define SNR_TRIALS 1000
#define SNR_STEPS ((MAX_SNR_DB - MIN_SNR_DB) / SNR_STEP_DB + 1)

int snr_min = MIN_SNR_DB;
int snr_max = MAX_SNR_DB;
int snr_steps = SNR_STEPS;
int snr_trials = SNR_TRIALS;
int snr_plot = 0;

channel_model_t channel_model;

bool testing_mode = false;
bool testing_timing = false;

void AIOT_D2R_PHY_TX_calc_packet_sizes(const int payloadSize, NR_AIOT_UL_FRAME_PARMS *frame)
{
  const double tau = 2 / 15e3; // in seconds
  const double fsampling = 1.92e6; // sampling frequency in Hz

  // Calculate bit duration
  double T_bit = T_BIT_TABLE[frame->T_bit] * tau; // in seconds
  frame->N_bit = T_bit * fsampling; // number of samples in D2R bit

  // Calculate small frequency shift
  frame->N_SFS = 1 << frame->R_SFS;

  // Calculate preamble size
  frame->N_preamble = (frame->L_preamble) ? 31 : 7;

  // Calculate chip duration
  double T_chip = T_bit / (2 * frame->N_SFS); // in seconds
  frame->N_chip = T_chip * fsampling; // number of samples in D2R chip
  
  // Calculate midamble spacing
  int scale = (int) ((1.0/T_BIT_TABLE[frame->T_bit]) * 2.0);
  frame->N_midamble_space = scale * I_BIT_TABLE[frame->I_bit];

  // Calculate packet size
  frame->payload_size = payloadSize;
  frame->packet_size = frame->payload_size;
  if(frame->I_add) {
    frame->packet_size += (1 + ceil(frame->payload_size / (double)frame->N_midamble_space)) * frame->N_preamble;
  } else {
    frame->packet_size += (1 + (frame->payload_size / frame->N_midamble_space)) * frame->N_preamble;
  }
  frame->packet_size *= frame->N_SFS * 2 * frame->N_chip;

  printf("Calculated chip samples: %d, bit samples: %d, midamble spacing: %d\n", frame->N_chip, frame->N_bit, frame->N_midamble_space);
  printf("Calculated D2R packet size: %d samples\n", frame->packet_size);
}

void AIOT_D2R_PHY_TX_Signal(c16_t *txData, const uint8_t *payload, NR_AIOT_UL_FRAME_PARMS *frame)
{
  if(txData == NULL) {
    printf("Memory allocation failed\n");
    return;
  }

  int dataIndex = 0;

  // Add D-TAS preamble (short or long)
  uint32_t D_TAS_preamble = (frame->L_preamble) ? D_TAS_31_BITS : D_TAS_7_BITS;

  for(int i = 0; i < frame->N_preamble; i++) {
    unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - i)) & 0x01;

    // Prepare bits for Manchester encoding
    const c16_t valueBit0 = {((bit) ? 0 : 32767), 0};
    const c16_t valueBit1 = {((bit) ? 32767 : 0), 0};

    // Repeat each bit N_SFS times with OOK modulation
    for(int j = 0; j < frame->N_SFS; j++) {
      for(int k = 0; k < frame->N_chip; k++) {
        txData[dataIndex++] = valueBit0;
      }
      for(int k = 0; k < frame->N_chip; k++) {
        txData[dataIndex++] = valueBit1;
      }
    }
  }

  // Add payload data
  for(int i = 0; i < frame->payload_size; i++) {
    if(i % frame->N_midamble_space == 0 && i != 0) {
      // Insert midamble (fixed pattern 101010...)
      for(int m = 0; m < frame->N_preamble; m++) {
        unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - m)) & 0x01;

        // Prepare bits for Manchester encoding
        const c16_t valueBit0 = {((bit) ? 0 : 32767), 0};
        const c16_t valueBit1 = {((bit) ? 32767 : 0), 0};

        // Repeat each bit N_SFS times with OOK modulation
        for(int j = 0; j < frame->N_SFS; j++) {
          for(int k = 0; k < frame->N_chip; k++) {
            txData[dataIndex++] = valueBit0;
          }
          for(int k = 0; k < frame->N_chip; k++) {
            txData[dataIndex++] = valueBit1;
          }
        }
      }
    }

    unsigned bit = (payload[i / 8] >> (7 - (i % 8))) & 0x01;

    // Prepare bits for Manchester encoding
    const c16_t valueBit0 = {((bit) ? 0 : 32767), 0};
    const c16_t valueBit1 = {((bit) ? 32767 : 0), 0};

    // Repeat each bit N_SFS times with OOK modulation
    for(int j = 0; j < frame->N_SFS; j++) {
      for(int k = 0; k < frame->N_chip; k++) {
        txData[dataIndex++] = valueBit0;
      }
      for(int k = 0; k < frame->N_chip; k++) {
        txData[dataIndex++] = valueBit1;
      }
    }
  }

  // Insert midamble (fixed pattern 101010...)
  if(frame->I_add) {
    for(int m = 0; m < frame->N_preamble; m++) {
      unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - m)) & 0x01;

      // Prepare bits for Manchester encoding
      const c16_t valueBit0 = {((bit) ? 0 : 32767), 0};
      const c16_t valueBit1 = {((bit) ? 32767 : 0), 0};

      // Repeat each bit N_SFS times with OOK modulation
      for(int j = 0; j < frame->N_SFS; j++) {
        for(int k = 0; k < frame->N_chip; k++) {
          txData[dataIndex++] = valueBit0;
        }
        for(int k = 0; k < frame->N_chip; k++) {
          txData[dataIndex++] = valueBit1;
        }
      }
    }
  }
}


void AIOT_D2R_PHY_TX_Signal_free(c16_t *txData)
{
  if (txData != NULL)
    free(txData);
}

time_stats_t time_stats = {0};

void BER_test(uint8_t *payload, int payloadSize, NR_AIOT_UL_FRAME_PARMS *frame_parms, channel_model_t *channel_model)
{
  AIOT_D2R_PHY_TX_calc_packet_sizes((const int) payloadSize, frame_parms);
  
  printf("D2R packet parameters:\n");
  printf("  RBs: %d\n", frame_parms->nr_frame_parms.N_RB_DL);
  printf("  Bit duration samples: %d\n", frame_parms->N_bit);
  printf("  Chip duration samples: %d\n", frame_parms->N_chip);
  printf("  Small frequency shift factor: %d\n", frame_parms->N_SFS);
  printf("  Preamble length: %s\n", (frame_parms->L_preamble) ? "long (31 bits)" : "short (7 bits)");
  printf("  Additional midamble: %s\n", (frame_parms->I_add) ? "yes" : "no");
  printf("  Midamble interval: %d bits\n", frame_parms->N_midamble_space);
  printf("  Channel coding: %s\n", (frame_parms->R_code) ? "FEC" : "No FEC");
  printf("  Block repetition: %s\n", (frame_parms->R_block) ? "2" : "1");
  printf("  Payload size: %d bits\n", payloadSize);

  printf("Original payload (%d bits):\n", payloadSize);
  for (int i = 0; i < (payloadSize + 7) / 8; i++) {
    printf("%02X ", payload[i]);
  }
  printf("\n");

  // Setup radio channel
  channel_model->sampling_rate = 30.72; // N_RB2sampling_rate(frame_parms->nr_frame_parms.N_RB_DL); // in MHz
  channel_model->bw = frame_parms->nr_frame_parms.N_RB_DL * 0.2/2; // N_RB2channel_bandwidth(frame_parms->nr_frame_parms.N_RB_DL); // in MHz

  printf("Channel parameters:\n");
  printf("  Channel model: %s\n", channel_model->channel_model == AWGN ? "AWGN" : "TDL");
  printf("  Sampling rate: %f MHz\n", channel_model->sampling_rate);
  printf("  Bandwidth: %f MHz\n", channel_model->bw);
  printf("  Carrier frequency: %f MHz\n", ((double)channel_model->fc) / 1e6);
  printf("  Delay spread: %f us\n", channel_model->DS_TDL);
  printf("  SNR: %f dB\n", channel_model->SNR);
  printf("  Delay: %d samples\n", channel_model->delay);
  printf("  Path loss: %f dB\n", channel_model->path_loss_dB);
  printf("  Noise power: %f dB\n", channel_model->noise_power_dB);
  printf("  Transmit power: %f dBm\n", channel_model->tx_pwr_dBm);

  printf("Starting BER test over SNR range %d dB to %d dB with step %d dB (%d trials per SNR)...\n",
         snr_min, snr_max, snr_steps, snr_trials);

  channel_desc_t *channel_params = new_channel_desc_scm(frame_parms->nr_frame_parms.nb_antennas_tx,
                                        frame_parms->nr_frame_parms.nb_antennas_rx,
                                        channel_model->channel_model,
                                        channel_model->sampling_rate,
                                        channel_model->fc,
                                        channel_model->bw,
                                        channel_model->DS_TDL,
                                        0.0,
                                        CORR_LEVEL_LOW,
                                        0,
                                        channel_model->delay,
                                        channel_model->path_loss_dB,
                                        channel_model->noise_power_dB);

  c16_t *txData = NULL;

  txData = malloc(frame_parms->packet_size * sizeof(c16_t));
  memset(txData, 0, frame_parms->packet_size * sizeof(c16_t));

  // TODO: Block repetition
  // TODO: Channel coding

  AIOT_D2R_PHY_TX_Signal(txData, payload, frame_parms);

  sprintf(filename, "%s/D2R_TX_IQ.m", foldername);
  LOG_M(filename, "TX_IQ_sig", txData, frame_parms->packet_size, 1, 1);
  
  AIOT_D2R_PHY_TX_Signal_free(txData);

  free_channel_desc_scm(channel_params);
}

configmodule_interface_t *uniqCfg = NULL;
int main(int argc, char **argv)
{
  stop = false;
  __attribute__((unused)) struct sigaction oldaction;
  sigaction(SIGINT, &sigint_action, &oldaction);

  int loglvl = OAILOG_ERR;

  cpuf = get_cpu_freq_GHz();

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_AIOT_PDRCHSIM] Error, configuration module init failed\n");
  }

  uint8_t payload[MAX_AIOT_D2R_PAYLOAD_SIZE] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
                                                0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
  uint16_t payloadSize = 96;//PAYLOAD_SIZE_DEFAULT; // payload size in bits

  // Allocate memory for frame parameters
  NR_AIOT_UL_FRAME_PARMS frame_parms_storage;
  frame_parms = &frame_parms_storage;
  memset(frame_parms, 0, sizeof(NR_AIOT_UL_FRAME_PARMS));

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
  
  // AIoT D2R specific parameters
  frame_parms->T_bit = 0; // T_bit = 2 * tau
  frame_parms->R_block = false; // R_block = 1
  frame_parms->R_SFS = 0; // N_SFS = 1
  frame_parms->I_bit = 0; // interval I_bit not used in this
  frame_parms->L_preamble = false; // short preamble
  frame_parms->I_add = true; // additional midamble
  frame_parms->R_code = false; // no FEC

  // Fill in channel model default parameters
  channel_model_t channel_model = {
    .channel_model = AWGN,
    .fc = 897500000, // Carrier frequency n8 band, #50 RB
    .DS_TDL = .03,
    .SNR = 20.0,
    .path_loss_dB = 0.0,
    .noise_power_dB = -120.0,
    .delay = 1290,
    .tx_pwr_dBm = 46.0
  };

  int c;
  while ((c = getopt(argc, argv, "--:O:h:L:R:P:p:S:N:D:t:T:")) != -1) {
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
        printf("-p Payload size in bits (max %d)\n", MAX_AIOT_D2R_PAYLOAD_SIZE * 8);
        printf("-S SNR in dB\n");
        printf("-N Path loss in dB\n");
        printf("-D Delay in samples\n");
        printf("-t Testing mode, parameter is SNR to plot\n");
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
        if (payloadSize > MAX_AIOT_D2R_PAYLOAD_SIZE * 8) {
          printf("Error: maximum payload bit size is %d\n", MAX_AIOT_D2R_PAYLOAD_SIZE * 8);
          exit(-1);
        } else {
          printf("Using payload size %d bits\n", payloadSize);
        }
        break;

      case 'S':
        channel_model.SNR = atof(optarg);
        printf("Using SNR=%f dB\n", channel_model.SNR);
        break;

      case 'N':
        channel_model.path_loss_dB = -atof(optarg);
        printf("Using path_loss_dB=%f dB\n", channel_model.path_loss_dB);
        break;

      case 'D':
        channel_model.delay = atoi(optarg);
        printf("Using delay=%d samples\n", channel_model.delay);
        break;

      case 't':
        snr_trials = 1;
        testing_mode = true;
        snr_plot = atoi(optarg);
        break;
      
      case 'T':
        testing_timing = true;
        break;
    }
  }

  logInit();
  set_glog(loglvl);
  cpumeas(CPUMEAS_ENABLE);

  get_softmodem_params()->phy_test = 1;
  get_softmodem_params()->do_ra = 0;
  IS_SOFTMODEM_RFSIM = true;

  randominit(0);
  load_dftslib();
  crcTableInit();
  InitSinLUT();

  // ---------------------------------------------------------------

  BER_test(payload, payloadSize, frame_parms, &channel_model);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
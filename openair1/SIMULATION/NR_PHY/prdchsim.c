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
char foldername[] = "./results";

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void)
{
  return &softmodem_params;
}

// Default parametrs
#define RBs_DEFAULT 1
#define M_DEFAULT 1
#define ZC_Ones_DEFAULT true
#define PAYLOAD_SIZE_DEFAULT 96 // in bits
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
} channel_model_t;

void AIOT_R2D_PHY_TX_calc_packet_sizes(int payloadSize, NR_AIOT_DL_FRAME_PARMS *frame)
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

c16_t *AIOT_R2D_PHY_TX_REs(uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame)
{
  //printf("Creating R2D packet: R-TAS-SIP, R-TAS-CAS, payload, postamble\n");

  // Allocate memory for the whole R2D packet
  c16_t *REsPacket = malloc(frame->packet_symbols * frame->packet_subcarriers * sizeof(c16_t));
  if (REsPacket == NULL) {
    printf("Memory allocation failed\n");
    return NULL;
  }

  // Copy R-TAS SIP preamble to the packet
  const c16_t *SIP_SCs_selection = SIP_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->Zadoff_Chu);
  int SIP_length = R_TAS_SIP_N / R_TAS_SIP_M * frame->packet_subcarriers;
  memcpy(REsPacket, SIP_SCs_selection, SIP_length * sizeof(c16_t)); // SIP

  // Copy R-TAS CAS preamble to the packet
  const c16_t *CAS_SCs_selection = CAS_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int CAS_length = R_TAS_CAP_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length, CAS_SCs_selection, CAS_length * sizeof(c16_t)); // CAS

  // Copy payload to the packet (using Manchester line encoding embedded in the symbol mapping)
  int Payload_length = (frame->packet_encoded_size + (frame->M - 1)) / frame->M * frame->packet_subcarriers;
  
  int M_bits = 1;
  while((1 << M_bits) < frame->M) {
    M_bits++;
  }

  for (int i = 0, j = 0; i < frame->packet_encoded_size / 2; i += M_bits, j++) {
    uint8_t symbol = 0;
    for (int k = 0; k < M_bits; k++) {
      if (i + k < frame->packet_encoded_size / 2) {
        int bit = (payload[(i + k) / 8] >> (7 - ((i + k) % 8))) & 0x01;
        symbol = (symbol << 1) | bit;
      } else {
        symbol = (symbol << 1);
      }
    }

    const c16_t *Payload_SCs_selection = Payload_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, symbol, frame->Zadoff_Chu);
    memcpy(REsPacket + SIP_length + CAS_length + j * frame->packet_subcarriers * ((frame->M == 1) ? 2 : 1),
           Payload_SCs_selection,
           frame->packet_subcarriers * ((frame->M == 1) ? 2 : 1) * sizeof(c16_t));
  }

  // Copy R-TAS postamble
  const c16_t *Postamble_SCs_selection = Postamble_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int Postamble_length = R2D_POSTAMBLE_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length + CAS_length + Payload_length,
         Postamble_SCs_selection,
         Postamble_length * sizeof(c16_t)); // Postamble

  return REsPacket;
}

c16_t *AIOT_R2D_PHY_TX_Signal(c16_t *REsPacket, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Allocate TX buffers
  c16_t *txData = calloc(1, frame->packet_samples * sizeof(c16_t));
  memset(txData, 0, frame->packet_samples * sizeof(c16_t));

  c16_t *txDataF = calloc(1, frame->packet_samples * sizeof(c16_t));
  memset(txDataF, 0, frame->packet_samples * sizeof(c16_t));

  if(txData == NULL || txDataF == NULL) {
    printf("Memory allocation failed\n");
    return NULL;
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

  return txData;
}

c16_t **SIM_Channel_propagate(c16_t *in, channel_model_t *channel, NR_AIOT_DL_FRAME_PARMS *frame)
{
  int rx_size = frame->packet_samples * 2;

  // Initialization of transmit IQ signals
  double **s_re = malloc(frame->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  double **s_im = malloc(frame->nr_frame_parms.nb_antennas_tx * sizeof(double *));

  for (int i = 0; i < frame->nr_frame_parms.nb_antennas_tx; i++) {
    s_re[i] = calloc(1, rx_size * sizeof(double));
    s_im[i] = calloc(1, rx_size * sizeof(double));
  }

  bzero(s_re[0], rx_size * sizeof(double));
  bzero(s_im[0], rx_size * sizeof(double));

  for (int i = 0; i < frame->packet_samples; i++) {
    s_re[0][i] = (double)in[i].r;
    s_im[0][i] = (double)in[i].i;
  }

  // Initialization of receive IQ signals
  double **r_re = malloc(frame->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  double **r_im = malloc(frame->nr_frame_parms.nb_antennas_rx * sizeof(double *));

  for (int i = 0; i < frame->nr_frame_parms.nb_antennas_rx; i++) {
    r_re[i] = calloc(1, rx_size * sizeof(double));
    r_im[i] = calloc(1, rx_size * sizeof(double));
  }

  bzero(r_re[0], rx_size * sizeof(double));
  bzero(r_im[0], rx_size * sizeof(double));

  channel_desc_t *channel_params = new_channel_desc_scm(frame->nr_frame_parms.nb_antennas_tx,
                                                 frame->nr_frame_parms.nb_antennas_rx,
                                                 channel->channel_model,
                                                 channel->sampling_rate,
                                                 channel->fc,
                                                 channel->bw,
                                                 channel->DS_TDL,
                                                 0.0,
                                                 CORR_LEVEL_LOW,
                                                 0,
                                                 channel->delay,
                                                 channel->path_loss_dB,
                                                 channel->noise_power_dB);

  int txlev = signal_energy((int32_t *) in, frame->nr_frame_parms.ofdm_symbol_size + frame->nr_frame_parms.nb_prefix_samples0);
  double txlev_dB = 10 * log10((double)txlev);
  //printf("Signal energy: %d (%f dB)\n", txlev, txlev_dB);

  double ts = 1.0 / (frame->nr_frame_parms.subcarrier_spacing * frame->nr_frame_parms.ofdm_symbol_size);
  // Compute AWGN variance
  double sigma2_dB = 10 * log10((double)txlev) + channel->path_loss_dB - channel->SNR; // *((double)frame_parms->ofdm_symbol_size / r2d_subcarriers)
  double sigma2 = pow(10, sigma2_dB / 10);
  //printf("Noise sigma2: %f (%f dB)\n", sigma2, sigma2_dB);

  c16_t **rxData = malloc(frame->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));

  for (int i = 0; i < frame->nr_frame_parms.nb_antennas_rx; i++) {
    rxData[i] = calloc(1, rx_size * sizeof(c16_t));
    memset(rxData[i], 0, rx_size * sizeof(c16_t));
  }

  // TODO: Transform to fc and back

  multipath_channel(channel_params, s_re, s_im, r_re, r_im, rx_size, 0, 1);
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
            frame->nr_frame_parms.nb_antennas_rx);

  // Save channel output
  double *output = malloc(rx_size * 2 * sizeof(double));

  for (int i = 0; i < rx_size; i++) {
    output[2 * i] = r_re[0][i];
    output[2 * i + 1] = r_im[0][i];
  }

  static int pass = -10;
  if(pass++ == 20) {
    sprintf(filename, "%s/C_Channel.m", foldername);
    LOG_M(filename, "Channel_sig", output, rx_size, 1, 8);
  }

  free(output);

  for (int i = 0; i < frame->nr_frame_parms.nb_antennas_tx; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }

  free(s_re);
  free(s_im);

  for (int i = 0; i < frame->nr_frame_parms.nb_antennas_rx; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

  free(r_re);
  free(r_im);

  free_channel_desc_scm(channel_params);

  return rxData;
}

c16_t *AIOT_R2D_PHY_RX_Envelope_Detector(c16_t **rxData, int rx_size)
{
  c16_t *envelope = malloc(rx_size * sizeof(c16_t *));

  // Envelope detector (squared)
  for (int i = 0; i < rx_size; i++) {
    envelope[i] = c16MulConjShift(rxData[0][i], rxData[0][i], 15);
  }

  return envelope;
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

c16_t *AIOT_R2D_PHY_RX_Filter(c16_t *in, int length)
{
  c16_t *out = malloc(length * sizeof(c16_t));
  iir_state_t st;
  memset(&st, 0, sizeof(iir_state_t));

  // Butterworth filter 3rd order
  butterworth_filter(in, out, length, &st);

  return out;
}

c16_t *AIOT_R2D_PHY_RX_Downsample(c16_t *in, int length, NR_AIOT_DL_FRAME_PARMS *frame)
{
  frame->packet_downsampled_samples = length / frame->N;
  c16_t *out = malloc(frame->packet_downsampled_samples * sizeof(c16_t));

  for (int i = 0; i < frame->packet_downsampled_samples; i++) {
    out[i] = in[i * frame->N];
  }

  return out;
}

int AIOT_R2D_PHY_RX_getSIPindex(c16_t *signal, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Compute lengths for downsampled SIP ideal sequence
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_chip_size = downsampled_OFDM_size / R_TAS_SIP_M;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  int SIP_downsampled_length = downsampled_CP0_size + downsampled_OFDM_size + downsampled_CP_size + downsampled_OFDM_size;

  // Generate ideal SIP sequence for correlation
  c16_t SIP_ideal[SIP_downsampled_length];
  c16_t value0 = (c16_t){-16384, 0};
  //c16_t value0_5 = (c16_t){16384/2, 0};
  c16_t value1 = (c16_t){16384, 0};

  int samples = 0;

  // Add Cyclic prefix (Longer Type 0)
  for (samples = 0; samples < downsampled_CP0_size; samples++) {
    SIP_ideal[samples] = value0;
  }

  // Add first symbol of R-TAS SIP
  for (int i = 0; i < R_TAS_SIP_M; i++) {
    c16_t value = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;

    for (int j = 0; j < downsampled_chip_size; j++) {
      SIP_ideal[samples + j] = value;
    }

    samples += downsampled_chip_size;
  }

  // Add Cyclic prefix (Shorter/Normal Type)
  for (int k = 0; k < downsampled_CP_size; k++) {
    SIP_ideal[samples + k] = value0;
  }
  samples += downsampled_CP_size;

  // Add second symbol of R-TAS SIP
  for (int i = R_TAS_SIP_M; i < R_TAS_SIP_M + R_TAS_SIP_M; i++) {
    c16_t value = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;

    for (int j = 0; j < downsampled_OFDM_size / R_TAS_SIP_M; j++) {
      SIP_ideal[samples + j] = value;
    }

    samples += downsampled_chip_size;
  }

  static int pass = -10;
  if(pass == 20) {
    sprintf(filename, "%s/SIP_Ideal.m", foldername);
    LOG_M(filename, "SIP_Ideal_sig", SIP_ideal, SIP_downsampled_length, 1, 1);
  }

  double *correlation = malloc(frame_parms->packet_downsampled_samples * sizeof(double));
  memset(correlation, 0, frame_parms->packet_downsampled_samples * sizeof(double));

  double max_value = 0.0;
  int SIP_offset = 0;

  for (int i = 0; i < frame_parms->packet_downsampled_samples - SIP_downsampled_length; i++) {
    double sum = 0.0;
    for (int j = 0; j < SIP_downsampled_length; j++) {
      sum += signal[i + j].r * SIP_ideal[j].r;
    }
    
    correlation[i] = sum;

    if (max_value < sum) {
      max_value = sum;
      SIP_offset = i;
    }
  }

  if(pass++ == 20) {
    sprintf(filename, "%s/H_Correlation.m", foldername);
    LOG_M(filename, "Correlation_sig", correlation, frame_parms->packet_downsampled_samples, 1, 7);
  }

  free(correlation);

  // TODO: Get M according to CAS

  return SIP_offset;
}

c16_t *AIOT_R2D_PHY_RX_Move_And_Remove_CP(c16_t *in, int SIP_offset, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Compute lengths for downsampled SIP ideal sequence
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  int cleared_length_samples = frame_parms->packet_downsampled_samples - SIP_offset;
  frame_parms->packet_received_symbols = 0;

  //printf("SIP offset: %d samples\n", SIP_offset);

  // Allocate memory for cleared signal
  c16_t *cleared = malloc(cleared_length_samples * sizeof(c16_t));
  memset(cleared, 0, cleared_length_samples * sizeof(c16_t));

  // Move the start and remove CP (CP0 size = 10, CP size = 9, OFDM symbol size = 128)
  for (int i = 0, cp = 0; SIP_offset + (i+1)*downsampled_OFDM_size + cp < frame_parms->packet_downsampled_samples; i++)
  {
    cp += ((i%7) == 0) ? downsampled_CP0_size : downsampled_CP_size;
    for (int j = 0; j < downsampled_OFDM_size; j++) {
      cleared[i*downsampled_OFDM_size + j] = in[SIP_offset + i*downsampled_OFDM_size + j + cp];
    }
    frame_parms->packet_received_symbols++;
  }

  return cleared;
}

int *AIOT_R2D_PHY_RX_Energy(c16_t *in, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  int downsampled_chip_size = (frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N) / frame_parms->M;
  int downsampled_SIP_chip_size = (frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N) / R_TAS_SIP_M;

  // Calculate energy of chips
  int energy_length = R_TAS_SIP_N + (frame_parms->packet_received_symbols - (R_TAS_SIP_N/R_TAS_SIP_M)) * frame_parms->M;
  int *energy = malloc(energy_length * sizeof(int));
  memset(energy, 0, energy_length * sizeof(int));

  for (int i = 0; i < R_TAS_SIP_N; i++) {
    for (int j = 0; j < downsampled_SIP_chip_size; j++)
    {
      energy[i] += in[i*downsampled_SIP_chip_size + j].r;
    }
  }

  for (int i = 0; i < energy_length - R_TAS_SIP_N; i++) {
    for (int j = 0; j < downsampled_chip_size; j++)
    {
      energy[R_TAS_SIP_N + i] += in[R_TAS_SIP_N*downsampled_SIP_chip_size + i*downsampled_chip_size + j].r;
    }
  }

  frame_parms->packet_payload_size = energy_length;
  return energy;
}

uint8_t *AIOT_R2D_PHY_RX_Decode(int *in, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  /*// Find threshold
  int threshold = 0;
  for (int i = R_TAS_SIP_N; i < R_TAS_SIP_N + R_TAS_CAP_N; i++)
  {
    threshold += in[i];
  }
  threshold /= R_TAS_CAP_N;
  //threshold /= 2; // postamble has 2x lower energy
  printf("Energy threshold: %d\n", threshold);*/

  // Decode line encoding
  uint8_t *rx_payload = malloc(MAX_AIOT_R2D_PACKET_SIZE);
  memset(rx_payload, 0, MAX_AIOT_R2D_PACKET_SIZE);

  uint8_t end = 0;

  for (int i = R_TAS_SIP_N + R_TAS_CAP_N, j=0; i < frame_parms->packet_payload_size; i+=2)
  {
    //end = (end << 2) | ((in[i] >= threshold) ? 2 : 0) | ((in[i+1] >= threshold) ? 1 : 0);
    if(abs(in[i] - in[i+1]) < 600)
      end++;
    else
      end = 0;

    if(end >= 2) { //((end & 0x0F) == 0x0F) {
      //printf("End of payload detected, length: %d\n", j);
      frame_parms->packet_payload_size = j-1; // exclude the last 2 bits (postamble)
      break;
    }

    int bit0 = (in[i] > in[i+1]) ? 1 : 0;
    rx_payload[j/8] = (rx_payload[j/8] << 1) | bit0;
    j++;
  }

  return rx_payload;
}

void AIOT_R2D_PHY_TX_REs_free(c16_t *REsPacket)
{
  if (REsPacket != NULL)
    free(REsPacket);
}

void AIOT_R2D_PHY_TX_Signal_free(c16_t *txData)
{
  if (txData != NULL)
    free(txData);
}

void SIM_Channel_propagate_free(c16_t **rxData, int nb_antennas_rx)
{
  if (rxData != NULL) {
    for (int i = 0; i < nb_antennas_rx; i++) {
      if (rxData[i] != NULL)
        free(rxData[i]);
    }
    free(rxData);
  }
}

void AIOT_R2D_PHY_RX_Downsample_free(c16_t *downSampled)
{
  if (downSampled != NULL)
    free(downSampled);
}

void AIOT_R2D_PHY_RX_Filter_free(c16_t *filteredData)
{
  if (filteredData != NULL)
    free(filteredData);
}

void AIOT_R2D_PHY_RX_Envelope_Detector_free(c16_t *envelope)
{
  if (envelope != NULL)
    free(envelope);
}

void AIOT_R2D_PHY_RX_Move_And_Remove_CP_free(c16_t *cleared)
{
  if (cleared != NULL)
    free(cleared);
}

void AIOT_R2D_PHY_RX_Energy_free(int *energy)
{
  if (energy != NULL)
    free(energy);
}

void AIOT_R2D_PHY_RX_Decode_free(uint8_t *rx_payload)
{
  if (rx_payload != NULL)
    free(rx_payload);
}

double calculate_BER(uint8_t *rx_payload, uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  int BER_errors = 0;
  double BER = 0.0;
  for (int i = 0; i < frame_parms->packet_payload_size/8; i++) {
    int diff = payload[i] ^ rx_payload[i];

    // Count ones in diff
    for (int b = 0; b < 8; b++) {
      BER_errors += (diff >> b) & 0x01;
    }
  }

  BER = (double)BER_errors / (double)frame_parms->packet_payload_size;
  return BER;
}

#define MIN_SNR_DB (-10)
#define MAX_SNR_DB 20
#define SNR_STEP_DB 1
#define SNR_TRIALS 100
#define SNR_STEPS ((MAX_SNR_DB - MIN_SNR_DB) / SNR_STEP_DB + 1)

void BER_test(uint8_t *payload, int payloadSize, NR_AIOT_DL_FRAME_PARMS *frame_parms, channel_model_t *channel_model)
{
  AIOT_R2D_PHY_TX_calc_packet_sizes(payloadSize, frame_parms);

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
  channel_model->sampling_rate = 30.72; // N_RB2sampling_rate(frame_parms->nr_frame_parms.N_RB_DL); // in MHz
  channel_model->bw = frame_parms->nr_frame_parms.N_RB_DL * 0.2; // N_RB2channel_bandwidth(frame_parms->nr_frame_parms.N_RB_DL); // in MHz

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

  c16_t *REsPacket = NULL, *txData = NULL;
  c16_t **rxData, *envelope, *filteredData, *downSampled, *cleared;
  int *energy;
  uint8_t *rx_payload;

  double ber_results[SNR_STEPS] = {0.0};

  for(int snr = MIN_SNR_DB; snr <= MAX_SNR_DB; snr += SNR_STEP_DB) {
    channel_model->SNR = snr;

    for(int trials = 0; trials < SNR_TRIALS; trials++) {
      for(int i = 0; i < payloadSize / 8; i++) {
        payload[i] = uniformrandom() * 256;
      }
      
      REsPacket = AIOT_R2D_PHY_TX_REs(payload, frame_parms);

      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/A_REs_Packet.m", foldername);
        LOG_M(filename, "REs_Packet_sig", REsPacket, frame_parms->packet_symbols * frame_parms->packet_subcarriers, 1, 1);
      }

      txData = AIOT_R2D_PHY_TX_Signal(REsPacket, frame_parms);

      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/B_TX_IQ.m", foldername);
        LOG_M(filename, "TX_IQ_sig", txData, frame_parms->packet_samples, 1, 1);
      }

      int rx_size = frame_parms->packet_samples * 2;

      rxData = SIM_Channel_propagate(txData, channel_model, frame_parms);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/D_RX_IQ.m", foldername);
        LOG_M(filename, "RX_IQ_sig", rxData[0], rx_size, 1, 1);
      }

      // Envelope detector (squared)
      envelope = AIOT_R2D_PHY_RX_Envelope_Detector(rxData, rx_size);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/E_Envelope.m", foldername);
        LOG_M(filename, "Envelope_sig", envelope, rx_size, 1, 1);
      }

      // Low-pass filter
      filteredData = AIOT_R2D_PHY_RX_Filter(envelope, rx_size);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/F_Filter.m", foldername);
        LOG_M(filename, "Filter_sig", filteredData, rx_size, 1, 1);
      }

      // Downsample the signal by N = 16
      frame_parms->N = 16;
      downSampled = AIOT_R2D_PHY_RX_Downsample(filteredData, rx_size, frame_parms);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/G_Downsampled.m", foldername);
        LOG_M(filename, "Downsampled_sig", downSampled, frame_parms->packet_downsampled_samples, 1, 1);
      }

      // Correlate the received signal with known SIP sequence
      int SIP_offset = AIOT_R2D_PHY_RX_getSIPindex(downSampled, frame_parms);

      cleared = AIOT_R2D_PHY_RX_Move_And_Remove_CP(downSampled, SIP_offset, frame_parms);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/I_Cleared.m", foldername);
        LOG_M(filename, "Cleared_sig", cleared, frame_parms->packet_received_symbols * frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N, 1, 1);
      }

      // Calculate energy of chips
      energy = AIOT_R2D_PHY_RX_Energy(cleared, frame_parms);
      if(snr == MAX_SNR_DB && trials == 0) {
        sprintf(filename, "%s/J_Energy.m", foldername);
        LOG_M(filename, "Energy_sig", energy, R_TAS_SIP_N + (frame_parms->packet_received_symbols-2) * frame_parms->M, 1, 2);
      }

      // Decode line encoding
      rx_payload = AIOT_R2D_PHY_RX_Decode(energy, frame_parms);

      double ber = calculate_BER(rx_payload, payload, frame_parms);
      if(frame_parms->packet_payload_size != payloadSize || (ber > 0.0)) {
        ber_results[snr - MIN_SNR_DB] += 1;

        /*printf("-------------------------------\n");
        printf("Trial %d/%d for SNR %d dB\n", trials + 1, SNR_TRIALS, snr);
        printf("Payload: \n");
        for(int i = 0; i < payloadSize / 8; i++) {
          printf("%02X ", payload[i]);
        }
        printf("\n");

        printf("Received payload (%d bits):\n", frame_parms->packet_payload_size);
        for(int i = 0; i < (frame_parms->packet_payload_size+7)/8; i++) {
          printf("%02X ", rx_payload[i]);
        }
        printf("\n");
        printf("BER: %f, Payload size: %d\n", ber, frame_parms->packet_payload_size);*/
      }

      AIOT_R2D_PHY_TX_REs_free(REsPacket);
      AIOT_R2D_PHY_TX_Signal_free(txData);
      SIM_Channel_propagate_free(rxData, frame_parms->nr_frame_parms.nb_antennas_rx);
      AIOT_R2D_PHY_RX_Envelope_Detector_free(envelope);
      AIOT_R2D_PHY_RX_Filter_free(filteredData);
      AIOT_R2D_PHY_RX_Downsample_free(downSampled);
      AIOT_R2D_PHY_RX_Move_And_Remove_CP_free(cleared);
      AIOT_R2D_PHY_RX_Energy_free(energy);
      AIOT_R2D_PHY_RX_Decode_free(rx_payload);
    }

    printf("-------------------------------\n");
    ber_results[snr - MIN_SNR_DB] /= SNR_TRIALS;
    printf("Completed SNR %d dB: BLER = %f\n", snr, ber_results[snr - MIN_SNR_DB]);
  }

  for(int snr = MIN_SNR_DB; snr <= MAX_SNR_DB; snr += SNR_STEP_DB) {
    printf("SNR %d dB: BER = %f\n", snr, ber_results[snr - MIN_SNR_DB]);
  }

  sprintf(filename, "%s/BLER_M%d_%s_SIZE%d_%dRBs.m", foldername, frame_parms->M, frame_parms->Zadoff_Chu ? "ZC" : "Ones", payloadSize, frame_parms->nr_frame_parms.N_RB_DL);
  LOG_M(filename, "BLER", ber_results, MAX_SNR_DB - MIN_SNR_DB + 1, 1, 7);
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

  // Fill in channel model default parameters
  channel_model_t channel_model = {
    .channel_model = AWGN,
    .fc = 897500000, // Carrier frequency n8 band, #50 RB
    .DS_TDL = .03,
    .SNR = 20.0,
    .path_loss_dB = -40.0,
    .noise_power_dB = -140.0,
    .delay = 1290
  };

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

  BER_test(payload, payloadSize, frame_parms, &channel_model);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
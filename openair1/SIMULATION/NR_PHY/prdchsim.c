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

double **s_re, **s_im; // TX signal
double **r_re, **r_im; // RX signal
channel_model_t channel_model;

bool testing_mode = false;
bool testing_timing = true;

void AIOT_R2D_PHY_TX_calc_packet_sizes(const int payloadSize, NR_AIOT_DL_FRAME_PARMS *frame)
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

void AIOT_R2D_PHY_TX_REs(c16_t *REsPacket, const uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame)
{
  //printf("Creating R2D packet: R-TAS-SIP, R-TAS-CAP, payload, postamble\n");

  // Copy R-TAS SIP preamble to the packet
  const c16_t *SIP_SCs_selection = SIP_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->Zadoff_Chu);
  int SIP_length = R_TAS_SIP_N / R_TAS_SIP_M * frame->packet_subcarriers;
  memcpy(REsPacket, SIP_SCs_selection, SIP_length * sizeof(c16_t)); // SIP

  // Copy R-TAS CAP preamble to the packet
  const c16_t *CAP_SCs_selection = CAP_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int CAP_length = R_TAS_CAP_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length, CAP_SCs_selection, CAP_length * sizeof(c16_t)); // CAP

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
    memcpy(REsPacket + SIP_length + CAP_length + j * frame->packet_subcarriers * ((frame->M == 1) ? 2 : 1),
           Payload_SCs_selection,
           frame->packet_subcarriers * ((frame->M == 1) ? 2 : 1) * sizeof(c16_t));
  }

  // Copy R-TAS postamble
  const c16_t *Postamble_SCs_selection = Postamble_SCs_select(frame->nr_frame_parms.N_RB_DL, frame->M, frame->Zadoff_Chu);
  int Postamble_length = R2D_POSTAMBLE_N / frame->M * frame->packet_subcarriers;
  memcpy(REsPacket + SIP_length + CAP_length + Payload_length,
         Postamble_SCs_selection,
         Postamble_length * sizeof(c16_t)); // Postamble
}

void AIOT_R2D_PHY_TX_Signal(c16_t *txData, const c16_t *REsPacket, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Allocate TX buffers
  c16_t *txDataF = calloc(1, frame->packet_samples * sizeof(c16_t));
  memset(txDataF, 0, frame->packet_samples * sizeof(c16_t));

  if(txData == NULL || txDataF == NULL) {
    printf("Memory allocation failed\n");
    return;
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
}

void SIM_Channel_propagate(c16_t **rxData, const c16_t *in, channel_model_t *channel, NR_AIOT_DL_FRAME_PARMS *frame)
{
  int rx_size = frame->packet_samples * 2;

  for (int i = 0; i < frame->packet_samples; i++) {
    s_re[0][i] = (double) in[i].r;
    s_im[0][i] = (double) in[i].i;
  }

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
  double txlev_dBm = 10 * log10((double)txlev);
  //printf("Signal energy: %d (%f dB)\n", txlev, txlev_dBm);

  double ts = 1.0 / (frame->nr_frame_parms.subcarrier_spacing * frame->nr_frame_parms.ofdm_symbol_size);
  // Compute AWGN variance
  double sigma2_dBm = txlev_dBm + channel->path_loss_dB - channel->SNR; // *((double)frame_parms->ofdm_symbol_size / r2d_subcarriers)
  double sigma2 = pow(10, sigma2_dBm / 10);
  //printf("Noise sigma2: %f (%f dB)\n", sigma2, sigma2_dBm);

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

  static int pass = MIN_SNR_DB;
  if(testing_mode && pass++ == snr_plot) {
    // Save channel output
    double *output = malloc(rx_size * 2 * sizeof(double));

    for (int i = 0; i < rx_size; i++) {
      output[2 * i] = r_re[0][i];
      output[2 * i + 1] = r_im[0][i];
    }
    
    sprintf(filename, "%s/C_Channel.m", foldername);
    LOG_M(filename, "Channel_sig", output, rx_size, 1, 8);

    free(output);
  }
}

void AIOT_R2D_PHY_RX_Envelope_Detector(uint16_t *envelope, const c16_t **rxData, int rx_size)
{
  // Envelope detector (squared)
  for (int i = 0; i < rx_size; i++) {
    //envelope[i] = iSqrt(rxData[0][i].r * rxData[0][i].r + rxData[0][i].i * rxData[0][i].i);

    // 1. Calculate the absolute values (still Q1.15).
    int16_t abs_I = abs(rxData[0][i].r);
    int16_t abs_Q = abs(rxData[0][i].i);

    int16_t max_val;
    int16_t min_val;

    // 2. Determine Max and Min
    if (abs_I > abs_Q) {
        max_val = abs_I;
        min_val = abs_Q;
    } else {
        max_val = abs_Q;
        min_val = abs_I;
    }

    // 3. Calculate: Magnitude ≈ (1 * max_val) + (1/4 * min_val)
    
    // Multiplication by 1/4 is a right shift by 2 (>> 2).
    // The result 'beta_min' is still in Q1.15.
    int16_t beta_min = min_val >> 2; 

    // 4. Final Addition
    envelope[i] = max_val + beta_min;
  }
}

// Filter state
typedef struct {
  double xr[4]; // past real inputs
  double xi[4]; // past imag inputs
  double yr[4]; // past real outputs
  double yi[4]; // past imag outputs
} iir_state_t;

/**
 * @brief Design a 3rd-order (order-3) digital low-pass Butterworth IIR filter.
 *
 * This function computes a 3rd-order Butterworth low-pass filter by first
 * prewarping the desired digital cutoff frequency (fc) to its analog
 * equivalent (wc = 2*fs*tan(pi*fc/fs)), forming the normalized 3rd-order
 * Butterworth prototype, and then applying the bilinear transform
 * (k = 2*fs) to obtain polynomials in x = z^{-1}. The resulting denominator is normalized so that
 * a[0] == 1.0 and the numerator is scaled so the DC gain (sum(b)/sum(a)) == 1.0.
 *
 * @param[in]  fs     Sampling frequency in Hz (must be > 0).
 * @param[in]  fc     Digital cutoff frequency in Hz (must satisfy 0 < fc < fs/2).
 * @param[out] out_b  Output array of 4 numerator coefficients {b0, b1, b2, b3}
 *                    corresponding to H(z) numerator b0 + b1 z^{-1} + b2 z^{-2} + b3 z^{-3}.
 * @param[out] out_a  Output array of 4 denominator coefficients {a0, a1, a2, a3}
 *                    corresponding to H(z) denominator a0 + a1 z^{-1} + a2 z^{-2} + a3 z^{-3}.
 *                    Coefficients are normalized so that a0 == 1.
 *
 * @return 0 on success; -1 on invalid input (null pointers, out-of-range frequencies)
 *         or numerical failure (extremely small intermediate values).
 *
 * @note The routine performs checks for fs, fc and for very small intermediate
 *       values (to avoid division by zero). The arrays out_b and out_a must
 *       point to at least 4 contiguous doubles each.
 */
int butterworth3_design(double fs, double fc, double out_b[4], double out_a[4])
{
  if (!out_b || !out_a || fs <= 0.0 || fc <= 0.0 || fc >= 0.5 * fs)
    return -1;

  /* Prewarp cutoff (analog) */
  double wc = 2.0 * fs * tan(M_PI * fc / fs); /* rad/s analog cutoff after prewarp */

  /* Prototype (normalized) coefficients for 3rd-order Butterworth:
     A_norm(s) = s^3 + 2 s^2 + 2 s + 1
     Scale for cutoff wc: A(s) = s^3 + 2*wc*s^2 + 2*wc^2*s + wc^3
  */
  double a3 = 1.0;
  double a2 = 2.0 * wc;
  double a1 = 2.0 * wc * wc;
  double a0 = wc * wc * wc;

  /* Bilinear transform constant */
  double k = 2.0 * fs;
  double k3 = k * k * k;
  double k2 = k * k;
  double k1 = k;

  /* Pre-expanded polynomials in x = z^-1 (coeff order: x^0 .. x^3):
     p1 = (1-x)^3         = 1 - 3x + 3x^2 - x^3
     p2 = (1-x)^2*(1+x)   = 1 -  x -  x^2 + x^3
     p3 = (1-x)*(1+x)^2   = 1 +  x -  x^2 - x^3
     p4 = (1+x)^3         = 1 + 3x + 3x^2 + x^3
  */
  const double p1[4] = {1.0, -3.0,  3.0, -1.0};
  const double p2[4] = {1.0, -1.0, -1.0,  1.0};
  const double p3[4] = {1.0,  1.0, -1.0, -1.0};
  const double p4[4] = {1.0,  3.0,  3.0,  1.0};

  double D[4] = {0.0, 0.0, 0.0, 0.0};
  double B[4] = {0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < 4; i++) {
    D[i] = a3 * k3 * p1[i] + a2 * k2 * p2[i] + a1 * k1 * p3[i] + a0 * p4[i];
    B[i] = a0 * p4[i];
  }

  /* Normalize so that a[0] == 1.0 (MATLAB convention) */
  if (fabs(D[0]) < 1e-300)
    return -1;

  for (int i = 0; i < 4; i++) {
    out_a[i] = D[i] / D[0];
    out_b[i] = B[i] / D[0];
  }

  /* Scale numerator so DC gain == 1.0 (sum(b)/sum(a) == 1) */
  double sum_a = out_a[0] + out_a[1] + out_a[2] + out_a[3];
  double sum_b = out_b[0] + out_b[1] + out_b[2] + out_b[3];

  if (fabs(sum_a) < 1e-300 || fabs(sum_b) < 1e-300)
    return -1;

  double dc_gain = sum_b / sum_a;
  if (dc_gain == 0.0)
    return -1;

  for (int i = 0; i < 4; i++)
    out_b[i] /= dc_gain;

  return 0;
}

/* Filter real-valued sequences (Direct Form II Transposed 3rd-order Butterworth).
 * Input and output are uint16_t. Internally computations use double.
 *
 */
static void butterworth_filter(uint16_t *out, const uint16_t *in, int length, iir_state_t *st, double *b, double *a)
{
  if (!in || !out || !st || length <= 0)
    return;

  /* Use st->xr[0..2] as DF-II-T states for the real filter (s0,s1,s2).
     The state values are stored in the same floating domain as 'x' (i.e. [-1..1]). */
  for (int n = 0; n < length; n++) {
    /* convert Q1.15 input to double */
    double x = (double)in[n] / UINT16_MAX;

    double s0 = st->xr[0];
    double s1 = st->xr[1];
    double s2 = st->xr[2];

    /* DF-II-T output (double domain) */
    double y = b[0] * x + s0;

    /* update states */
    double ns0 = b[1] * x - a[1] * y + s1;
    double ns1 = b[2] * x - a[2] * y + s2;
    double ns2 = b[3] * x - a[3] * y;

    st->xr[0] = ns0;
    st->xr[1] = ns1;
    st->xr[2] = ns2;

    /* convert back to Q1.15 int16_t with rounding and clamping */
    long tmp = lround(y * UINT16_MAX); /* use 32767 for symmetric positive scaling */
    if (tmp > UINT32_MAX)
      tmp = UINT32_MAX;
    if (tmp < 0)
      tmp = 0;
    out[n] = (uint16_t)tmp;
  }
}

void AIOT_R2D_PHY_RX_Filter(uint16_t *out, const uint16_t *in, int length, double *b, double *a)
{
  iir_state_t st = {0};

  // Butterworth filter 3rd order
  butterworth_filter(out, in, length, &st, b, a);
}

void AIOT_R2D_PHY_RX_Downsample(uint16_t *out, const uint16_t *in, int length, NR_AIOT_DL_FRAME_PARMS *frame)
{
  for (int i = 0; i < frame->packet_downsampled_samples; i++) {
    out[i] = in[i * frame->N];
  }
}

double *generate_SIP_ideal_sequence(NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Compute lengths for downsampled SIP ideal sequence;
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_chip_size = downsampled_OFDM_size / R_TAS_SIP_M;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  int SIP_downsampled_length = downsampled_CP0_size + downsampled_OFDM_size + downsampled_CP_size + downsampled_OFDM_size;

  // Generate ideal SIP sequence for correlation
  double *SIP_ideal = malloc(SIP_downsampled_length * sizeof(double));
  double value_minus_1 = -1.5;
  double value0 = 0.0;
  double value1 = 1.0;

  int samples = 0;

  // Add Cyclic prefix (Longer Type 0)
  for (samples = 0; samples < downsampled_CP0_size; samples++) {
    SIP_ideal[samples] = value0;
  }

  // Add first symbol of R-TAS SIP
  for (int i = 0; i < R_TAS_SIP_M; i++) {
    int16_t value = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;

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
    int16_t value = R_TAS_SIP & (1 << (7 - i)) ? value1 : value0;

    if(i == R_TAS_SIP_M + 1) {
      value = value_minus_1; // Second chip of second symbol is always -1 (fix: mitigate similarity of SIP and M2 1010 sequence)
    }

    for (int j = 0; j < downsampled_OFDM_size / R_TAS_SIP_M; j++) {
      SIP_ideal[samples + j] = value;
    }

    samples += downsampled_chip_size;
  }

  // Find mean of the SIP signal
  double SIP_mean = 0;
  for (int i = 0; i < SIP_downsampled_length; i++) {
    SIP_mean += SIP_ideal[i];
  }
  SIP_mean /= SIP_downsampled_length;

  // Remove DC component
  for (int i = 0; i < SIP_downsampled_length; i++) {
    SIP_ideal[i] -= SIP_mean;
  }

  return SIP_ideal;
}

void free_SIP_ideal_sequence(double *SIP_ideal)
{
  if(SIP_ideal) {
    free(SIP_ideal);
  }
}

#define CORR_THRESHOLD (frame_parms->Zadoff_Chu ? 5000 : 500)

void AIOT_R2D_PHY_RX_GetPacket(uint8_t *rx_payload, const uint16_t *signal, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  double *SIP_ideal = generate_SIP_ideal_sequence(frame_parms);

  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  int SIP_downsampled_length = downsampled_CP0_size + downsampled_OFDM_size + downsampled_CP_size + downsampled_OFDM_size;
  int M4_chip_size = downsampled_OFDM_size / R_TAS_SIP_M;
  frame_parms->packet_payload_size = frame_parms->packet_downsampled_samples - SIP_downsampled_length; // Max payload size in bits

  // Correlate received signal with ideal SIP
  int SIP_offset = 0;
  double max_corr = 0;

  for (int i = SIP_downsampled_length; i < frame_parms->packet_downsampled_samples; i++) {
    double sum = 0;
    for (int j = 0; j < SIP_downsampled_length; j++) {
      sum += signal[i + j - SIP_downsampled_length] * SIP_ideal[j];
    }

    if (sum > CORR_THRESHOLD) { // Threshold to detect SIP presence
      if(sum > max_corr) {
        max_corr = sum;
        SIP_offset = i - SIP_downsampled_length;
      }
    } else {
      if(SIP_offset != 0) {
        break;
      }
    }
  }

  if(testing_mode) {
    printf("Detected SIP at offset %d\n", SIP_offset);

    // Correlate received signal with ideal SIP
    double *correlation = malloc((frame_parms->packet_downsampled_samples - SIP_downsampled_length) * sizeof(double));
    memset(correlation, 0, (frame_parms->packet_downsampled_samples - SIP_downsampled_length) * sizeof(double));

    for (int i = 0; i < frame_parms->packet_downsampled_samples - SIP_downsampled_length; i++) {
      for (int j = 0; j < SIP_downsampled_length; j++) {
        correlation[i] += signal[i + j] * SIP_ideal[j];
      }
    }

    static int pass = MIN_SNR_DB;
    if(testing_mode && pass++ == snr_plot) {
      sprintf(filename, "%s/H_Correlation.m", foldername);
      LOG_M(filename, "Correlation_sig", correlation, frame_parms->packet_downsampled_samples - SIP_downsampled_length, 1, 7);
    }

    free(correlation);
  }

  int CAP_energy[R_TAS_CAP_N] = {0};
  int SIP_bit0_energy = 0;

  // Get energy of last bit0 of SIP (its zero)
  for (int j = 0; j < M4_chip_size; j++)
  {
    SIP_bit0_energy += signal[SIP_offset + (SIP_downsampled_length - M4_chip_size) + j];
  }

  int position = SIP_offset + SIP_downsampled_length + downsampled_CP_size; // Start of R-TAS-CAP (after CP)

  // Get energy of three chips of R-TAS-CAP in M=4
  for (int i = 0; i < R_TAS_CAP_N; i++) {
    for (int j = 0; j < M4_chip_size; j++)
    {
      CAP_energy[i] += signal[position + i*M4_chip_size + j];
    }
  }

  // Compute initial threshold for M detection
  uint32_t thr_max = CAP_energy[0]; // first bit of CAP is always 1
  uint32_t thr_min = SIP_bit0_energy; // last bit of SIP is always 0
  uint32_t threshold = ((thr_max + thr_min) / 2); // decrease threshold to be effective for lower amplitudes

  if(testing_mode) {
    printf("CAP threshold: %d\n", threshold);
  }

  // Find parameter M from R-TAS-CAP sequence
  if(CAP_energy[1] < threshold) {
    frame_parms->received_M = 4;

  } else if(CAP_energy[2] < threshold) {
    frame_parms->received_M = 2;

  } else { // default M=1
    frame_parms->received_M = 1;
  }

  if(testing_mode) {
    printf("Detected M=%d from R-TAS-CAP\n", frame_parms->received_M);
    printf("CAP energies: %d %d %d %d\n", CAP_energy[0], CAP_energy[1], CAP_energy[2], CAP_energy[3]);
  }

  int received_chip_size = downsampled_OFDM_size / frame_parms->received_M;

  // Calculate energies of CAP (in actual M) and adapt threshold
  for (int i = 0; i < R_TAS_CAP_N; i++) {
    CAP_energy[i] = 0;
    for (int j = 0; j < received_chip_size; j++)
    {
      CAP_energy[i] += signal[position + i*received_chip_size + j];
    }
  }

  // Calculate actual initial threshold for postamble detection from CAP
  thr_max = (CAP_energy[0] + CAP_energy[2]) / 2;
  thr_min = (CAP_energy[1] + CAP_energy[3]) / 2;

  threshold = ((thr_max + thr_min) / 2); // adapt threshold
  if(testing_mode) {
    printf("Postamble detection threshold: %d\n", threshold);
  }

  // Calculate position of the start of payload
  int CAP_symbols = R_TAS_CAP_N / frame_parms->received_M;
  position += (CAP_symbols * downsampled_OFDM_size) + ((CAP_symbols - 1) * downsampled_CP_size); // Move to payload start (before CP)

  // Energy plotting
  uint32_t energy_plot[frame_parms->packet_payload_size];
  memset(energy_plot, 0, frame_parms->packet_payload_size * sizeof(uint32_t));
  int energy_index = 0;

  // Threshold plotting
  uint32_t thr_plot[frame_parms->packet_payload_size];
  memset(thr_plot, 0, frame_parms->packet_payload_size * sizeof(uint32_t));
  int thr_index = 0;

  int symbolCounter = (R_TAS_SIP_N / R_TAS_SIP_M) + (R_TAS_CAP_N / frame_parms->received_M);
  int endCounter = 0;
  int bits = 0;
  int chip = 0;
  uint32_t energy[2] = {0};
  frame_parms->packet_payload_size = 0;

  // Threshold for next bit
  thr_plot[thr_index++] = threshold;

  while(position + 2*received_chip_size < frame_parms->packet_downsampled_samples)
  {
    // Add CP if needed
    if((chip % frame_parms->received_M) == 0) {
      if((symbolCounter++ % 7) == 0) {
        position += downsampled_CP0_size;
      } else {
        position += downsampled_CP_size;
      }
    }

    // Calculate energy of the current chip
    for (int j = 0; j < received_chip_size; j++)
    {
      energy[chip%2] += signal[position + j];
    }

    // Save energy for plotting
    energy_plot[energy_index++] = energy[chip%2];

    // Move to next chip
    position += received_chip_size;

    if((++chip % 2) == 1) {
      continue; // wait for two chips to form a bit
    }

    if(energy[0] > threshold && energy[1] > threshold) {
      if(++endCounter >= 2) {
        frame_parms->packet_payload_size = bits-1; // exclude the last bit (postamble)
        break;
      }
    } else {
      endCounter = 0;
    }

    int bit0 = (energy[0] > energy[1]) ? 1 : 0;
    rx_payload[bits/8] = (rx_payload[bits/8] << 1) | bit0;
    bits++;

    if(bit0 == 0) {
      thr_max = (thr_max*3 + energy[1]) / 4;
      thr_min = (thr_min*3 + energy[0]) / 4;
    } else {
      thr_max = (thr_max*3 + energy[0]) / 4;
      thr_min = (thr_min*3 + energy[1]) / 4;
    }

    threshold = ((thr_max + thr_min) / 2);// - ((thr_max - thr_min) / 4); // adapt threshold
    thr_plot[thr_index++] = threshold;

    energy[0] = 0;
    energy[1] = 0;
  }

  static int pass = MIN_SNR_DB;
  if(testing_mode && pass++ == snr_plot) {
    sprintf(filename, "%s/Adaptive_threshold.m", foldername);
    LOG_M(filename, "Adaptive_threshold_sig", thr_plot, thr_index, 1, 2);

    sprintf(filename, "%s/J_Energy.m", foldername);
    LOG_M(filename, "Energy_sig", energy_plot, energy_index, 1, 2);
  }

  free_SIP_ideal_sequence(SIP_ideal);
}

/*void AIOT_R2D_PHY_RX_Move_And_Remove_CP(uint16_t *cleared, const uint16_t *in, int SIP_offset, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Compute lengths for downsampled SIP ideal sequence
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  frame_parms->packet_received_symbols = 0;

  if(testing_mode) printf("SIP offset: %d samples\n", SIP_offset);

  // Move the start and remove CP (CP0 size = 10, CP size = 9, OFDM symbol size = 128)
  for (int i = 0, cp = 0; SIP_offset + (i+1)*downsampled_OFDM_size + cp < frame_parms->packet_downsampled_samples; i++)
  {
    cp += ((i%7) == 0) ? downsampled_CP0_size : downsampled_CP_size;
    for (int j = 0; j < downsampled_OFDM_size; j++) {
      cleared[i*downsampled_OFDM_size + j] = in[SIP_offset + i*downsampled_OFDM_size + j + cp];
    }
    frame_parms->packet_received_symbols++;
  }
}

void AIOT_R2D_PHY_RX_Energy(uint32_t* energy, const uint16_t *in, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  int downsampled_chip_size = (frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N) / frame_parms->M;
  int downsampled_SIP_chip_size = (frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N) / R_TAS_SIP_M;
  int energy_length = R_TAS_SIP_N + (frame_parms->packet_received_symbols - (R_TAS_SIP_N/R_TAS_SIP_M)) * frame_parms->M;
  memset(energy, 0, energy_length * sizeof(uint32_t));

  for (int i = 0; i < R_TAS_SIP_N; i++) {
    for (int j = 0; j < downsampled_SIP_chip_size; j++)
    {
      energy[i] += in[i*downsampled_SIP_chip_size + j];
    }
  }

  for (int i = 0; i < energy_length - R_TAS_SIP_N; i++) {
    for (int j = 0; j < downsampled_chip_size; j++)
    {
      energy[R_TAS_SIP_N + i] += in[R_TAS_SIP_N*downsampled_SIP_chip_size + i*downsampled_chip_size + j];
    }
  }

  frame_parms->packet_payload_size = energy_length;
}

void AIOT_R2D_PHY_RX_Decode(uint8_t *rx_payload, const uint32_t *in, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Find threshold
  uint32_t thr_max = (in[8] + in[10]) / 2;
  uint32_t thr_min = (in[9] + in[11]) / 2;
  uint32_t threshold = ((thr_max + thr_min) / 2) - ((thr_max - thr_min) / 3); // decrease threshold to be effective for the power level of postamble

  uint32_t thr_plot[frame_parms->packet_payload_size];
  memset(thr_plot, 0, frame_parms->packet_payload_size * sizeof(uint32_t));
  int thr_index = 0;
  thr_plot[thr_index++] = threshold;
  
  if(testing_mode) printf("Threshold: %d\n", threshold);

  int endCounter = 0;

  for (int i = R_TAS_SIP_N + R_TAS_CAP_N, j=0; i < frame_parms->packet_payload_size; i+=2)
  {
    if(in[i] > threshold && in[i+1] > threshold) {
      if(++endCounter >= 2) {
        frame_parms->packet_payload_size = j-1; // exclude the last 2 bits (postamble)
        break;
      }
    } else {
      endCounter = 0;
    }

    int bit0 = (in[i] > in[i+1]) ? 1 : 0;
    rx_payload[j/8] = (rx_payload[j/8] << 1) | bit0;
    j++;

    if(bit0 == 0) {
      thr_max = (thr_max*3 + in[i+1]) / 4;
      thr_min = (thr_min*3 + in[i]) / 4;
    } else {
      thr_max = (thr_max*3 + in[i]) / 4;
      thr_min = (thr_min*3 + in[i+1]) / 4;
    }

    threshold = ((thr_max + thr_min) / 2) - ((thr_max - thr_min) / 3); // adapt threshold
    thr_plot[thr_index++] = threshold;
  }

  static int pass = MIN_SNR_DB;
  if(testing_mode && pass++ == snr_plot) {
    sprintf(filename, "%s/Adaptive_threshold.m", foldername);
    LOG_M(filename, "Adaptive_threshold_sig", thr_plot, thr_index, 1, 2);
  }
}*/

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

void AIOT_R2D_PHY_RX_Downsample_free(uint16_t *downSampled)
{
  if (downSampled != NULL)
    free(downSampled);
}

void AIOT_R2D_PHY_RX_Filter_free(uint16_t *filteredData)
{
  if (filteredData != NULL)
    free(filteredData);
}

void AIOT_R2D_PHY_RX_Envelope_Detector_free(uint16_t *envelope)
{
  if (envelope != NULL)
    free(envelope);
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

static inline int64_t time_start(void) {
    struct timespec ts;
    /* use CLOCK_MONOTONIC or CLOCK_MONOTONIC_RAW if available */
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1;
    return (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000);
}

static inline double time_elapsed_ms(int64_t start_ms) {
    int64_t now = time_start();
    if (now < 0 || start_ms < 0) return -1.0;
    return (double)(now - start_ms);
}

time_stats_t time_stats = {0};

void BER_test(uint8_t *payload, int payloadSize, NR_AIOT_DL_FRAME_PARMS *frame_parms, channel_model_t *channel_model)
{
  AIOT_R2D_PHY_TX_calc_packet_sizes((const int) payloadSize, frame_parms);

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

  c16_t *REsPacket = NULL, *txData = NULL;
  c16_t **rxData;
  uint16_t *envelope, *filteredData, *downSampled, *cleared;
  uint32_t *energy;
  uint8_t *rx_payload;
  int rx_size = frame_parms->packet_samples * 2;

  // Butterworth 3rd-order LPF coefficients
  double b[4] = { 0 };
  double a[4] = { 0 };

  butterworth3_design((double)channel_model->sampling_rate * 1e6, channel_model->bw * 1e6, b, a);

  printf("Butterworth 3rd-order LPF coefficients:\n");
  printf("b : %f, %f, %f, %f\n", b[0], b[1], b[2], b[3]);
  printf("a : %f, %f, %f, %f\n", a[0], a[1], a[2], a[3]);

  // Allocate memory for the whole R2D packet
  REsPacket = malloc(frame_parms->packet_symbols * frame_parms->packet_subcarriers * sizeof(c16_t));

  txData = malloc(frame_parms->packet_samples * sizeof(c16_t));
  memset(txData, 0, frame_parms->packet_samples * sizeof(c16_t));

  rxData = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    rxData[i] = calloc(1, rx_size * sizeof(c16_t));
    memset(rxData[i], 0, rx_size * sizeof(c16_t));
  }

  envelope = malloc(rx_size * sizeof(uint16_t));
  filteredData = malloc(rx_size * sizeof(uint16_t));

  frame_parms->packet_downsampled_samples = rx_size / frame_parms->N;
  downSampled = malloc(frame_parms->packet_downsampled_samples * sizeof(uint16_t));

  // Allocate memory for cleared signal
  cleared = malloc(frame_parms->packet_downsampled_samples * sizeof(uint16_t));
  memset(cleared, 0, frame_parms->packet_downsampled_samples * sizeof(uint16_t));

  // Calculate energy of chips
  int energy_length = R_TAS_SIP_N + (MAX_AIOT_R2D_PAYLOAD_SIZE*8 - (R_TAS_SIP_N/R_TAS_SIP_M)) * frame_parms->M;
  energy = malloc(energy_length * sizeof(uint32_t));
  memset(energy, 0, energy_length * sizeof(uint32_t));

  rx_payload = malloc(MAX_AIOT_R2D_PACKET_SIZE);
  memset(rx_payload, 0, MAX_AIOT_R2D_PACKET_SIZE);

  double ber_results[snr_steps];
  memset(ber_results, 0, snr_steps * sizeof(double));

  // Initialization of transmit IQ signals
  s_re = malloc(frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  s_im = malloc(frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    s_re[i] = calloc(1, frame_parms->packet_samples * 2 * sizeof(double));
    s_im[i] = calloc(1, frame_parms->packet_samples * 2 * sizeof(double));
  }

  bzero(s_re[0], frame_parms->packet_samples * 2 * sizeof(double));
  bzero(s_im[0], frame_parms->packet_samples * 2 * sizeof(double));

  // Initialization of receive IQ signals
  r_re = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  r_im = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    r_re[i] = calloc(1, frame_parms->packet_samples * 2 * sizeof(double));
    r_im[i] = calloc(1, frame_parms->packet_samples * 2 * sizeof(double));
  }

  bzero(r_re[0], frame_parms->packet_samples * 2 * sizeof(double));
  bzero(r_im[0], frame_parms->packet_samples * 2 * sizeof(double));

  //double *SIP_ideal = generate_SIP_ideal_sequence(frame_parms);

  if(testing_timing) {
    start_meas(&time_stats);
  }

  for(int snr = snr_min; snr <= snr_max; snr += SNR_STEP_DB) {
    channel_model->SNR = snr;

    for(int trials = 0; trials < snr_trials; trials++) {
      for(int i = 0; i < payloadSize / 8; i++) {
        payload[i] = uniformrandom() * 256;
      }

      if(testing_mode) {
        printf("-------------------------------\n");
        printf("Trial %d/%d for SNR %d dB\n", trials + 1, snr_trials, snr);
      }

      AIOT_R2D_PHY_TX_REs(REsPacket, (const uint8_t *) payload, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/A_REs_Packet.m", foldername);
        LOG_M(filename, "REs_Packet_sig", REsPacket, frame_parms->packet_symbols * frame_parms->packet_subcarriers, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("TX REs generation time: %f us\n", elapsed_us);
      }

      AIOT_R2D_PHY_TX_Signal(txData, (const c16_t *) REsPacket, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/B_TX_IQ.m", foldername);
        LOG_M(filename, "TX_IQ_sig", txData, frame_parms->packet_samples, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("TX signal generation time: %f us\n", elapsed_us);
      }

      SIM_Channel_propagate(rxData, (const c16_t *) txData, channel_model, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D_RX_IQ.m", foldername);
        LOG_M(filename, "RX_IQ_sig", rxData[0], rx_size, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Channel propagation time: %f us\n", elapsed_us);
      }

      // Envelope detector (squared)
      AIOT_R2D_PHY_RX_Envelope_Detector(envelope, (const c16_t **) rxData, rx_size);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/E_Envelope.m", foldername);
        LOG_M(filename, "Envelope_sig", envelope, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Envelope detection time: %f us\n", elapsed_us);
      }

      // Low-pass filter
      AIOT_R2D_PHY_RX_Filter(filteredData, (const uint16_t *) envelope, rx_size, b, a);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/F_Filter.m", foldername);
        LOG_M(filename, "Filter_sig", filteredData, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Filtering time: %f us\n", elapsed_us);
      }

      // Downsample the signal by N = 16
      AIOT_R2D_PHY_RX_Downsample(downSampled, (const uint16_t *) filteredData, rx_size, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/G_Downsampled.m", foldername);
        LOG_M(filename, "Downsampled_sig", downSampled, frame_parms->packet_downsampled_samples, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Downsampling time: %f us\n", elapsed_us);
      }

      // Correlate the received signal with known SIP sequence
      AIOT_R2D_PHY_RX_GetPacket(rx_payload, (const uint16_t *) downSampled, frame_parms);

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Packet extraction time: %f us\n", elapsed_us);
      }

      double ber = calculate_BER(rx_payload, payload, frame_parms);
      if(frame_parms->packet_payload_size != payloadSize || (ber > 0.0)) {
        ber_results[snr - snr_min] += 1;

        if(testing_mode) {
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
          printf("BER: %f, Payload size: %d\n", ber, frame_parms->packet_payload_size);
        }
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        double elapsed_us = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("BER calculation time: %f us\n", elapsed_us);
      }
    }

    ber_results[snr - snr_min] /= snr_trials;
    printf("Completed SNR %d dB: BLER = %f\n", snr, ber_results[snr - snr_min]);
  }

  AIOT_R2D_PHY_TX_REs_free(REsPacket);
  AIOT_R2D_PHY_TX_Signal_free(txData);
  SIM_Channel_propagate_free(rxData, frame_parms->nr_frame_parms.nb_antennas_rx);
  AIOT_R2D_PHY_RX_Envelope_Detector_free(envelope);
  AIOT_R2D_PHY_RX_Filter_free(filteredData);
  AIOT_R2D_PHY_RX_Downsample_free(downSampled);

  free(rx_payload);

  for(int snr = MIN_SNR_DB; snr <= MAX_SNR_DB; snr += SNR_STEP_DB) {
    printf("SNR %d dB: BER = %f\n", snr, ber_results[snr - MIN_SNR_DB]);
  }

  if(!testing_mode) {
    sprintf(filename, "%s/BLER_M%d_%s_SIZE%d_%dRBs.m", foldername, frame_parms->M, frame_parms->Zadoff_Chu ? "ZC" : "Ones", payloadSize, frame_parms->nr_frame_parms.N_RB_DL);
    LOG_M(filename, "BLER", ber_results, MAX_SNR_DB - MIN_SNR_DB + 1, 1, 7);
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
  frame_parms->N = 16; // Downsampling factor

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
  while ((c = getopt(argc, argv, "--:O:h:L:R:P:p:M:Z:S:N:D:t:")) != -1) {
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

      case 't':
        snr_trials = 1;
        testing_mode = true;
        snr_plot = atoi(optarg);
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

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }

  free(s_re);
  free(s_im);

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

  free(r_re);
  free(r_im);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
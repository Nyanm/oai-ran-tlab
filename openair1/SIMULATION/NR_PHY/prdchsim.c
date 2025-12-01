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
#include <pthread.h>
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

#include "PHY_AIOT/defs_aiot_r2d.h"

/* Compile-time feature detection */
#if defined(__AVX512F__) && defined(__AVX512BW__)
  #define HAVE_AVX512 1
#else
  #define HAVE_AVX512 0
#endif

#if defined(__AVX2__)
  #define HAVE_AVX2 1
#else
  #define HAVE_AVX2 0
#endif

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, not finished yet */
  return "detect_leaks=0";
}

NR_AIOT_DL_FRAME_PARMS *frame_parms;

double cpuf;
char filename[50];
char foldername[] = "./R2D_results";

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

// Note: s_re, s_im, r_re, r_im are now thread-local variables
// Global channel_model is used as template, each thread gets its own copy
channel_model_t channel_model;

bool testing_mode = false;
bool testing_timing = false;

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

/*void AIOT_R2D_PHY_TX_AddCRC(uint8_t *output, const uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Copy payload
  int payload_bytes = (frame->packet_encoded_size + 7) / 8;
  memcpy(output, payload, payload_bytes);

  // Compute CRC32
  uint32_t crc = crc32(0, NULL, 0);
  crc = crc32(crc, payload, payload_bytes);

  // Append CRC32 at the end of payload
  output[payload_bytes + 0] = (crc >> 24) & 0xFF;
  output[payload_bytes + 1] = (crc >> 16) & 0xFF;
  output[payload_bytes + 2] = (crc >> 8) & 0xFF;
  output[payload_bytes + 3] = (crc >> 0) & 0xFF;
}*/

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

void AIOT_R2D_PHY_TX_Signal(c16_t *txData, c16_t* txDataF, const c16_t *REsPacket, NR_AIOT_DL_FRAME_PARMS *frame)
{
  if(txData == NULL || txDataF == NULL) {
    printf("Memory allocation failed\n");
    return;
  }

  memset(txDataF, 0, frame->packet_samples * sizeof(c16_t));

  for (int sym = 0; sym < frame->packet_symbols; sym++) {
    int in_offset = sym * frame->packet_subcarriers;
    int out_offset = sym * frame->nr_frame_parms.ofdm_symbol_size;

    memcpy(txDataF + out_offset, REsPacket + in_offset, frame->packet_subcarriers * sizeof(c16_t));
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
}

void SIM_Channel_propagate(c16_t **rxData, const c16_t *in, channel_desc_t *channel, double SNR, NR_AIOT_DL_FRAME_PARMS *frame,
                           double **s_re, double **s_im, double **r_re, double **r_im)
{
  int rx_size = frame->packet_samples + channel->channel_offset + 200;

  for (int i = 0; i < frame->packet_samples; i++) {
    s_re[0][i] = (double) in[i].r;
    s_im[0][i] = (double) in[i].i;
  }

  int txlev = signal_energy((int32_t *) in, frame->nr_frame_parms.ofdm_symbol_size + frame->nr_frame_parms.nb_prefix_samples0);
  double txlev_dBm = 10 * log10((double)txlev);
  //printf("Signal energy: %d (%f dB)\n", txlev, txlev_dBm);

  double ts = 1.0 / (frame->nr_frame_parms.subcarrier_spacing * frame->nr_frame_parms.ofdm_symbol_size);
  // Compute AWGN variance
  double sigma2_dBm = txlev_dBm + channel->path_loss_dB - SNR; // *((double)frame_parms->ofdm_symbol_size / r2d_subcarriers)
  double sigma2 = pow(10, sigma2_dBm / 10);
  //printf("Noise sigma2: %f (%f dB)\n", sigma2, sigma2_dBm);

  static uint8_t initialized = 0;
  multipath_channel(channel, s_re, s_im, r_re, r_im, frame->packet_samples + channel->channel_offset + 200, initialized, 1);
  if(initialized == 0) {
    initialized = 1;
  }

  add_noise(rxData,
            (const double **)r_re,
            (const double **)r_im,
            sigma2,
            frame->packet_samples + channel->channel_offset + 200,
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
    
    sprintf(filename, "%s/R2D_Channel.m", foldername);
    LOG_M(filename, "Channel_sig", output, rx_size, 1, 8);

    free(output);
  }
}

void AIOT_R2D_PHY_RX_Envelope_Detector(int16_t *envelope, const c16_t **rxData, int rx_size)
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
    envelope[i] = min(max_val + beta_min, 32767); // Clamp to int16_t max
  }
}

// --- 1st-Order Section (Floating-Point) ---
typedef struct {
    double b[2]; // b0, b1
    double a[2]; // a0, a1 (a0 is always 1.0)
    double s[1]; // state
} iir_ord1_f64_t;

// --- 2nd-Order (Biquad) Section (Floating-Point) ---
typedef struct {
    double b[3]; // b0, b1, b2
    double a[3]; // a0, a1, a2 (a0 is always 1.0)
    double s[2]; // states s1, s2
} iir_biquad_f64_t;

// --- 3rd-Order Filter (Cascade) ---
typedef struct {
    iir_ord1_f64_t   sec1; // 1st-order section
    iir_biquad_f64_t sec2; // 2nd-order section
} iir_butter3_f64_t;

iir_butter3_f64_t filter;

/**
 * @brief Initializes the filter state to zero.
 */
void iir_butter3_init(iir_butter3_f64_t* filt) {
    filt->sec1.s[0] = 0.0;
    filt->sec2.s[0] = 0.0;
    filt->sec2.s[1] = 0.0;
}

/**
 * @brief Generates 3rd-Order Butterworth LPF coefficients.
 * (This is identical to the fixed-point generator, just simpler storage)
 */
void generate_butter_coeffs_f64(iir_butter3_f64_t* filt, double fc, double fs) {
    // --- 1. Pre-warp frequency ---
    double w = 2.0 * fs * tan(M_PI * fc / fs);
    double T = 1.0 / fs;

    // --- 2. Calculate 1st-Order Section ---
    // H(z) = (b0 + b1*z^-1) / (a0 + a1*z^-1)
    filt->sec1.a[0] = 2.0 + w * T;
    filt->sec1.b[0] = w * T;
    filt->sec1.b[1] = w * T;
    filt->sec1.a[1] = w * T - 2.0;
    
    // --- 3. Calculate 2nd-Order Section ---
    // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
    filt->sec2.a[0] = 4.0 + 2.0*w*T + w*w*T*T;
    filt->sec2.b[0] = w*w*T*T;
    filt->sec2.b[1] = 2.0*w*w*T*T;
    filt->sec2.b[2] = w*w*T*T;
    filt->sec2.a[1] = -8.0 + 2.0*w*w*T*T;
    filt->sec2.a[2] = 4.0 - 2.0*w*T + w*w*T*T;

    // --- 4. Normalize coefficients (so a0 is 1.0) ---
    // This is crucial for the Direct Form II Transposed implementation.
    double a0_inv_1 = 1.0 / filt->sec1.a[0];
    filt->sec1.b[0] *= a0_inv_1;
    filt->sec1.b[1] *= a0_inv_1;
    filt->sec1.a[1] *= a0_inv_1;
    filt->sec1.a[0] = 1.0;

    double a0_inv_2 = 1.0 / filt->sec2.a[0];
    filt->sec2.b[0] *= a0_inv_2;
    filt->sec2.b[1] *= a0_inv_2;
    filt->sec2.b[2] *= a0_inv_2;
    filt->sec2.a[1] *= a0_inv_2;
    filt->sec2.a[2] *= a0_inv_2;
    filt->sec2.a[0] = 1.0;

    // 5. Clear state
    iir_butter3_init(filt);
}

/**
 * @brief Processes one sample through the 3rd-order filter.
 * This is the high-speed, floating-point function.
 */
double iir_butter3_filter(iir_butter3_f64_t* filt, double input) {
    // --- Process 1st-Order Section (DF-II Transposed) ---
    double y1 = (filt->sec1.b[0] * input) + filt->sec1.s[0];
    filt->sec1.s[0] = (filt->sec1.b[1] * input) - (filt->sec1.a[1] * y1);
    
    // --- Process 2nd-Order Section (DF-II Transposed) ---
    double y2 = (filt->sec2.b[0] * y1) + filt->sec2.s[0];
    filt->sec2.s[0] = (filt->sec2.b[1] * y1) - (filt->sec2.a[1] * y2) + filt->sec2.s[1];
    filt->sec2.s[1] = (filt->sec2.b[2] * y1) - (filt->sec2.a[2] * y2);
    
    return y2; // Final output
}

void AIOT_R2D_PHY_RX_Filter(int16_t *out, const int16_t *in, int length, iir_butter3_f64_t *filt)
{
  iir_butter3_init(filt);

  for (int n = 0; n < length; n++) {
    out[n] = iir_butter3_filter(filt, in[n]);
  }
}

void AIOT_R2D_PHY_RX_Downsample(int16_t *out, const int16_t *in, int length, NR_AIOT_DL_FRAME_PARMS *frame)
{
  for (int i = 0; i < frame->packet_downsampled_samples; i++) {
    out[i] = in[i * frame->N];
  }
}

int *SIP_ideal = NULL;

int *generate_SIP_ideal_sequence(NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  // Compute lengths for downsampled SIP ideal sequence;
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_chip_size = downsampled_OFDM_size / R_TAS_SIP_M;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  frame_parms->SIP_samples = 2*downsampled_chip_size + downsampled_CP0_size + downsampled_OFDM_size + downsampled_CP_size + downsampled_OFDM_size;

  // Generate ideal SIP sequence for correlation
  SIP_ideal = malloc(frame_parms->SIP_samples * sizeof(int));
  int value0dis = -2;
  int value0 = -1;
  int value1 = 1;

  int samples = 0;

  // Add two zero chips
  for (samples = 0; samples < 2*downsampled_chip_size; samples++) {
    SIP_ideal[samples] = value0;
  }

  // Add Cyclic prefix (Longer Type 0)
  for (int i = 0; i < downsampled_CP0_size; i++) {
    SIP_ideal[samples + i] = value0;
  }
  samples += downsampled_CP0_size;

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
      value = value0dis;
    }

    for (int j = 0; j < downsampled_OFDM_size / R_TAS_SIP_M; j++) {
      SIP_ideal[samples + j] = value;
    }

    samples += downsampled_chip_size;
  }

  return SIP_ideal;
}

int AIOT_R2D_PHY_RX_Synchronize(int *correlation, const int16_t *signal, int *SIP_ideal, NR_AIOT_DL_FRAME_PARMS *frame)
{
  // Correlate received signal with ideal Preamble (vectorized with AVX2 when available)
  int Preamble_offset = 0;
  int max_corr = INT_MIN;

  int N = frame->packet_downsampled_samples - frame->SIP_samples;
  int L = frame->SIP_samples;

  for (int i = 0; i < N; i++) {
    int acc = 0;

#if HAVE_AVX512
    /* AVX-512 implementation (compile-time selected) */
    int j = 0;
    const int16_t *sig_ptr = signal + i;
    const int *sip_ptr = SIP_ideal;

    int j16 = (L / 16) * 16;
    for (; j < j16; j += 16) {
      __m256i s16 = _mm256_loadu_si256((const __m256i *)(sig_ptr + j)); // 16 x int16
      __m512i s32 = _mm512_cvtepi16_epi32(s16);                         // 16 x int32

      __m512i p32 = _mm512_loadu_si512((const void *)(sip_ptr + j));    // 16 x int32

      __m512i prod = _mm512_mullo_epi32(s32, p32);

      int32_t tmp[16];
      _mm512_storeu_si512((__m512i *)tmp, prod);
      acc += tmp[0]  + tmp[1]  + tmp[2]  + tmp[3]
           + tmp[4]  + tmp[5]  + tmp[6]  + tmp[7]
           + tmp[8]  + tmp[9]  + tmp[10] + tmp[11]
           + tmp[12] + tmp[13] + tmp[14] + tmp[15];
    }

    /* Use AVX2 for remaining multiple-of-8 chunk if available at compile time */
    int j8 = (L / 8) * 8;
    for (; j < j8; j += 8) {
      __m128i s16_lo = _mm_loadu_si128((const __m128i *)(sig_ptr + j)); // 8 x int16
      __m256i s32_8   = _mm256_cvtepi16_epi32(s16_lo);                  // 8 x int32
      __m256i p32_8   = _mm256_loadu_si256((const __m256i *)(sip_ptr + j));
      __m256i prod8   = _mm256_mullo_epi32(s32_8, p32_8);
      int32_t tmp8[8];
      _mm256_storeu_si256((__m256i *)tmp8, prod8);
      acc += tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];
    }

    for (; j < L; j++) {
      acc += (int)sig_ptr[j] * sip_ptr[j];
    }

#elif HAVE_AVX2
    /* AVX2-only implementation (compile-time selected) */
    int j = 0;
    const int16_t *sig_ptr = signal + i;
    const int *sip_ptr = SIP_ideal;

    int j8 = (L / 8) * 8;
    for (; j < j8; j += 8) {
      __m128i s16 = _mm_loadu_si128((const __m128i *)(sig_ptr + j)); // 8 x int16
      __m256i s32 = _mm256_cvtepi16_epi32(s16);                      // 8 x int32

      __m256i p32 = _mm256_loadu_si256((const __m256i *)(sip_ptr + j)); // 8 x int32

      __m256i prod = _mm256_mullo_epi32(s32, p32);

      int32_t tmp[8];
      _mm256_storeu_si256((__m256i *)tmp, prod);
      acc += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }

    for (; j < L; j++) {
      acc += (int)sig_ptr[j] * sip_ptr[j];
    }

#else
    /* Scalar fallback (no SIMD) */
    for (int j = 0; j < L; j++) {
      acc += (int)signal[i + j] * SIP_ideal[j];
    }
#endif

    correlation[i] = acc;

    if (acc > max_corr) {
      max_corr = acc;
      Preamble_offset = i;
    }
  }

  if(testing_mode) {
    printf("Detected SIP at offset %d, Corr: %d\n", Preamble_offset, max_corr);
  }
  return Preamble_offset;
}

void AIOT_R2D_PHY_RX_GetPacket(uint8_t *rx_payload, const int16_t *signal, int SIP_offset, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  int downsampled_OFDM_size = frame_parms->nr_frame_parms.ofdm_symbol_size / frame_parms->N;
  int downsampled_CP_size = frame_parms->nr_frame_parms.nb_prefix_samples / frame_parms->N;
  int downsampled_CP0_size = frame_parms->nr_frame_parms.nb_prefix_samples0 / frame_parms->N;
  int M4_chip_size = downsampled_OFDM_size / R_TAS_SIP_M;
  frame_parms->packet_payload_size = frame_parms->packet_downsampled_samples - frame_parms->SIP_samples; // Max payload size in bits

  static int pass = MIN_SNR_DB;

  int CAP_energy[R_TAS_CAP_N] = {0};
  int SIP_bit0_energy = 0;

  // Get energy of last bit0 of SIP (its zero)
  for (int j = 0; j < M4_chip_size; j++)
  {
    SIP_bit0_energy += signal[SIP_offset + (frame_parms->SIP_samples - M4_chip_size) + j];
  }

  int position = SIP_offset + frame_parms->SIP_samples + downsampled_CP_size; // Start of R-TAS-CAP (after CP)

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

  uint32_t *energy_plot = NULL, *thr_plot = NULL;
  int energy_index = 0, thr_index = 0;

  if(testing_mode && pass == snr_plot) {
    // Energy plotting
    energy_plot = malloc(frame_parms->packet_payload_size * sizeof(uint32_t));
    memset(energy_plot, 0, frame_parms->packet_payload_size * sizeof(uint32_t));

    // Threshold plotting
    thr_plot = malloc(frame_parms->packet_payload_size * sizeof(uint32_t));
    memset(thr_plot, 0, frame_parms->packet_payload_size * sizeof(uint32_t));

    // Threshold for next bit
    thr_plot[thr_index++] = threshold;
  }

  int symbolCounter = (R_TAS_SIP_N / R_TAS_SIP_M) + (R_TAS_CAP_N / frame_parms->received_M);
  int endCounter = 0;
  int bits = 0;
  int chip = 0;
  uint32_t energy[2] = {0};
  frame_parms->packet_payload_size = 0;

  while(position + received_chip_size <= frame_parms->packet_downsampled_samples)
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

    if(testing_mode && pass == snr_plot) {
      // Save energy for plotting
      energy_plot[energy_index++] = energy[chip%2];
    }

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

    if(testing_mode && pass == snr_plot) {
      // Save threshold for plotting
      thr_plot[thr_index++] = threshold;
    }

    energy[0] = 0;
    energy[1] = 0;
  }

  if(testing_mode && pass == snr_plot) {
    sprintf(filename, "%s/R2D_Adaptive_threshold.m", foldername);
    LOG_M(filename, "Adaptive_threshold_sig", thr_plot, thr_index, 1, 2);

    sprintf(filename, "%s/R2D_Energy.m", foldername);
    LOG_M(filename, "Energy_sig", energy_plot, energy_index, 1, 2);
  }

  pass++;
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

double calculate_BER(uint8_t *rx_payload, uint8_t *payload, NR_AIOT_DL_FRAME_PARMS *frame_parms)
{
  int BER_errors = 0;
  for (int i = 0; i < frame_parms->packet_payload_size/8; i++) {
    int diff = payload[i] ^ rx_payload[i];

    // Count ones in diff
    for (int b = 0; b < 8; b++) {
      BER_errors += (diff >> b) & 0x01;
    }
  }

  return (double)BER_errors / (double)frame_parms->packet_payload_size;
}

time_stats_t time_stats = {0};

// Thread data structure for parallel SNR processing
typedef struct {
  int snr_start;
  int snr_end;
  int thread_id;
  uint8_t *payload;
  int payloadSize;
  NR_AIOT_DL_FRAME_PARMS *frame_parms;
  channel_model_t *channel_model;
  double *ber_results;
  pthread_mutex_t *print_mutex;
  // Per-thread timing results
  double time_tx_REs;
  double time_tx_signal;
  double time_channel;
  double time_envelope;
  double time_filter;
  double time_downsample;
  double time_sync;
  double time_rx_packet;
  double time_ber;
} snr_thread_data_t;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

// Thread function for processing SNR range
void* process_snr_range(void* arg) {
  snr_thread_data_t *data = (snr_thread_data_t*)arg;
  
  // Per-thread variables to avoid conflicts
  channel_desc_t *channel_params = new_channel_desc_scm(data->frame_parms->nr_frame_parms.nb_antennas_tx,
                                        data->frame_parms->nr_frame_parms.nb_antennas_rx,
                                        data->channel_model->channel_model,
                                        data->channel_model->sampling_rate,
                                        data->channel_model->fc,
                                        data->channel_model->bw,
                                        data->channel_model->DS_TDL,
                                        0.0,
                                        CORR_LEVEL_LOW,
                                        0,
                                        data->channel_model->delay,
                                        data->channel_model->path_loss_dB,
                                        data->channel_model->noise_power_dB);

  c16_t *REsPacket = malloc(data->frame_parms->packet_symbols * data->frame_parms->packet_subcarriers * sizeof(c16_t));
  c16_t *txData = malloc(data->frame_parms->packet_samples * sizeof(c16_t));
  c16_t *txDataF = malloc(data->frame_parms->packet_samples * sizeof(c16_t));
  
  int rx_size = data->frame_parms->packet_samples + data->channel_model->delay + 200;
  c16_t **rxData = malloc(data->frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    rxData[i] = calloc(1, rx_size * sizeof(c16_t));
  }
  
  int16_t *envelope = malloc(rx_size * sizeof(uint16_t));
  int16_t *filteredData = malloc(rx_size * sizeof(uint16_t));
  int16_t *downSampled = malloc((rx_size / data->frame_parms->N) * sizeof(uint16_t));
  uint8_t *rx_payload = malloc(MAX_AIOT_R2D_PACKET_SIZE);
  int *correlation = malloc((rx_size / data->frame_parms->N - data->frame_parms->SIP_samples) * sizeof(int));
  
  // Thread-local payload buffer
  uint8_t *local_payload = malloc(MAX_AIOT_R2D_PAYLOAD_SIZE);
  memcpy(local_payload, data->payload, (data->payloadSize + 7) / 8);
  
  // Thread-local filter
  iir_butter3_f64_t local_filter;
  generate_butter_coeffs_f64(&local_filter, data->channel_model->bw * 1e6, (double)data->channel_model->sampling_rate * 1e6);
  
  // Thread-local IQ signal buffers
  double **s_re = malloc(data->frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  double **s_im = malloc(data->frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    s_re[i] = calloc(1, rx_size * sizeof(double));
    s_im[i] = calloc(1, rx_size * sizeof(double));
  }
  
  double **r_re = malloc(data->frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  double **r_im = malloc(data->frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    r_re[i] = calloc(1, rx_size * sizeof(double));
    r_im[i] = calloc(1, rx_size * sizeof(double));
  }
  
  // Thread-local timing stats
  time_stats_t local_time_stats = {0};
  
  // Process SNR range assigned to this thread
  for(int snr = data->snr_start; snr <= data->snr_end; snr += SNR_STEP_DB) {
    channel_model_t local_channel_model = *data->channel_model;
    local_channel_model.SNR = snr;
    
    for(int trials = 0; trials < snr_trials; trials++) {
      // Generate new payload for each trial
      for(int i = 0; i < data->payloadSize / 8; i++) {
        local_payload[i] = uniformrandom() * 256;
      }
      
      if(testing_mode) {
        pthread_mutex_lock(data->print_mutex);
        printf("*************************\n");
        printf("Thread %d: Testing SNR %d dB\n", data->thread_id, snr);
        pthread_mutex_unlock(data->print_mutex);
      }
      
      if(testing_timing && snr != snr_plot) {
        start_meas(&local_time_stats);
      }
      
      AIOT_R2D_PHY_TX_REs(REsPacket, (const uint8_t *) local_payload, data->frame_parms);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_tx_REs += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_R2D_PHY_TX_Signal(txData, txDataF, (const c16_t *) REsPacket, data->frame_parms);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_tx_signal += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      SIM_Channel_propagate(rxData, (const c16_t *) txData, channel_params, local_channel_model.SNR, data->frame_parms,
                            s_re, s_im, r_re, r_im);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_channel += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_R2D_PHY_RX_Envelope_Detector(envelope, (const c16_t **) rxData, rx_size);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_envelope += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_R2D_PHY_RX_Filter(filteredData, (const int16_t *) envelope, rx_size, &local_filter);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_filter += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_R2D_PHY_RX_Downsample(downSampled, (const int16_t *) filteredData, rx_size, data->frame_parms);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_downsample += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      int SIP_offset = AIOT_R2D_PHY_RX_Synchronize(correlation, (const int16_t *) downSampled, SIP_ideal, data->frame_parms);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_sync += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      memset(rx_payload, 0, MAX_AIOT_R2D_PACKET_SIZE);
      AIOT_R2D_PHY_RX_GetPacket(rx_payload, (const int16_t *) downSampled, SIP_offset, data->frame_parms);
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_rx_packet += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      double ber = calculate_BER(rx_payload, local_payload, data->frame_parms);
      if(data->frame_parms->packet_payload_size != data->payloadSize || (ber > 0.0)) {
        data->ber_results[snr - snr_min] += 1;
      }
      
      if(testing_timing && snr != snr_plot) {
        stop_meas(&local_time_stats);
        data->time_ber += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
      }
    }
    
    data->ber_results[snr - snr_min] /= snr_trials;
    
    pthread_mutex_lock(data->print_mutex);
    printf("Thread %d completed SNR %d dB: BLER = %f\n", data->thread_id, snr, data->ber_results[snr - snr_min]);
    pthread_mutex_unlock(data->print_mutex);
  }
  
  // Cleanup thread-local resources
  free(REsPacket);
  free(txData);
  free(txDataF);
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    free(rxData[i]);
  }
  free(rxData);
  free(envelope);
  free(filteredData);
  free(downSampled);
  free(rx_payload);
  free(correlation);
  free(local_payload);
  
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }
  free(s_re);
  free(s_im);
  
  for (int i = 0; i < data->frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }
  free(r_re);
  free(r_im);
  
  free_channel_desc_scm(channel_params);
  
  return NULL;
}

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

  // Generate SIP ideal sequence (shared by all threads)
  frame_parms->packet_downsampled_samples = (frame_parms->packet_samples + channel_model->delay + 200) / frame_parms->N;
  SIP_ideal = generate_SIP_ideal_sequence(frame_parms);

  if(testing_mode) {
    sprintf(filename, "%s/R2D_SIP_Ideal.m", foldername);
    LOG_M(filename, "SIP_Ideal_sig", SIP_ideal, frame_parms->SIP_samples, 1, 2);
  }

  double ber_results[snr_steps];
  memset(ber_results, 0, snr_steps * sizeof(double));

  if(!testing_mode) {
    // Determine number of threads (use number of CPU cores or 4, whichever is smaller)
    int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads < 1) num_threads = 1;
    int total_snr_points = (snr_max - snr_min) / SNR_STEP_DB + 1;
    
    if (num_threads > total_snr_points) {
      num_threads = total_snr_points;
    }

    printf("Using %d threads for parallel processing\n", num_threads);
    
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    snr_thread_data_t *thread_data = malloc(num_threads * sizeof(snr_thread_data_t));
    
    // Calculate SNR range for each thread
    int snr_per_thread = total_snr_points / num_threads;
    int remaining_snr = total_snr_points % num_threads;
    
    for (int t = 0; t < num_threads; t++) {
      thread_data[t].thread_id = t;
      thread_data[t].payload = payload;
      thread_data[t].payloadSize = payloadSize;
      thread_data[t].frame_parms = frame_parms;
      thread_data[t].channel_model = channel_model;
      thread_data[t].ber_results = ber_results;
      thread_data[t].print_mutex = &print_mutex;
      
      // Calculate SNR range for this thread
      int start_idx = t * snr_per_thread + (t < remaining_snr ? t : remaining_snr);
      int end_idx = start_idx + snr_per_thread + (t < remaining_snr ? 1 : 0) - 1;
      
      thread_data[t].snr_start = snr_min + start_idx * SNR_STEP_DB;
      thread_data[t].snr_end = snr_min + end_idx * SNR_STEP_DB;
      
      printf("Thread %d will process SNR range %d to %d dB\n", t, thread_data[t].snr_start, thread_data[t].snr_end);
    }
    
    // Create and start threads
    for (int t = 0; t < num_threads; t++) {
      if (pthread_create(&threads[t], NULL, process_snr_range, &thread_data[t]) != 0) {
        printf("Error creating thread %d\n", t);
        exit(1);
      }
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
      pthread_join(threads[t], NULL);
    }
    
    // Cleanup
    free(threads);
    free(thread_data);
  
  } else { // testing mode
    snr_thread_data_t thread_data;
    
    thread_data.thread_id = 0;
    thread_data.payload = payload;
    thread_data.payloadSize = payloadSize;
    thread_data.frame_parms = frame_parms;
    thread_data.channel_model = channel_model;
    thread_data.ber_results = ber_results;
    thread_data.print_mutex = &print_mutex;
    
    if(testing_timing) {
      // Initialize timing results
      thread_data.time_tx_REs = 0.0;
      thread_data.time_tx_signal = 0.0;
      thread_data.time_channel = 0.0;
      thread_data.time_envelope = 0.0;
      thread_data.time_filter = 0.0;
      thread_data.time_downsample = 0.0;
      thread_data.time_sync = 0.0;
      thread_data.time_rx_packet = 0.0;
      thread_data.time_ber = 0.0;
    }
      
    thread_data.snr_start = snr_min;
    thread_data.snr_end = snr_max;
    
    // Create and start threads
    process_snr_range(&thread_data);

    if(testing_timing) {
      thread_data.time_tx_REs /= snr_steps - 1;
      thread_data.time_tx_signal /= snr_steps - 1;
      thread_data.time_channel /= snr_steps - 1;
      thread_data.time_envelope /= snr_steps - 1;
      thread_data.time_filter /= snr_steps - 1;
      thread_data.time_downsample /= snr_steps - 1;
      thread_data.time_sync /= snr_steps - 1;
      thread_data.time_rx_packet /= snr_steps - 1;
      thread_data.time_ber /= snr_steps - 1;
    }

    if(testing_timing) {
      printf("-------------------------------\n");
      printf("Timing results (average per packet in us):\n");
      printf("-------------------------------\n");
      printf("TX REs generation:       %f us\n", thread_data.time_tx_REs);
      printf("TX signal generation:    %f us\n", thread_data.time_tx_signal);
      printf("Channel propagation:     %f us\n", thread_data.time_channel);
      printf("Envelope detection:      %f us\n", thread_data.time_envelope);
      printf("Filtering:               %f us\n", thread_data.time_filter);
      printf("Downsampling:            %f us\n", thread_data.time_downsample);
      printf("Synchronization:         %f us\n", thread_data.time_sync);
      printf("RX packet extraction:    %f us\n", thread_data.time_rx_packet);
      printf("BER calculation:         %f us\n", thread_data.time_ber);
      double total_time = thread_data.time_tx_REs + thread_data.time_tx_signal + thread_data.time_channel + thread_data.time_envelope +
                          thread_data.time_filter + thread_data.time_downsample + thread_data.time_sync + thread_data.time_rx_packet + thread_data.time_ber;
      printf("---\n");
      printf("Total time:              %f us\n", total_time);
    }
  }

  free(SIP_ideal);

  printf("-------------------------------\n");
  printf("            Results\n");
  printf("-------------------------------\n");
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

  int loglvl = OAILOG_ERR;

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
  while ((c = getopt(argc, argv, "--:O:h:L:R:P:p:M:Z:S:N:D:t:T:r:")) != -1) {
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
        printf("-r SNR trials per SNR point\n");
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
      
      case 'T':
        testing_timing = true;
        break;

      case 'r':
        snr_trials = atoi(optarg);
        printf("Using %d trials per SNR point\n", snr_trials);
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
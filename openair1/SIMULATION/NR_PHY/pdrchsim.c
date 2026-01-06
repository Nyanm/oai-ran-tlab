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

#include "PHY_AIOT/defs_aiot_d2r.h"

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, not finished yet */
  return "detect_leaks=0";
}

NR_AIOT_UL_FRAME_PARMS *frame_parms;

double cpuf;
int num_threads;
char filename[100];
char foldername[] = "./D2R_results";
char folderplots[50];

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

#define MIN_SNR_DB (-30)
#define MAX_SNR_DB 20

#define MIN_SNR_DB_DEFAULT (-15)
#define MAX_SNR_DB_DEFAULT 10

#define SNR_STEP_DB 1
#define SNR_TRIALS 1000
#define SNR_STEPS ((MAX_SNR_DB_DEFAULT - MIN_SNR_DB_DEFAULT) / SNR_STEP_DB + 1)

int snr_min = MIN_SNR_DB_DEFAULT;
int snr_max = MAX_SNR_DB_DEFAULT;
int snr_steps = SNR_STEPS;
int snr_iters = SNR_TRIALS;
int snr_plot = 0;

int rx_size;

channel_model_t channel_model;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

bool testing_mode = false;
bool testing_timing = false;

void AIOT_D2R_PHY_TX_AddCRC(uint8_t *output, uint8_t *payload, NR_AIOT_UL_FRAME_PARMS *frame)
{
  int payloadBits = frame->payload_size;
  int crc_bits = (frame->payload_size > 24) ? 16 : 6;
  uint32_t crc = 0;

  if(payloadBits > 24) {
      crc = crc16((unsigned char *) payload, payloadBits) >> 16;
  } else {
      crc = crc6((unsigned char *) payload, payloadBits) >> 26;
  }

  if(testing_mode && !testing_timing) {
    if(crc_bits == 16) {
      printf("[TX CRC] CRC: 0x%04X, 16-bit\n", crc);
    } else {
      printf("[TX CRC] CRC: 0x%02X, 6-bit\n", crc);
    }
  }

  // Place CRC msb-first immediately after last payload bit
  for (int i = 0; i < crc_bits; i++) {
    int bitpos = payloadBits + i;
    int byte_idx = bitpos / 8;
    int bit_idx  = 7 - (bitpos & 0x7); // msb-first within byte
    uint8_t bit = (crc >> (crc_bits - 1 - i)) & 0x1;

    if (bit)
      output[byte_idx] |= (1 << bit_idx);
    else
      output[byte_idx] &= ~(1 << bit_idx);
  }
}

/* ***************          *************************  */

void AIOT_D2R_PHY_TX_calc_packet_sizes(NR_AIOT_UL_FRAME_PARMS *frame)
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

  // Compute samples for preamble and midamble
  frame->preamble_samples = (frame->N_preamble + 1) * frame->N_bit; // +1 for preceeding zero bit
  frame->midamble_samples = frame->N_preamble * frame->N_bit;

  printf("Calculated preamble samples: %d, midamble samples: %d\n", frame->preamble_samples, frame->midamble_samples);

  // Calculate packet size
  frame->packet_size = frame->payload_size + ((frame->payload_size > 24) ? 16 : 6);
  frame->packet_samples = frame->packet_size;
  if(frame->I_add) {
    frame->packet_samples += (1 + ceil(frame->packet_size / (double)frame->N_midamble_space)) * frame->N_preamble;
  } else {
    frame->packet_samples += (1 + (frame->packet_size / frame->N_midamble_space)) * frame->N_preamble;
  }
  frame->packet_samples *= frame->N_bit;

  // 2x block repetition if R_block is true
  frame->packet_samples *= frame->R_block ? 2 : 1;

  printf("Calculated chip samples: %d, bit samples: %d, midamble spacing: %d\n", frame->N_chip, frame->N_bit, frame->N_midamble_space);
  printf("Calculated D2R packet size: %d samples\n", frame->packet_samples);

  // Calculate small frequency shift
  double f_sfs = 1/(2*T_chip); // in Hz
  frame->f_min = max(f_sfs - 4e3, 0);
  frame->f_max = f_sfs + 4e3;
  printf("Calculated SFS min: %.2f Hz, max: %.2f Hz\n", frame->f_min, frame->f_max);
}

void AIOT_D2R_PHY_TX_Signal(c16_t *txData, const uint8_t *payload, NR_AIOT_UL_FRAME_PARMS *frame)
{
  if(txData == NULL) {
    printf("Memory allocation failed\n");
    return;
  }

  const int value1 = 32767;
  const int value0 = 0;

  int dataIndex = 0;

  // Add D-TAS preamble (short or long)
  uint32_t D_TAS_preamble = (frame->L_preamble) ? D_TAS_31_BITS : D_TAS_7_BITS;

  for(int i = 0; i < frame->N_preamble; i++) {
    unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - i)) & 0x01;

    // Prepare bits for Manchester encoding
    const c16_t valueBit0 = {((bit) ? value0 : value1), 0};
    const c16_t valueBit1 = {((bit) ? value1 : value0), 0};

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
  for(int i = 0; i < frame->packet_size; i++) {
    if(i % frame->N_midamble_space == 0 && i != 0) {
      // Insert midamble (fixed pattern 101010...)
      for(int m = 0; m < frame->N_preamble; m++) {
        unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - m)) & 0x01;

        // Prepare bits for Manchester encoding
        const c16_t valueBit0 = {((bit) ? value0 : value1), 0};
        const c16_t valueBit1 = {((bit) ? value1 : value0), 0};

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
    const c16_t valueBit0 = {((bit) ? value0 : value1), 0};
    const c16_t valueBit1 = {((bit) ? value1 : value0), 0};

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
      const c16_t valueBit0 = {((bit) ? value0 : value1), 0};
      const c16_t valueBit1 = {((bit) ? value1 : value0), 0};

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

void SIM_Channel_propagate(c16_t **rxData, const c16_t *in, channel_desc_t *channel, double SNR, NR_AIOT_UL_FRAME_PARMS *frame,
                           double **s_re, double **s_im, double **r_re, double **r_im, double *time_multipath, double *time_noise, gaussZiggurat_MT_t *gz)
{
  time_stats_t time_multipath_stats = {0};
  time_stats_t time_noise_stats = {0};

  start_meas(&time_multipath_stats);

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

  multipath_channel(channel, s_re, s_im, r_re, r_im, frame->packet_samples + channel->channel_offset + 2048, 0, 1);

  stop_meas(&time_multipath_stats);
  *time_multipath += time_multipath_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;

  start_meas(&time_noise_stats);

  add_noise_MT(rxData,
            (const double **)r_re,
            (const double **)r_im,
            sigma2,
            frame->packet_samples + channel->channel_offset + 2048,
            0,
            ts,
            0,
            0x0,
            0x1,
            frame->nr_frame_parms.nb_antennas_rx,
            gz);

  stop_meas(&time_noise_stats);
  *time_noise += time_noise_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
}

void AIOT_D2R_PHY_RX_Envelope_Detector(int16_t *envelope, const c16_t **rxData, int rx_size)
{
  // Cast to linear int16 pointer for easier SIMD indexing
  const int16_t *src = (const int16_t*)rxData[0];
  int i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i max_val = _mm512_set1_epi32(32767);

  for (; i + 15 < rx_size; i += 16) {
    __m512i vec = _mm512_loadu_si512(&src[2 * i]);
    
    // 1. Calc Re^2 + Im^2 (Result: 16 x int32)
    __m512i sq_32 = _mm512_madd_epi16(vec, vec);

    // 2. Convert to Float -> Sqrt -> Back to Int32
    __m512i root_32 = _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(sq_32)));

    // 3. Saturate (min), Down-convert (int32->int16), and Store
    // Note: _mm512_cvtepi32_epi16 converts 512-bit int32 to 256-bit int16
    __m256i result = _mm512_cvtepi32_epi16(_mm512_min_epi32(root_32, max_val));
    _mm256_storeu_si256((__m256i*)&envelope[i], result);
  }

#elif defined(__AVX2__)
  for (; i + 7 < rx_size; i += 8) {
    __m256i vec = _mm256_loadu_si256((__m256i*)&src[2 * i]);
    
    // 1. Calc Re^2 + Im^2 (Result: 8 x int32)
    __m256i sq_32 = _mm256_madd_epi16(vec, vec);

    // 2. Convert to Float -> Sqrt -> Back to Int32
    __m256i root_32 = _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(sq_32)));

    // 3. Pack 32-bit integers to 16-bit (Auto-saturates)
    // We must split the 256-bit register to pack it into a 128-bit register
    __m128i packed = _mm_packs_epi32(
        _mm256_castsi256_si128(root_32),      // Low 128 bits
        _mm256_extracti128_si256(root_32, 1)  // High 128 bits
    );
    _mm_storeu_si128((__m128i*)&envelope[i], packed);
  }
#endif

  // Scalar Fallback
  for (; i < rx_size; i++) {
    int32_t re = src[2 * i];
    int32_t im = src[2 * i + 1];
    
    // Sqrt(Re^2 + Im^2)
    int32_t mag = (int32_t)sqrtf((float)(re * re + im * im));
    
    // Saturate and Store
    envelope[i] = (mag > 32767) ? 32767 : (int16_t)mag;
  }
}

/* 3rd Order Butterworth Filter Structure
   Implemented as Cascaded Sections:
   [Section 1: 1st Order] -> [Section 2: 2nd Order (Biquad)]
*/
typedef struct {
  // --- Coefficients (Q2.29) ---
  int32_t b0_1, b1_1, a1_1;             // Stage 1
  int32_t b0_2, b1_2, b2_2, a1_2, a2_2; // Stage 2
} Butter3_Q15;

// --- Fixed Point Configuration ---
// Data: Q1.15 [-1, 1)
// Coeffs: Q2.29 [-4, 4) to handle high sample rate precision
#define SHIFT_COEFF 29

// Helper: Convert float to Q2.29
static int32_t dbl_to_fixed(double x) {
    return (int32_t)(x * (1 << SHIFT_COEFF));
}

// Helper: Saturation (Clamp to valid int16 range)
// Prevents wrapping if the filter overshoots 32767
static int16_t saturate_q15(int64_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

void AIOT_D2R_PHY_RX_Design_Filter(Butter3_Q15 *f, double Fc, double Fs) {
  // Bilinear Transform Math (Pre-warping)
  double omega = 2.0 * M_PI * Fc;
  double T = 1.0 / Fs;
  double wa = (2.0 / T) * tan(omega * T / 2.0);

  // Calculate Stage 1 (1st Order)
  // H(s) = 1 / (s + 1) normalized
  double gamma1 = wa * T / 2.0;
  double D1 = 1.0 + gamma1;
  
  f->b0_1 = dbl_to_fixed(gamma1 / D1);
  f->b1_1 = dbl_to_fixed(gamma1 / D1);
  f->a1_1 = dbl_to_fixed((gamma1 - 1.0) / D1);

  // Calculate Stage 2 (2nd Order)
  // H(s) = 1 / (s^2 + s + 1) normalized
  double gamma2 = wa * T / 2.0;
  double D2 = 1.0 + gamma2 + gamma2 * gamma2;

  f->b0_2 = dbl_to_fixed((gamma2 * gamma2) / D2);
  f->b1_2 = dbl_to_fixed(2.0 * (gamma2 * gamma2) / D2);
  f->b2_2 = dbl_to_fixed((gamma2 * gamma2) / D2);
  f->a1_2 = dbl_to_fixed((2.0 * gamma2 * gamma2 - 2.0) / D2);
  f->a2_2 = dbl_to_fixed((1.0 - gamma2 + gamma2 * gamma2) / D2);
}

void AIOT_D2R_PHY_RX_Filter(int16_t *output, const int16_t *input, int length, Butter3_Q15 *f)
{
  int16_t x1 = 0, y1 = 0;     // Stage 1 history
  int16_t x2_1 = 0, x2_2 = 0; // Stage 2 input history
  int16_t y2_1 = 0, y2_2 = 0; // Stage 2 output history

  for (size_t i = 0; i < length; i++) {
    int16_t in_sample = input[i];

    // --- STAGE 1: 1st Order ---
    // Math: (Q1.15 * Q2.29) = Q3.44. Accumulator needs 64-bit.
    int64_t acc1 = 0;
    acc1 += (int64_t)in_sample * f->b0_1;
    acc1 += (int64_t)x1        * f->b1_1;
    acc1 -= (int64_t)y1        * f->a1_1;

    // Rounding (add half LSB) and Shift back to Q1.15
    // Result is technically Q3.15 now, so we must saturate before casting to int16
    int64_t stage1_raw = (acc1 + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF;
    int16_t stage1_out = saturate_q15(stage1_raw);

    // Update Stage 1 State
    x1 = in_sample;
    y1 = stage1_out;

    // --- STAGE 2: 2nd Order ---
    int64_t acc2 = 0;
    acc2 += (int64_t)stage1_out * f->b0_2;
    acc2 += (int64_t)x2_1       * f->b1_2;
    acc2 += (int64_t)x2_2       * f->b2_2;
    acc2 -= (int64_t)y2_1       * f->a1_2;
    acc2 -= (int64_t)y2_2       * f->a2_2;

    // Rounding and shifting
    int64_t stage2_raw = (acc2 + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF;
    int16_t final_out = saturate_q15(stage2_raw);

    // Update Stage 2 State
    x2_2 = x2_1;
    x2_1 = stage1_out;
    y2_2 = y2_1;
    y2_1 = final_out;

    output[i] = final_out;
  }
}

int *Preamble_ideal = NULL;

int *generate_preamble_ideal_sequence(NR_AIOT_UL_FRAME_PARMS *frame)
{
  // Generate ideal SIP sequence for correlation
  Preamble_ideal = malloc(frame->preamble_samples * sizeof(int));
  int value0 = -1;
  int value1 = 1;

  // Put zero bit at the beginning
  for(int j = 0; j < frame->N_bit; j++) {
      Preamble_ideal[j] = value0;
  }

  // Add D-TAS preamble (short or long)
  uint32_t D_TAS_preamble = (frame->L_preamble) ? D_TAS_31_BITS : D_TAS_7_BITS;

  int samples = frame->N_bit;
  for(int i = 0; i < frame->N_preamble; i++) {
    unsigned bit = (D_TAS_preamble >> (frame->N_preamble - 1 - i)) & 0x01;

    // Prepare bits for Manchester encoding
    const int valueBit0 = ((bit) ? value0 : value1);
    const int valueBit1 = ((bit) ? value1 : value0);

    // Repeat each bit N_SFS times with OOK modulation
    for(int j = 0; j < frame->N_SFS; j++) {
      for(int k = 0; k < frame->N_chip; k++) {
        Preamble_ideal[samples++] = valueBit0;
      }
      for(int k = 0; k < frame->N_chip; k++) {
        Preamble_ideal[samples++] = valueBit1;
      }
    }
  }

  return Preamble_ideal;
}

int AIOT_D2R_PHY_RX_Synchronize(int *correlation, const int16_t *signal, int *Preamble_ideal, NR_AIOT_UL_FRAME_PARMS *frame)
{
  // Correlate received signal with ideal Preamble (vectorized with AVX2 when available)
  int Preamble_offset = 0;
  int max_corr = INT_MIN;

  int N = frame->packet_samples - frame->preamble_samples;
  int L = frame->preamble_samples;

  for (int i = 0; i < N; i++) {
    int acc = 0;
    int j = 0;

    const int16_t *sig_ptr = signal + i;
    const int *sip_ptr = Preamble_ideal;

#if defined(__AVX512F__) && defined(__AVX512BW__)
    /* AVX-512 intrinsics */
    int j16 = (L / 16) * 16;
    for (; j < j16; j += 16) {
      __m256i s16 = _mm256_loadu_si256((const __m256i *)(sig_ptr + j));    // 16 x int16
      __m512i s32 = _mm512_cvtepi16_epi32(s16);                            // widen to 16 x int32

      __m512i p32 = _mm512_loadu_si512((const void *)(sip_ptr + j));       // 16 x int32

      __m512i prod = _mm512_mullo_epi32(s32, p32);

      // use the AVX-512 reduce intrinsic instead of manual summation
      int32_t sum = _mm512_reduce_add_epi32(prod);
      acc += sum;
    }

#elif defined(__AVX2__)
    /* AVX2 intrinsics */
    int j8 = (L / 8) * 8;
    for (; j < j8; j += 8) {
      __m128i s16 = _mm_loadu_si128((const __m128i *)(sig_ptr + j));      // 8 x int16
      __m256i s32 = _mm256_cvtepi16_epi32(s16);                            // widen to 8 x int32

      __m256i p32 = _mm256_loadu_si256((const __m256i *)(sip_ptr + j));    // 8 x int32

      __m256i prod = _mm256_mullo_epi32(s32, p32);

      int32_t tmp[8];
      _mm256_storeu_si256((__m256i*)tmp, prod);

      int32_t sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      acc += sum;
    }
#endif

    // Finish remaining elements with scalar code
    // Or process all if no SIMD
    for (; j < L; j++) {
      acc += (int)sig_ptr[j] * sip_ptr[j];
    }

    correlation[i] = acc;

    if (acc > max_corr) {
      max_corr = acc;
      Preamble_offset = i;
    }
  }

  // Check if it is midamble instead of preamble
  int diff = frame->N_bit*frame->N_midamble_space + frame->preamble_samples;
  for(int i = Preamble_offset - diff; i >= 0; i -= diff) {
    if(abs(correlation[i] - max_corr) < max_corr/8) {
      max_corr = correlation[i];
      Preamble_offset = i;
    }
  }

  if(testing_mode && !testing_timing) {
    printf("[RX Synchronize] Preamble at offset %d, Corr: %d\n", Preamble_offset, max_corr);
  }
  return Preamble_offset;
}

int getpacket_snr_pass;
void AIOT_D2R_PHY_RX_GetPacket(uint8_t *rx_payload, const int16_t *signal, int Preamble_offset, NR_AIOT_UL_FRAME_PARMS *frame)
{
  int index = Preamble_offset + frame->preamble_samples;

  uint32_t *energy_plot = NULL;
  int energy_index = 0;

  if(testing_mode && !testing_timing && getpacket_snr_pass == snr_plot) {
    // Energy plotting
    energy_plot = malloc(2*frame->packet_size * sizeof(uint32_t));
  }

  int i = 0;

  for(; i < frame->packet_size && (index + frame->N_bit) < frame->packet_payload_size; i++) {
    if(i % frame->N_midamble_space == 0 && i != 0) {
      // Insert midamble (fixed pattern 101010...)
      index += frame->midamble_samples;

      if(index + frame->N_bit >= frame->packet_payload_size) {
        break;
      }
    }

    int sum1 = 0;
    int sum2 = 0;

    // Repeat each bit N_SFS times with OOK modulation
    for(int j = 0; j < frame->N_SFS; j++) {
      for(int k = 0; k < frame->N_chip; k++) {
        sum1 += signal[index++];
      }
      for(int k = 0; k < frame->N_chip; k++) {
        sum2 += signal[index++];
      }
    }

    if(testing_mode && !testing_timing && getpacket_snr_pass == snr_plot) {
      // Save energy for plotting
      energy_plot[energy_index++] = sum1;
      energy_plot[energy_index++] = sum2;
    }

    // Decision based on sums
    int bit0 = (sum1 > sum2) ? 0 : 1;
    rx_payload[i/8] = (rx_payload[i/8] << 1) | bit0;
  }

  frame->packet_payload_size = frame->packet_size; // actual size in bits

  int remainder = frame->packet_payload_size % 8;
  int shift = 8 - remainder;

  rx_payload[i/8] <<= shift; // shift to align last byte

  if(testing_mode && !testing_timing && getpacket_snr_pass == snr_plot) {
    sprintf(filename, "%s/D2R_Energy.m", folderplots);
    LOG_M(filename, "Energy_sig", energy_plot, energy_index, 1, 2);
    
    free(energy_plot);
  }
  getpacket_snr_pass++;
}

bool AIOT_D2R_PHY_RX_CheckCRC(uint8_t *packet, NR_AIOT_UL_FRAME_PARMS *frame)
{
  if (frame->packet_payload_size <= 0) return false;

  int crc_bits = (frame->packet_payload_size > 30) ? 16 : 6;
  int payloadBits = frame->packet_payload_size - crc_bits;
  uint32_t crc_calculated = 0;
  uint16_t crc_received = 0;

  if (crc_bits == 16) {
    crc_calculated = crc16((unsigned char *)packet, payloadBits) >> 16;
  } else {
    crc_calculated = crc6((unsigned char *)packet, payloadBits) >> 26;
  }

  for (int i = 0; i < crc_bits; i++) {
    int bitpos = payloadBits + i;
    int byte_idx = bitpos / 8;
    int bit_idx = 7 - (bitpos & 0x7);
    uint8_t bit = (packet[byte_idx] >> bit_idx) & 0x1;
    crc_received = (crc_received << 1) | bit;
  }

  if(testing_mode && !testing_timing) {
    if(crc_calculated == crc_received) {
      printf("[RX CRC] CRC check passed\n");
    } else {
      printf("[RX CRC] CRC error: Calculated: %X, Received: %X\n", crc_calculated, crc_received);
    }
  }

  return (crc_calculated == crc_received);
}

time_stats_t time_stats = {0};

// Thread data structure for parallel SNR processing
typedef struct {
  int snr_start;
  int snr_end;
  int thread_id;
  NR_AIOT_UL_FRAME_PARMS *frame_parms; // Shared (read-only)
  NR_AIOT_UL_FRAME_PARMS thread_frame_parms; // Thread-local copy for writes
  channel_model_t *channel_model;
  double *ber_results;
  pthread_mutex_t *print_mutex;

  // Per-thread timing results
  double time_tx_CRC;
  double time_tx_packet;
  double time_tx_filter;
  double time_channel;
  double time_multipath;
  double time_noise;
  double time_rx_envelope;
  double time_rx_filter;
  double time_rx_sync;
  double time_rx_packet;
  double time_ber;
  double time_total;

} snr_thread_data_t;

// Thread function for processing SNR range
void* process_snr_range(void* arg) {
  snr_thread_data_t *data = (snr_thread_data_t*)arg;

  // Create thread-local copy of frame_parms to avoid race conditions on received_M and packet_payload_size
  NR_AIOT_UL_FRAME_PARMS *local_frame_parms = &data->thread_frame_parms;
  memcpy(local_frame_parms, data->frame_parms, sizeof(NR_AIOT_UL_FRAME_PARMS));
  
  // Per-thread variables to avoid conflicts
  channel_desc_t *channel_params = new_channel_desc_scm(local_frame_parms->nr_frame_parms.nb_antennas_tx,
                                        local_frame_parms->nr_frame_parms.nb_antennas_rx,
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

  c16_t *txData = malloc(local_frame_parms->packet_samples * sizeof(c16_t));
  c16_t *txFiltered = malloc(local_frame_parms->packet_samples * sizeof(c16_t));
  
  int rx_size = local_frame_parms->packet_samples + data->channel_model->delay + 2048;
  c16_t **rxData = malloc(local_frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    rxData[i] = calloc(1, rx_size * sizeof(c16_t));
  }
  
  int16_t *envelope = malloc(rx_size * sizeof(int16_t));
  int16_t *filteredData = malloc(rx_size * sizeof(int16_t));
  int *correlation = malloc(rx_size * sizeof(int));
  uint8_t *rx_payload = malloc(MAX_AIOT_D2R_PACKET_SIZE);

  // Thread-local payload buffer
  uint8_t *payload = malloc(local_frame_parms->packet_size / 8 + 1);
  
  // Thread-local RX filter
  Butter3_Q15 rx_filter;
  AIOT_D2R_PHY_RX_Design_Filter(&rx_filter, 15e3/2 * local_frame_parms->N_SFS, (double)data->channel_model->sampling_rate * 1e6);
  
  // Thread-local IQ signal buffers
  double **s_re = malloc(local_frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  double **s_im = malloc(local_frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    s_re[i] = calloc(1, rx_size * sizeof(double));
    s_im[i] = calloc(1, rx_size * sizeof(double));
  }
  
  double **r_re = malloc(local_frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  double **r_im = malloc(local_frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    r_re[i] = calloc(1, rx_size * sizeof(double));
    r_im[i] = calloc(1, rx_size * sizeof(double));
  }
  
  // Thread-local Gaussian noise generator
  gaussZiggurat_MT_t gz = {0};
  
  // Thread-local timing stats
  time_stats_t local_time_stats = {0};
  char filename[128] = {0};
  
  // Process SNR range assigned to this thread
  for(int snr = data->snr_start; snr <= data->snr_end; snr += SNR_STEP_DB) {
    channel_model_t local_channel_model = *data->channel_model;
    local_channel_model.SNR = snr;
    
    for(int iters = 0; iters < snr_iters; iters++) {
      if(testing_mode && !testing_timing) {
        pthread_mutex_lock(data->print_mutex);
        printf("*************************\n");
        printf("Thread %d: Testing SNR %d dB\n", data->thread_id, snr);
        printf("-------------------------\n");
        pthread_mutex_unlock(data->print_mutex);
      }

      if(testing_timing) {
        start_meas(&local_time_stats);
      }

      memset(rx_payload, 0, MAX_AIOT_D2R_PACKET_SIZE);

      // Generate new payload for each trial
      for(int i = 0; i < local_frame_parms->payload_size / 8; i++) {
        payload[i] = uniformrandom() * 256;
      }
      for(int i = local_frame_parms->payload_size / 8; i < local_frame_parms->packet_size / 8 + 1; i++) {
        payload[i] = 0;
      }
      
      // Add CRC to payload
      AIOT_D2R_PHY_TX_AddCRC(payload, payload, local_frame_parms);
      
      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_tx_CRC += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_D2R_PHY_TX_Signal(txData, (const uint8_t *) payload, local_frame_parms);

      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        sprintf(filename, "%s/D2R_TX_Packet.m", folderplots);
        LOG_M(filename, "TX_Packet_sig", txData, local_frame_parms->packet_samples, 1, 1);
      }
      
      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_tx_packet += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_tx_filter += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      SIM_Channel_propagate(rxData, (const c16_t *) txData, channel_params, local_channel_model.SNR, local_frame_parms,
                            s_re, s_im, r_re, r_im, &data->time_multipath, &data->time_noise, &gz);
      
      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        // Save channel output
        double *output = malloc(rx_size * 2 * sizeof(double));

        for (int i = 0; i < rx_size; i++) {
          output[2 * i] = r_re[0][i];
          output[2 * i + 1] = r_im[0][i];
        }
        
        sprintf(filename, "%s/D2R_Channel.m", folderplots);
        LOG_M(filename, "Channel_sig", output, rx_size, 1, 8);

        free(output);
      }

      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        sprintf(filename, "%s/D2R_RX_IQ.m", folderplots);
        LOG_M(filename, "RX_IQ_sig", rxData[0], rx_size, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_channel += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_D2R_PHY_RX_Envelope_Detector(envelope, (const c16_t **) rxData, rx_size);
      
      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        sprintf(filename, "%s/D2R_Envelope.m", folderplots);
        LOG_M(filename, "Envelope_sig", envelope, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_rx_envelope += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      AIOT_D2R_PHY_RX_Filter(filteredData, (const int16_t *) envelope, rx_size, &rx_filter);
      
      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        sprintf(filename, "%s/D2R_Filter.m", folderplots);
        LOG_M(filename, "Filter_sig", filteredData, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_rx_filter += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      int Preamble_offset = AIOT_D2R_PHY_RX_Synchronize(correlation, (const int16_t *) envelope, Preamble_ideal, local_frame_parms);
      
      if(testing_mode && !testing_timing && snr == snr_plot && iters == 0) {
        sprintf(filename, "%s/D2R_Correlation.m", folderplots);
        LOG_M(filename, "Correlation_sig", correlation, rx_size - local_frame_parms->preamble_samples, 1, 2);
      }

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_rx_sync += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      local_frame_parms->packet_payload_size = rx_size;
      AIOT_D2R_PHY_RX_GetPacket(rx_payload, (const int16_t *) envelope, Preamble_offset, local_frame_parms);
      
      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_rx_packet += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
        start_meas(&local_time_stats);
      }
      
      bool crc_ok = AIOT_D2R_PHY_RX_CheckCRC(rx_payload, local_frame_parms);

      if(testing_timing) {
        stop_meas(&local_time_stats);
        data->time_ber += local_time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&local_time_stats);
      }

      if(!crc_ok) {
        data->ber_results[snr - snr_min] += 1;

        if(testing_mode && !testing_timing) {
          printf("Transmitted Packet (%d bits):\n", local_frame_parms->packet_size);
          for(int i = 0; i < (local_frame_parms->packet_size+7) / 8; i++) {
            printf("%02X ", payload[i]);
          }
          printf("\n");

          printf("Received packet    (%d bits):\n", local_frame_parms->packet_payload_size);
          for(int i = 0; i < (local_frame_parms->packet_payload_size+7)/8; i++) {
            printf("%02X ", rx_payload[i]);
          }
          printf("\n");
        }
      }
    }
    
    data->ber_results[snr - snr_min] /= snr_iters;
    
    pthread_mutex_lock(data->print_mutex);
    printf("Thread %d: Completed SNR %3d dB: BLER = %f\n", data->thread_id, snr, data->ber_results[snr - snr_min]);
    pthread_mutex_unlock(data->print_mutex);
  }
  
  // Cleanup thread-local resources
  free(txData);
  free(txFiltered);
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    free(rxData[i]);
  }
  free(rxData);
  free(envelope);
  free(filteredData);
  free(correlation);
  free(rx_payload);
  free(payload);
  
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    free(s_re[i]);
    free(s_im[i]);
  }
  free(s_re);
  free(s_im);
  
  for (int i = 0; i < local_frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }
  free(r_re);
  free(r_im);
  
  free_channel_desc_scm(channel_params);
  
  return NULL;
}

void BLER_test(NR_AIOT_UL_FRAME_PARMS *frame_parms, channel_model_t *channel_model)
{
  AIOT_D2R_PHY_TX_calc_packet_sizes(frame_parms);

  printf("*************************\n");
  printf("D2R packet parameters:\n");
  printf("  Bit duration samples: %d\n", frame_parms->N_bit);
  printf("  Chip duration samples: %d\n", frame_parms->N_chip);
  printf("  Small frequency shift factor: %d\n", frame_parms->N_SFS);
  printf("  Preamble length: %s\n", (frame_parms->L_preamble) ? "long (31 bits)" : "short (7 bits)");
  printf("  Additional midamble: %s\n", (frame_parms->I_add) ? "yes" : "no");
  printf("  Midamble interval: %d bits\n", frame_parms->N_midamble_space);
  printf("  Channel coding: %s\n", (frame_parms->R_code) ? "FEC" : "No FEC");
  printf("  Block repetition: %s\n", (frame_parms->R_block) ? "2" : "1");
  printf("  Payload size: %d bits\n", frame_parms->payload_size);

  // Setup radio channel
  channel_model->sampling_rate = 1.92;
  channel_model->bw = 0.015;

  printf("Channel parameters:\n");
  printf("  Channel model: %s\n", channel_model->channel_model == AWGN ? "AWGN" : "TDL");
  printf("  Sampling rate: %f MHz\n", channel_model->sampling_rate);
  printf("  Bandwidth: %f MHz\n", channel_model->bw);
  printf("  Delay: %d samples\n", channel_model->delay);

  printf("*************************\n");
  printf("Starting BLER vs SNR: \n");
  printf("  SNR range: %d to %d dB\n", snr_min, snr_max);
  printf("  Iterations per SNR point: %d\n", snr_iters);
  printf("*************************\n");

  // Generate Preamble ideal sequence (shared by all threads)
  Preamble_ideal = generate_preamble_ideal_sequence(frame_parms);

  if(testing_mode) {
    sprintf(filename, "%s/D2R_Preamble_Ideal.m", folderplots);
    LOG_M(filename, "Preamble_Ideal_sig", Preamble_ideal, frame_parms->preamble_samples, 1, 2);
  }

  double ber_results[snr_steps];
  memset(ber_results, 0, snr_steps * sizeof(double));

  if(!testing_mode) {
    int total_snr_points = (snr_max - snr_min) / SNR_STEP_DB + 1;
    
    if (num_threads > total_snr_points) {
      num_threads = total_snr_points;
    }
    
    printf("Using %d threads for parallel processing\n", num_threads);

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    snr_thread_data_t *thread_data = malloc(num_threads * sizeof(snr_thread_data_t));
    memset(thread_data, 0, num_threads * sizeof(snr_thread_data_t));

    // Calculate SNR range for each thread
    int snr_per_thread = total_snr_points / num_threads;
    int remaining_snr = total_snr_points % num_threads;
    
    for (int t = 0; t < num_threads; t++) {
      thread_data[t].thread_id = t;
      thread_data[t].frame_parms = frame_parms;
      thread_data[t].channel_model = channel_model;
      thread_data[t].ber_results = ber_results;
      thread_data[t].print_mutex = &print_mutex;
      
      // Calculate SNR range for this thread
      int start_idx = t * snr_per_thread + (t < remaining_snr ? t : remaining_snr);
      int end_idx = start_idx + snr_per_thread + (t < remaining_snr ? 1 : 0) - 1;
      
      thread_data[t].snr_start = snr_min + start_idx * SNR_STEP_DB;
      thread_data[t].snr_end = snr_min + end_idx * SNR_STEP_DB;
      
      printf("Thread %d: SNR range %3d to %d dB\n", t, thread_data[t].snr_start, thread_data[t].snr_end);
    }

    printf("-------------------------------\n");
    printf("         Simulation run\n");
    printf("-------------------------------\n");
    
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
    
    for (int t = 0; t < num_threads; t++) {
      thread_data[0].time_tx_CRC += thread_data[t].time_tx_CRC;
      thread_data[0].time_tx_packet += thread_data[t].time_tx_packet;
      thread_data[0].time_tx_filter += thread_data[t].time_tx_filter;
      thread_data[0].time_channel += thread_data[t].time_channel;
      thread_data[0].time_multipath += thread_data[t].time_multipath;
      thread_data[0].time_noise += thread_data[t].time_noise;
      thread_data[0].time_rx_envelope += thread_data[t].time_rx_envelope;
      thread_data[0].time_rx_filter += thread_data[t].time_rx_filter;
      thread_data[0].time_rx_sync += thread_data[t].time_rx_sync;
      thread_data[0].time_rx_packet += thread_data[t].time_rx_packet;
      thread_data[0].time_ber += thread_data[t].time_ber;
      thread_data[0].time_total += thread_data[t].time_total;
    }
      
    // Average the timing results
    int total_measurements = snr_steps * snr_iters;

    thread_data[0].time_tx_CRC /= total_measurements;
    thread_data[0].time_tx_packet /= total_measurements;
    thread_data[0].time_tx_filter /= total_measurements;
    thread_data[0].time_channel /= total_measurements;
    thread_data[0].time_multipath /= total_measurements;
    thread_data[0].time_noise /= total_measurements;
    thread_data[0].time_rx_envelope /= total_measurements;
    thread_data[0].time_rx_filter /= total_measurements;
    thread_data[0].time_rx_sync /= total_measurements;
    thread_data[0].time_rx_packet /= total_measurements;
    thread_data[0].time_ber /= total_measurements;
    thread_data[0].time_total /= total_measurements;

    if(testing_timing) {
      printf("-------------------------------\n");
      printf("Timing results (average per packet in us):\n");
      printf("-------------------------------\n");
      printf("TX CRC addition:         %f us\n", thread_data[0].time_tx_CRC);
      printf("TX signal generation:    %f us\n", thread_data[0].time_tx_packet);
      printf("TX filtering:            %f us\n", thread_data[0].time_tx_filter);
      printf("Channel propagation:     %f us\n", thread_data[0].time_channel);
      printf("  of which multipath:    %f us\n", thread_data[0].time_multipath);
      printf("  of which noise:        %f us\n", thread_data[0].time_noise);
      printf("Envelope detection:      %f us\n", thread_data[0].time_rx_envelope);
      printf("Filtering:               %f us\n", thread_data[0].time_rx_filter);
      printf("Synchronization:         %f us\n", thread_data[0].time_rx_sync);
      printf("RX packet extraction:    %f us\n", thread_data[0].time_rx_packet);
      printf("BER calculation:         %f us\n", thread_data[0].time_ber);
      double total_time = thread_data[0].time_tx_packet + thread_data[0].time_channel + thread_data[0].time_rx_envelope +
                          thread_data[0].time_rx_filter + thread_data[0].time_rx_sync + thread_data[0].time_rx_packet + thread_data[0].time_ber;
      printf("---\n");
      printf("Total time:              %f us\n", total_time);
    }

    // Cleanup
    free(threads);
    free(thread_data);

  } else { // testing mode
    snr_thread_data_t thread_data;
    
    thread_data.thread_id = 0;
    thread_data.frame_parms = frame_parms;
    thread_data.channel_model = channel_model;
    thread_data.ber_results = ber_results;
    thread_data.print_mutex = &print_mutex;
    
    thread_data.snr_start = snr_min;
    thread_data.snr_end = snr_max;
    
    // Create and start threads
    process_snr_range(&thread_data);
  }
  
  free(Preamble_ideal);

  printf("-------------------------------\n");
  printf("            Results\n");
  printf("-------------------------------\n");
  for(int snr = snr_min; snr <= snr_max; snr += SNR_STEP_DB) {
    printf("SNR %3d dB: BLER = %f\n", snr, ber_results[snr - snr_min]);
  }

  if(!testing_mode) {
    sprintf(filename, "%s/BLER_SIZE%d_RSFS%d_PREAMB%d_BIT%d.m", foldername, frame_parms->payload_size, frame_parms->R_SFS, frame_parms->L_preamble ? 1 : 0, frame_parms->T_bit);
    LOG_M(filename, "BLER", ber_results, snr_max - snr_min + 1, 1, 7);
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
  num_threads = sysconf(_SC_NPROCESSORS_ONLN);

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_AIOT_PDRCHSIM] Error, configuration module init failed\n");
  }

  printf("===========================================================\n");
  printf("            Ambient-IoT Rel 19 PDRCH Simulator\n");
  printf("===========================================================\n\n");

  // **************************
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
  frame_parms->I_add = false; // additional midamble
  frame_parms->R_code = false; // no FEC

  // Payload size
  frame_parms->payload_size = PAYLOAD_SIZE_DEFAULT;

  // Fill in channel model default parameters
  channel_model_t channel_model = {
    .channel_model = AWGN,
    .fc = 897500000, // Carrier frequency n8 band, #50 RB
    .DS_TDL = .03,
    .SNR = 20.0,
    .path_loss_dB = -15.0,
    .noise_power_dB = -120.0,
    .delay = 1500,
    .tx_pwr_dBm = 46.0
  };

  int c;
  while ((c = getopt(argc, argv, "--:O:h:L:p:f:I:l:a:b:i:s:S:t:T:")) != -1) {
    /* ignore long options starting with '--', option '-O' and their arguments that are handled by configmodule */
    /* with this opstring getopt returns 1 for non-option arguments, refer to 'man 3 getopt' */
    if (c == 1 || c == '-' || c == 'O')
      continue;

    printf("handling optarg %c\n", c);
    switch (c) {
      default:
      case 'h':
        printf("%s <options>\n", argv[0]);
        printf("-h This help page\n");
        printf("-L OAI log level <0(errors) default, 1(warning), 2(analysis), 3(info), 4(debug), 5(trace)>\n");
        printf("-p Payload size in bits (max: %d) (default: %d)\n", MAX_AIOT_D2R_PAYLOAD_SIZE * 8, frame_parms->payload_size);
        /*printf("-R Channel coding: 0=No FEC, 1=FEC\n");
        printf("-r Block repetition: 1=1, 2=2\n");*/
        printf("-f Small frequency shift factor: 0 to 2 (default: %d)\n", frame_parms->R_SFS);
        printf("-I Midamble interval: 0 (S*48), 1 (S*96), 2 (S*168), 3 (S*240 bits) (see 3GPP TS 38.391) (default: %d)\n", frame_parms->I_bit);
        printf("-l Preamble length: 0 (short(7 bits)), 1 (long(31 bits)) (default: %d)\n", frame_parms->L_preamble ? 1 : 0);
        printf("-a Additional midamble: 0=no, 1=yes (default: %d)\n", frame_parms->I_add ? 1 : 0);
        printf("-b Bit duration option T_bit: 0 (T_bit=2*tau), 1 (tau), 2 (1/2*tau), 3 (1/4*tau) (see 3GPP TS 38.391) (default: %d)\n", frame_parms->T_bit);
        
        printf("-i Iterations per SNR point (default: %d)\n", SNR_TRIALS);
        printf("-s SNR min in dB (default: %d)\n", MIN_SNR_DB);
        printf("-S SNR max in dB (default: %d)\n", MAX_SNR_DB);

        printf("\n*** Testing options:\n");
        printf("-t Testing mode (the parameter specifies the SNR cut to save to plot)\n");
        printf("-T Timing mode (measure processing time per packet)\n");
        exit(-1);
        break;

      // Change log level
      case 'L':
        loglvl = atoi(optarg);
        break;

      // Change payload size
      case 'p':
        frame_parms->payload_size = atoi(optarg);
        if (frame_parms->payload_size > MAX_AIOT_D2R_PAYLOAD_SIZE * 8) {
          printf("Error: maximum payload bit size is %d\n", MAX_AIOT_D2R_PAYLOAD_SIZE * 8);
          exit(-1);
        } else {
          printf("Using payload size %d bits\n", frame_parms->payload_size);
        }
        break;

      // Testing mode
      case 't':
        if (testing_timing) {
          printf("Error: Timing and testing modes cannot be enabled simultaneously\n");
          exit(-1);
        }

        snr_iters = 1;
        testing_mode = true;
        snr_plot = atoi(optarg);
        printf("Enabling testing mode, saving results for SNR=%d dB\n", snr_plot);
        break;
      
      // Timing mode
      case 'T':
        if (testing_mode) {
          printf("Error: Timing and testing modes cannot be enabled simultaneously\n");
          exit(-1);
        }
        
        printf("Enabling timing mode\n");
        testing_timing = true;
        break;

      // Change iterations per SNR
      case 'i':
        if (testing_mode) {
          printf("Error: Iterations per SNR point cannot be set in testing mode\n");
          exit(-1);
        }

        snr_iters = atoi(optarg);
        printf("Using %d iterations per SNR point\n", snr_iters);
        break;

      // Change SNR min
      case 's':
        if(snr_min < MIN_SNR_DB || snr_min > MAX_SNR_DB) {
          printf("Error: SNR min must be between %d and %d dB\n", MIN_SNR_DB, MAX_SNR_DB);
          exit(-1);
        }
        snr_min = atoi(optarg);
        snr_steps = ((snr_max - snr_min) / SNR_STEP_DB + 1);
        printf("Using SNR min: %d dB\n", snr_min);
        break;

      // Change SNR max
      case 'S':
        if(snr_max < MIN_SNR_DB || snr_max > MAX_SNR_DB) {
          printf("Error: SNR max must be between %d and %d dB\n", MIN_SNR_DB, MAX_SNR_DB);
          exit(-1);
        }
        snr_max = atoi(optarg);
        snr_steps = ((snr_max - snr_min) / SNR_STEP_DB + 1);
        printf("Using SNR max: %d dB\n", snr_max);
        break;

      // Set small frequency shift factor
      case 'f':
        if(atoi(optarg) < 0 || atoi(optarg) > 2) {
          printf("Error: R_SFS must be between 0 and 2\n");
          exit(-1);
        }
        frame_parms->R_SFS = atoi(optarg);
        printf("Using small frequency shift factor: %d\n", frame_parms->R_SFS);
        break;

      // Set midamble interval
      case 'I':
        if(atoi(optarg) < 0 || atoi(optarg) > 3) {
          printf("Error: I_bit must be between 0 and 3\n");
          exit(-1);
        }
        frame_parms->I_bit = atoi(optarg);
        printf("Using midamble interval: %d bits\n", I_BIT_TABLE[frame_parms->I_bit]);
        break;

      // Set preamble length
      case 'l':
        if(atoi(optarg) != 0 && atoi(optarg) != 1) {
          printf("Error: L_preamble must be 0 for short or 1 for long\n");
          exit(-1);
        }
        frame_parms->L_preamble = (atoi(optarg) == 1) ? true : false;
        printf("Using preamble length: %s\n", (frame_parms->L_preamble) ? "long (31 bits)" : "short (7 bits)");
        break;

      // Set additional midamble
      case 'a':
        if(atoi(optarg) != 0 && atoi(optarg) != 1) {
          printf("Error: I_add must be 0 for no or 1 for yes\n");
          exit(-1);
        }
        frame_parms->I_add = (atoi(optarg) == 1) ? true : false;
        printf("Using additional midamble: %s\n", (frame_parms->I_add) ? "yes" : "no");
        break;

      // Set bit duration
      case 'b':
        if(atoi(optarg) < 0 || atoi(optarg) > 3) {
          printf("Error: T_bit must be between 0 and 3\n");
          exit(-1);
        }
        frame_parms->T_bit = atoi(optarg);
        printf("Using bit duration option T_bit=%d\n", frame_parms->T_bit);
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

  printf("CPU features:\n");
  printf("  Threads: %d\n", num_threads);

  #if defined(__AVX2__)
    printf("  AVX2 supported\n");
  #else
    printf("  AVX2 not supported\n");
  #endif
  
  #if defined(__AVX512F__) && defined(__AVX512BW__)
    printf("  AVX512 supported\n");
  #else
    printf("  AVX512 not supported\n");
  #endif

  // ---------------------------------------------------------------

  getpacket_snr_pass = snr_min;

  // Create folders for results
  mkdir("./D2R_plots", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir(foldername, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if(testing_mode) {
    sprintf(folderplots, "./D2R_plots/SNR%d_RSFS%d_PREAMB%d_BIT%d_SIZE%d/", snr_plot, frame_parms->R_SFS, frame_parms->L_preamble ? 1 : 0, frame_parms->I_bit, frame_parms->payload_size);
    mkdir(folderplots, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  // Start BLER test
  BLER_test(frame_parms, &channel_model);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
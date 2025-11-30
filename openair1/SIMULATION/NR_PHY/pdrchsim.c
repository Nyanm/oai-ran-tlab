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
#define SNR_TRIALS 100
#define SNR_STEPS ((MAX_SNR_DB - MIN_SNR_DB) / SNR_STEP_DB + 1)

int snr_min = MIN_SNR_DB;
int snr_max = MAX_SNR_DB;
int snr_steps = SNR_STEPS;
int snr_trials = SNR_TRIALS;
int snr_plot = 0;

int rx_size;

double **s_re, **s_im; // TX signal
double **r_re, **r_im; // RX signal
channel_model_t channel_model;

bool testing_mode = false;
bool testing_timing = false;

// --- Fixed Point Configuration ---
// Data: Q1.15 [-1, 1)
// Coeffs: Q2.29 [-4, 4) to handle high sample rate precision
#define SHIFT_COEFF 29

// Helper: Convert float to Q2.29
int32_t dbl_to_fixed(double x) {
    return (int32_t)(x * (1 << SHIFT_COEFF));
}

// Helper: Saturation (Clamp to valid int16 range)
// Prevents wrapping if the filter overshoots 32767
int16_t saturate_q15(int64_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/* 3rd Order Butterworth Filter State 
   - Coefficients are shared between Real/Imag paths.
   - State history (x, y) must be separate for Real/Imag.
*/
typedef struct {
    // --- Coefficients (Q2.29) ---
    int32_t b0_1, b1_1, a1_1;             // Stage 1 (1st order)
    int32_t b0_2, b1_2, b2_2, a1_2, a2_2; // Stage 2 (2nd order)

    // --- State History: REAL Path (Q1.15) ---
    int16_t x1_r, y1_r;           
    int16_t x2_1_r, x2_2_r;       
    int16_t y2_1_r, y2_2_r;       

    // --- State History: IMAG Path (Q1.15) ---
    int16_t x1_i, y1_i;           
    int16_t x2_1_i, x2_2_i;       
    int16_t y2_1_i, y2_2_i;       
} Butter3_c16;

void butter3_c16_clear(Butter3_c16 *f) {
  // 1. Clear States
  f->x1_r = 0;
  f->y1_r = 0;
  f->x2_1_r = 0;
  f->x2_2_r = 0;
  f->y2_1_r = 0;
  f->y2_2_r = 0;
  
  f->x1_i = 0;
  f->y1_i = 0;
  f->x2_1_i = 0;
  f->x2_2_i = 0;
  f->y2_1_i = 0;
  f->y2_2_i = 0;
}

// Initialize Filter
void butter3_c16_init(Butter3_c16 *f, double Fc, double Fs) {
  // 1. Clear States
  butter3_c16_clear(f);

  // 2. Calculate Coefficients (Bilinear Transform)
  double omega = 2.0 * M_PI * Fc;
  double T = 1.0 / Fs;
  double wa = (2.0 / T) * tan(omega * T / 2.0);

  // Stage 1: 1st Order Section
  double g1 = wa * T / 2.0;
  double D1 = 1.0 + g1;
  
  f->b0_1 = dbl_to_fixed(g1 / D1);
  f->b1_1 = dbl_to_fixed(g1 / D1);
  f->a1_1 = dbl_to_fixed((g1 - 1.0) / D1);

  // Stage 2: 2nd Order Section
  double g2 = wa * T / 2.0;
  double D2 = 1.0 + g2 + g2 * g2;

  f->b0_2 = dbl_to_fixed((g2 * g2) / D2);
  f->b1_2 = dbl_to_fixed(2.0 * (g2 * g2) / D2);
  f->b2_2 = dbl_to_fixed((g2 * g2) / D2);
  f->a1_2 = dbl_to_fixed((2.0 * g2 * g2 - 2.0) / D2);
  f->a2_2 = dbl_to_fixed((1.0 - g2 + g2 * g2) / D2);

  printf("Filter Initialized (c16_t)\n");
}

// Process Block of c16_t data
void butter3_c16_process(Butter3_c16 *f, const c16_t *input, c16_t *output, size_t length) {
  for (size_t i = 0; i < length; i++) {
    
    // ==============================
    // PATH 1: REAL COMPONENT
    // ==============================
    int16_t in_r = input[i].r;
    
    // Stage 1 (1st Order)
    // Q1.15 * Q2.29 = Q3.44 (requires 64-bit accumulator)
    int64_t acc1_r = 0;
    acc1_r += (int64_t)in_r * f->b0_1;
    acc1_r += (int64_t)f->x1_r * f->b1_1;
    acc1_r -= (int64_t)f->y1_r * f->a1_1;
    
    // Shift back to Q15 (with rounding) and Saturate
    int16_t st1_out_r = saturate_q15((acc1_r + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF);
    
    f->x1_r = in_r; 
    f->y1_r = st1_out_r;

    // Stage 2 (2nd Order)
    int64_t acc2_r = 0;
    acc2_r += (int64_t)st1_out_r * f->b0_2;
    acc2_r += (int64_t)f->x2_1_r * f->b1_2;
    acc2_r += (int64_t)f->x2_2_r * f->b2_2;
    acc2_r -= (int64_t)f->y2_1_r * f->a1_2;
    acc2_r -= (int64_t)f->y2_2_r * f->a2_2;
    
    int16_t final_r = saturate_q15((acc2_r + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF);

    f->x2_2_r = f->x2_1_r; f->x2_1_r = st1_out_r;
    f->y2_2_r = f->y2_1_r; f->y2_1_r = final_r;

    // ==============================
    // PATH 2: IMAGINARY COMPONENT
    // ==============================
    int16_t in_i = input[i].i;

    // Stage 1 (1st Order)
    int64_t acc1_i = 0;
    acc1_i += (int64_t)in_i * f->b0_1;
    acc1_i += (int64_t)f->x1_i * f->b1_1;
    acc1_i -= (int64_t)f->y1_i * f->a1_1;
    
    int16_t st1_out_i = saturate_q15((acc1_i + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF);

    f->x1_i = in_i; 
    f->y1_i = st1_out_i;

    // Stage 2 (2nd Order)
    int64_t acc2_i = 0;
    acc2_i += (int64_t)st1_out_i * f->b0_2;
    acc2_i += (int64_t)f->x2_1_i * f->b1_2;
    acc2_i += (int64_t)f->x2_2_i * f->b2_2;
    acc2_i -= (int64_t)f->y2_1_i * f->a1_2;
    acc2_i -= (int64_t)f->y2_2_i * f->a2_2;
    
    int16_t final_i = saturate_q15((acc2_i + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF);

    f->x2_2_i = f->x2_1_i; f->x2_1_i = st1_out_i;
    f->y2_2_i = f->y2_1_i; f->y2_1_i = final_i;

    // ==============================
    // OUTPUT
    // ==============================
    output[i].r = final_r;
    output[i].i = final_i;
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

  // --- State History (Q1.15) ---
  int16_t x1, y1;               // Stage 1 history
  int16_t x2_1, x2_2;           // Stage 2 input history
  int16_t y2_1, y2_2;           // Stage 2 output history
} Butter3_Q15;

void butter3_clear(Butter3_Q15 *f) {
  f->x1 = 0;
  f->y1 = 0;

  f->x2_1 = 0;
  f->x2_2 = 0;
  f->y2_1 = 0;
  f->y2_2 = 0;
}

void butter3_init(Butter3_Q15 *f, double Fc, double Fs) {
  // 1. Clear State
  butter3_clear(f);

  // 2. Bilinear Transform Math (Pre-warping)
  double omega = 2.0 * M_PI * Fc;
  double T = 1.0 / Fs;
  double wa = (2.0 / T) * tan(omega * T / 2.0);

  // 3. Calculate Stage 1 (1st Order)
  // H(s) = 1 / (s + 1) normalized
  double gamma1 = wa * T / 2.0;
  double D1 = 1.0 + gamma1;
  
  f->b0_1 = dbl_to_fixed(gamma1 / D1);
  f->b1_1 = dbl_to_fixed(gamma1 / D1);
  f->a1_1 = dbl_to_fixed((gamma1 - 1.0) / D1);

  // 4. Calculate Stage 2 (2nd Order)
  // H(s) = 1 / (s^2 + s + 1) normalized
  double gamma2 = wa * T / 2.0;
  double D2 = 1.0 + gamma2 + gamma2 * gamma2;

  f->b0_2 = dbl_to_fixed((gamma2 * gamma2) / D2);
  f->b1_2 = dbl_to_fixed(2.0 * (gamma2 * gamma2) / D2);
  f->b2_2 = dbl_to_fixed((gamma2 * gamma2) / D2);
  f->a1_2 = dbl_to_fixed((2.0 * gamma2 * gamma2 - 2.0) / D2);
  f->a2_2 = dbl_to_fixed((1.0 - gamma2 + gamma2 * gamma2) / D2);

  printf("Filter Initialized for Q1.15 Data processing.\n");
}

void butter3_process(Butter3_Q15 *f, const int16_t *input, int16_t *output, size_t length) {
  for (size_t i = 0; i < length; i++) {
    int16_t in_sample = input[i];

    // --- STAGE 1: 1st Order ---
    // Math: (Q1.15 * Q2.29) = Q3.44. Accumulator needs 64-bit.
    int64_t acc1 = 0;
    acc1 += (int64_t)in_sample * f->b0_1;
    acc1 += (int64_t)f->x1     * f->b1_1;
    acc1 -= (int64_t)f->y1     * f->a1_1;

    // Rounding (add half LSB) and Shift back to Q1.15
    // Result is technically Q3.15 now, so we must saturate before casting to int16
    int64_t stage1_raw = (acc1 + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF;
    int16_t stage1_out = saturate_q15(stage1_raw);

    // Update Stage 1 State
    f->x1 = in_sample;
    f->y1 = stage1_out;

    // --- STAGE 2: 2nd Order ---
    int64_t acc2 = 0;
    acc2 += (int64_t)stage1_out * f->b0_2;
    acc2 += (int64_t)f->x2_1    * f->b1_2;
    acc2 += (int64_t)f->x2_2    * f->b2_2;
    acc2 -= (int64_t)f->y2_1    * f->a1_2;
    acc2 -= (int64_t)f->y2_2    * f->a2_2;

    // Rounding and shifting
    int64_t stage2_raw = (acc2 + (1 << (SHIFT_COEFF - 1))) >> SHIFT_COEFF;
    int16_t final_out = saturate_q15(stage2_raw);

    // Update Stage 2 State
    f->x2_2 = f->x2_1;
    f->x2_1 = stage1_out;
    f->y2_2 = f->y2_1;
    f->y2_1 = final_out;

    output[i] = final_out;
  }
}

/*  ************    BANDPASS BUTTERWORTH FILTER ************    */

#define Q_SHIFT 14
#define SCALE (1 << Q_SHIFT)

// Biquad section: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
typedef struct {
  int16_t b0, b1, b2;
  int16_t a1, a2;
  int16_t x1, x2;
  int16_t y1, y2;
} Biquad;

// 3rd Order Butterworth Bandpass (Total order 6)
typedef struct {
  Biquad sections[3];
} ButterworthBP3;

void BP_init(ButterworthBP3* filter) {
  memset(filter, 0, sizeof(ButterworthBP3));
}

static inline int16_t biquad_process(Biquad *bq, int16_t input) {
  int64_t accum;

  accum  = (int64_t)bq->b0 * input;
  accum += (int64_t)bq->b1 * bq->x1;
  accum += (int64_t)bq->b2 * bq->x2;
  accum -= (int64_t)bq->a1 * bq->y1;
  accum -= (int64_t)bq->a2 * bq->y2;

  // Rounding and scaling
  accum = (accum + (SCALE >> 1)) >> Q_SHIFT;

  bq->x2 = bq->x1;
  bq->x1 = input;
  bq->y2 = bq->y1;
  
  // Saturation
  if (accum > 32767) accum = 32767;
  if (accum < -32768) accum = -32768;
  
  bq->y1 = (int16_t)accum;

  return bq->y1;
}

int16_t BP_process(ButterworthBP3* filter, int16_t input) {
  int16_t signal = input;
  
  signal = biquad_process(&filter->sections[0], signal);
  signal = biquad_process(&filter->sections[1], signal);
  signal = biquad_process(&filter->sections[2], signal);
  
  return signal;
}

int16_t double_to_fixed(double val) {
  long raw = (long)(val * SCALE);
  if (raw > 32767) return 32767;
  if (raw < -32768) return -32768;
  return (int16_t)raw;
}

void BP_calculate_coefficients(double fs, double f_low, double f_high, ButterworthBP3* filter) {
  BP_init(filter);

  double w_low = 2.0 * M_PI * f_low;
  double w_high = 2.0 * M_PI * f_high;
  
  // Pre-warp
  double wa_low = tan(w_low / (2.0 * fs));
  double wa_high = tan(w_high / (2.0 * fs));
  
  double w0_sq = wa_low * wa_high;
  double bw = wa_high - wa_low;

  // Section 1: Derived from Real LP Pole
  {
      double r = 1.0; 
      double D = 1.0 + r * bw + w0_sq;
      
      double b0 = (r * bw) / D;
      double b1 = 0.0;
      double b2 = -(r * bw) / D;
      double a1 = (2.0 * (w0_sq - 1.0)) / D;
      double a2 = (1.0 - r * bw + w0_sq) / D;

      filter->sections[0].b0 = double_to_fixed(b0);
      filter->sections[0].b1 = double_to_fixed(b1);
      filter->sections[0].b2 = double_to_fixed(b2);
      filter->sections[0].a1 = double_to_fixed(a1);
      filter->sections[0].a2 = double_to_fixed(a2);
  }

  // Sections 2 & 3: Approximated Synchronous Bandpass sections
  for(int k=1; k<=2; k++) {
        double D = 1.0 + bw + w0_sq;
        double b0 = bw / D;
        double b2 = -bw / D;
        double a1 = (2.0 * (w0_sq - 1.0)) / D;
        double a2 = (1.0 - bw + w0_sq) / D;
        
        filter->sections[k].b0 = double_to_fixed(b0);
        filter->sections[k].b1 = 0;
        filter->sections[k].b2 = double_to_fixed(b2);
        filter->sections[k].a1 = double_to_fixed(a1);
        filter->sections[k].a2 = double_to_fixed(a2);
  }
}

// Clears filter history (state) to process a new signal without recalculating coefficients
void butterworth_reset(ButterworthBP3* filter) {
    for(int i=0; i<3; i++) {
        filter->sections[i].x1 = 0;
        filter->sections[i].x2 = 0;
        filter->sections[i].y1 = 0;
        filter->sections[i].y2 = 0;
    }
}

/* ***************          *************************  */

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

  // Compute samples for preamble and midamble
  frame->preamble_samples = (frame->N_preamble + 1) * frame->N_bit; // +1 for preceeding zero bit
  frame->midamble_samples = frame->N_preamble * frame->N_bit;

  // Calculate packet size
  frame->payload_size = payloadSize;
  frame->packet_samples = frame->payload_size;
  if(frame->I_add) {
    frame->packet_samples += (1 + ceil(frame->payload_size / (double)frame->N_midamble_space)) * frame->N_preamble;
  } else {
    frame->packet_samples += (1 + (frame->payload_size / frame->N_midamble_space)) * frame->N_preamble;
  }
  frame->packet_samples *= frame->N_bit;

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
  for(int i = 0; i < frame->payload_size; i++) {
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

void AIOT_D2R_PHY_TX_Filter(c16_t *out, const c16_t *in, int length, Butter3_c16 *filt)
{
  butter3_c16_clear(filt);
  butter3_c16_process(filt, in, out, length);
}

void SIM_Channel_propagate(c16_t **rxData, const c16_t *in, channel_desc_t *channel, double SNR, NR_AIOT_UL_FRAME_PARMS *frame)
{
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

  multipath_channel(channel, s_re, s_im, r_re, r_im, frame->packet_samples + channel->channel_offset + 200, 0, 1);
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
    
    sprintf(filename, "%s/D2R_Channel.m", foldername);
    LOG_M(filename, "Channel_sig", output, rx_size, 1, 8);

    free(output);
  }
}

void AIOT_D2R_PHY_RX_Envelope_Detector(int16_t *envelope, const c16_t **rxData, int rx_size)
{
  // Envelope detector (squared)
  for (int i = 0; i < rx_size; i++) {
    //envelope[i] = iSqrt(rxData[0][i].r * rxData[0][i].r + rxData[0][i].i * rxData[0][i].i);

    //envelope[i] = c16MulConjShift(rxData[0][i], rxData[0][i], 15).r;

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

void AIOT_D2R_PHY_RX_Filter(int16_t *out, const int16_t *in, int length, Butter3_Q15 *filt)
{
  butter3_clear(filt);
  butter3_process(filt, in, out, length);

  /*butterworth_reset(filt);
  for(int i = 0; i < length; i++) {
    out[i] = BP_process(filt, in[i]);
  }*/
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
  // Correlate received signal with ideal Preamble
  int Preamble_offset = 0;
  int max_corr = INT_MIN;

  for (int i = 0; i < rx_size - frame->preamble_samples; i++) {
    correlation[i] = 0;
    for (int j = 0; j < frame->preamble_samples; j++) {
      correlation[i] += signal[i + j] * Preamble_ideal[j];
    }

    if(correlation[i] > max_corr) {
      max_corr = correlation[i];
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

  if(testing_mode) {
    printf("Detected Preamble at offset %d\n", Preamble_offset);
  }
  return Preamble_offset;
}

#define CORR_THRESHOLD 1000000

void AIOT_D2R_PHY_RX_GetPacket(uint8_t *rx_payload, const int16_t *signal, int Preamble_offset, NR_AIOT_UL_FRAME_PARMS *frame)
{
  int index = Preamble_offset + frame->preamble_samples;
  frame->packet_payload_size = frame->payload_size;

  for(int i = 0; i < frame->packet_payload_size; i++) {
    if(i % frame->N_midamble_space == 0 && i != 0) {
      // Insert midamble (fixed pattern 101010...)
      index += frame->midamble_samples;
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

    // Decision based on sums
    int bit0 = (sum1 > sum2) ? 0 : 1;
    rx_payload[i/8] = (rx_payload[i/8] << 1) | bit0;
  }
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

void free_Preamble_ideal_sequence(int *Preamble_ideal)
{
  if(Preamble_ideal) {
    free(Preamble_ideal);
  }
}

void AIOT_D2R_PHY_TX_Signal_free(c16_t *txData)
{
  if (txData != NULL)
    free(txData);
}

double calculate_BER(uint8_t *rx_payload, uint8_t *payload, NR_AIOT_UL_FRAME_PARMS *frame_parms)
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
  channel_model->sampling_rate = 1.92; // N_RB2sampling_rate(frame_parms->nr_frame_parms.N_RB_DL); // in MHz
  channel_model->bw = 0.015; // N_RB2channel_bandwidth(frame_parms->nr_frame_parms.N_RB_DL); // in MHz

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

  c16_t *txData = NULL, *txFiltered = NULL;
  c16_t **rxData = NULL;
  int16_t *envelope, *filteredData;
  int *correlation;
  uint8_t *rx_payload;
  rx_size = frame_parms->packet_samples + channel_model->delay + 200;

  int Preamble_offset = 0;

  //generate_butter_coeffs_f64(&filter, channel_model->bw * 1e6, (double)channel_model->sampling_rate * 1e6);
  Butter3_c16 filter;
  butter3_c16_init(&filter, channel_model->bw * 1e6, (double)channel_model->sampling_rate * 1e6);

  Butter3_Q15 filter_q15;
  butter3_init(&filter_q15, 600e3, (double)channel_model->sampling_rate * 1e6);

  /*ButterworthBP3 bpFilter;
  BP_calculate_coefficients((double)channel_model->sampling_rate * 1e6, frame_parms->f_min, frame_parms->f_max, &bpFilter);

  printf("Bandpass Butterworth filter coefficients:\n");
  for(int i=0; i<3; i++) {
      printf(" Section %d: b0=%d, b1=%d, b2=%d, a1=%d, a2=%d\n", i,
             bpFilter.sections[i].b0,
             bpFilter.sections[i].b1,
             bpFilter.sections[i].b2,
             bpFilter.sections[i].a1,
             bpFilter.sections[i].a2);
  }*/

  txData = malloc(frame_parms->packet_samples * sizeof(c16_t));
  memset(txData, 0, frame_parms->packet_samples * sizeof(c16_t));

  txFiltered = malloc(frame_parms->packet_samples * sizeof(c16_t));
  memset(txFiltered, 0, frame_parms->packet_samples * sizeof(c16_t));

  rxData = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(c16_t *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    rxData[i] = calloc(1, rx_size * sizeof(c16_t));
    memset(rxData[i], 0, rx_size * sizeof(c16_t));
  }

  envelope = malloc(rx_size * sizeof(int16_t));
  filteredData = malloc(rx_size * sizeof(int16_t));
  correlation = malloc(rx_size * sizeof(int));

  rx_payload = malloc(MAX_AIOT_D2R_PACKET_SIZE);
  memset(rx_payload, 0, MAX_AIOT_D2R_PACKET_SIZE);

  double ber_results[snr_steps];
  memset(ber_results, 0, snr_steps * sizeof(double));

  // Initialization of transmit IQ signals
  s_re = malloc(frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));
  s_im = malloc(frame_parms->nr_frame_parms.nb_antennas_tx * sizeof(double *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_tx; i++) {
    s_re[i] = calloc(1, rx_size * sizeof(double));
    s_im[i] = calloc(1, rx_size * sizeof(double));
  }

  bzero(s_re[0], rx_size * sizeof(double));
  bzero(s_im[0], rx_size * sizeof(double));

  // Initialization of receive IQ signals
  r_re = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));
  r_im = malloc(frame_parms->nr_frame_parms.nb_antennas_rx * sizeof(double *));

  for (int i = 0; i < frame_parms->nr_frame_parms.nb_antennas_rx; i++) {
    r_re[i] = calloc(1, rx_size * sizeof(double));
    r_im[i] = calloc(1, rx_size * sizeof(double));
  }

  bzero(r_re[0], rx_size * sizeof(double));
  bzero(r_im[0], rx_size * sizeof(double));

  Preamble_ideal = generate_preamble_ideal_sequence(frame_parms);

  if(testing_mode) {
    sprintf(filename, "%s/D2R_Preamble_Ideal.m", foldername);
    LOG_M(filename, "Preamble_Ideal_sig", Preamble_ideal, frame_parms->preamble_samples, 1, 2);
  }

  if(testing_timing) {
    start_meas(&time_stats);
  }

  double time_tx_REs = 0.0;
  double time_tx_signal = 0.0;
  double time_channel = 0.0;
  double time_envelope = 0.0;
  double time_filter = 0.0;
  double time_sync = 0.0;
  double time_downsample = 0.0;
  double time_rx_packet = 0.0;
  double time_ber = 0.0;

  for(int snr = snr_min; snr <= snr_max; snr += SNR_STEP_DB) {
    channel_model->SNR = snr;

    for(int trials = 0; trials < snr_trials; trials++) {
      for(int i = 0; i < payloadSize / 8; i++) {
        payload[i] = uniformrandom() * 256;
      }

      if(testing_mode) {
        printf("-------------------------------\n");
        printf("Testing SNR %d dB\n", snr);
      }

      // TODO: Block repetition
      // TODO: Channel coding
      AIOT_D2R_PHY_TX_Signal(txData, payload, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_TX_IQ.m", foldername);
        LOG_M(filename, "TX_IQ_sig", txData, frame_parms->packet_samples, 1, 1);
      }

      AIOT_D2R_PHY_TX_Filter(txFiltered, (const c16_t *) txData, frame_parms->packet_samples, &filter);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_TX_Filter.m", foldername);
        LOG_M(filename, "TX_Filter_sig", txFiltered, frame_parms->packet_samples, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        time_tx_REs = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("TX Packet generation time: %f us\n", time_tx_REs);
      }

      SIM_Channel_propagate(rxData, (const c16_t *) txData, channel_params, channel_model->SNR, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_RX_IQ.m", foldername);
        LOG_M(filename, "RX_IQ_sig", rxData[0], rx_size, 1, 1);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        time_channel = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Channel propagation time: %f us\n", time_channel);
      }

      // Envelope detector
      AIOT_D2R_PHY_RX_Envelope_Detector(envelope, (const c16_t **) rxData, rx_size);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_Envelope.m", foldername);
        LOG_M(filename, "Envelope_sig", envelope, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        time_envelope = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Envelope detection time: %f us\n", time_envelope);
      }

      // Low-pass filter
      AIOT_D2R_PHY_RX_Filter(filteredData, (const int16_t *) envelope, rx_size, &filter_q15);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_Filter.m", foldername);
        LOG_M(filename, "Filter_sig", filteredData, rx_size, 1, 0);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        time_filter = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Filtering time: %f us\n", time_filter);
      }

      Preamble_offset = AIOT_D2R_PHY_RX_Synchronize(correlation, (const int16_t *) filteredData, Preamble_ideal, frame_parms);
      if(testing_mode && snr == snr_plot && trials == 0) {
        sprintf(filename, "%s/D2R_Correlation.m", foldername);
        LOG_M(filename, "Correlation_sig", correlation, rx_size - frame_parms->preamble_samples, 1, 2);
      }

      if(testing_timing) {
        stop_meas(&time_stats);
        time_sync = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Synchronization time: %f us\n", time_sync);
      }

      AIOT_D2R_PHY_RX_GetPacket(rx_payload, (const int16_t *) filteredData, Preamble_offset, frame_parms);

      if(testing_timing) {
        stop_meas(&time_stats);
        time_rx_packet = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("Packet extraction time: %f us\n", time_rx_packet);
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
        time_ber = time_stats.diff / (cpu_freq_GHz * 1e9) * 1e6;
        reset_meas(&time_stats);
        start_meas(&time_stats);
        printf("BER calculation time: %f us\n", time_ber);
        printf("Total time per packet: %f us\n", time_tx_REs + time_tx_signal + time_channel + time_envelope + time_filter + time_downsample + time_rx_packet + time_ber);
      }
    }

    ber_results[snr - snr_min] /= snr_trials;
    printf("Completed SNR %d dB: BLER = %f\n", snr, ber_results[snr - snr_min]);
  }

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
  
  AIOT_D2R_PHY_TX_Signal_free(txData);
  SIM_Channel_propagate_free(rxData, frame_parms->nr_frame_parms.nb_antennas_rx);

  free_Preamble_ideal_sequence(Preamble_ideal);

  free_channel_desc_scm(channel_params);

  free(rx_payload);
  free(envelope);
  free(filteredData);
  free(correlation);

  printf("-------------------------------\n");
  printf("            Results\n");
  printf("-------------------------------\n");
  for(int snr = MIN_SNR_DB; snr <= MAX_SNR_DB; snr += SNR_STEP_DB) {
    printf("SNR %d dB: BER = %f\n", snr, ber_results[snr - MIN_SNR_DB]);
  }

  if(!testing_mode) {
    sprintf(filename, "%s/BLER_SIZE%d_RSFS%d.m", foldername, payloadSize, frame_parms->N_SFS);
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
  frame_parms->I_add = false; // additional midamble
  frame_parms->R_code = false; // no FEC

  // Fill in channel model default parameters
  channel_model_t channel_model = {
    .channel_model = AWGN,
    .fc = 897500000, // Carrier frequency n8 band, #50 RB
    .DS_TDL = .03,
    .SNR = 20.0,
    .path_loss_dB = -15.0,
    .noise_power_dB = -120.0,
    .delay = 1290,
    .tx_pwr_dBm = 46.0
  };

  int c;
  while ((c = getopt(argc, argv, "--:O:h:L:P:p:S:N:D:t:T:R:r:s:i:l:a:b:")) != -1) {
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
        printf("-P Payload in hex string (e.g. 1234ABCD)\n");
        printf("-p Payload size in bits (max %d)\n", MAX_AIOT_D2R_PAYLOAD_SIZE * 8);
        printf("-S SNR in dB\n");
        printf("-N Path loss in dB\n");
        printf("-D Delay in samples\n");
        printf("-t Testing mode, parameter is SNR to plot\n");
        printf("-T Testing timing mode\n");

        printf("-R Channel coding: 0=No FEC, 1=FEC\n");
        printf("-r Block repetition: 1=1, 2=2\n");
        printf("-s Small frequency shift factor (0 to 7)\n");
        printf("-i Midamble interval: 0=16, 1=32, 2=64, 3=128\n");
        printf("-l Preamble length: 0=short(7 bits), 1=long(31 bits)\n");
        printf("-a Additional midamble: 0=no, 1=yes\n");
        printf("-b Bit duration option T_bit: 0(T_bit=2*tau), 1(4*tau), 2(8*tau), 3(16*tau)\n");
        exit(-1);
        break;
      case 'L':
        loglvl = atoi(optarg);
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

      case 'R':
        if (atoi(optarg) != 0 && atoi(optarg) != 1)
        {
          printf("Error: R_code must be 0 for No FEC or 1 for FEC\n");
          exit(-1);
        }
        
        frame_parms->R_code = (atoi(optarg) == 1) ? true : false;
        printf("Using channel coding: %s\n", (frame_parms->R_code) ? "FEC" : "No FEC");
        break;
      
      case 'r':
        if(atoi(optarg) != 1 && atoi(optarg) != 2)
        {
          printf("Error: R_block must be 1 or 2\n");
          exit(-1);
        }
        frame_parms->R_block = (atoi(optarg) == 2) ? true : false;
        printf("Using block repetition: %s\n", (frame_parms->R_block) ? "2" : "1");
        break;

      case 's':
        if(atoi(optarg) < 0 || atoi(optarg) > 7) {
          printf("Error: R_SFS must be between 0 and 7\n");
          exit(-1);
        }
        frame_parms->R_SFS = atoi(optarg);
        printf("Using small frequency shift factor: %d\n", frame_parms->R_SFS);
        break;

      case 'i':
        if(atoi(optarg) < 0 || atoi(optarg) > 3) {
          printf("Error: I_bit must be between 0 and 3\n");
          exit(-1);
        }
        frame_parms->I_bit = atoi(optarg);
        printf("Using midamble interval: %d bits\n", I_BIT_TABLE[frame_parms->I_bit]);
        break;

      case 'l':
        if(atoi(optarg) != 0 && atoi(optarg) != 1) {
          printf("Error: L_preamble must be 0 for short or 1 for long\n");
          exit(-1);
        }
        frame_parms->L_preamble = (atoi(optarg) == 1) ? true : false;
        printf("Using preamble length: %s\n", (frame_parms->L_preamble) ? "long (31 bits)" : "short (7 bits)");
        break;

      case 'a':
        if(atoi(optarg) != 0 && atoi(optarg) != 1) {
          printf("Error: I_add must be 0 for no or 1 for yes\n");
          exit(-1);
        }
        frame_parms->I_add = (atoi(optarg) == 1) ? true : false;
        printf("Using additional midamble: %s\n", (frame_parms->I_add) ? "yes" : "no");
        break;

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

  // ---------------------------------------------------------------

  BER_test(payload, payloadSize, frame_parms, &channel_model);

  end_configmodule(uniqCfg);
  logTerm();
  loader_reset();

  return 0;
}
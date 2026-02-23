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

#include <math.h>
#include "PHY/TOOLS/tools_defs.h"
#include "sim.h"
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>

//#define DEBUG_CH
//#define DOPPLER_DEBUG

uint8_t multipath_channel_nosigconv(channel_desc_t *desc)
{
  random_channel(desc, 0);
  return (1);
}

void interleave_channel_output(float **rx_sig_re, float **rx_sig_im, float **output_interleaved, int nb_rx, int num_samples)
{
  for (int i = 0; i < nb_rx; i++) {
    for (int j = 0; j < num_samples; j++) {
      output_interleaved[i][2 * j] = rx_sig_re[i][j];
      output_interleaved[i][2 * j + 1] = rx_sig_im[i][j];
    }
  }
}

//#define CHANNEL_SSE

#ifdef CHANNEL_SSE
void __attribute__((no_sanitize_address)) multipath_channel(channel_desc_t *desc,
                                                            double **tx_sig_re,
                                                            double **tx_sig_im,
                                                            double **rx_sig_re,
                                                            double **rx_sig_im,
                                                            uint32_t length,
                                                            uint8_t keep_channel,
                                                            int log_channel)
{
  int i, ii, j, l;
  // int simd_length;
  // int simd_length, tail_start;
  simde__m128d rx_tmp128_re, rx_tmp128_im, tx128_re, tx128_im, ch128_r, ch128_i, pathloss128;

  double path_loss = pow(10, desc->path_loss_dB / 20);
  uint64_t dd = desc->channel_offset;

  pathloss128 = simde_mm_set1_pd(path_loss);

  if (keep_channel == 0) {
    random_channel(desc, 0);
  }

  // simd_length = (length - dd) / 2;
  // tail_start = simd_length * 2;

  for (i = 0; i < (int)(length - dd); i += 2) {
    for (ii = 0; ii < desc->nb_rx; ii++) {
      rx_tmp128_re = simde_mm_setzero_pd();
      rx_tmp128_im = simde_mm_setzero_pd();

      for (i = 0; i < (int)(length - dd); i += 2) {
        for (ii = 0; ii < desc->nb_rx; ii++) {
          rx_tmp128_re = simde_mm_setzero_pd();
          rx_tmp128_im = simde_mm_setzero_pd();
          for (j = 0; j < desc->nb_tx; j++) {
            for (l = 0; l < (int)desc->channel_length; l++) {
              double tx_re[2] = {0.0, 0.0}, tx_im[2] = {0.0, 0.0};
              if ((i - l) >= 0)
                tx_re[0] = tx_sig_re[j][i - l];
              if ((i + 1 - l) >= 0)
                tx_re[1] = tx_sig_re[j][i + 1 - l];
              if ((i - l) >= 0)
                tx_im[0] = tx_sig_im[j][i - l];
              if ((i + 1 - l) >= 0)
                tx_im[1] = tx_sig_im[j][i + 1 - l];
              tx128_re = simde_mm_loadu_pd(tx_re);
              tx128_im = simde_mm_loadu_pd(tx_im);
              ch128_r = simde_mm_set1_pd(desc->ch[ii + (j * desc->nb_rx)][l].r);
              ch128_i = simde_mm_set1_pd(desc->ch[ii + (j * desc->nb_rx)][l].i);
              rx_tmp128_re =
                  simde_mm_add_pd(rx_tmp128_re,
                                  simde_mm_sub_pd(simde_mm_mul_pd(tx128_re, ch128_r), simde_mm_mul_pd(tx128_im, ch128_i)));
              rx_tmp128_im =
                  simde_mm_add_pd(rx_tmp128_im,
                                  simde_mm_add_pd(simde_mm_mul_pd(tx128_re, ch128_i), simde_mm_mul_pd(tx128_im, ch128_r)));
            }
          }
          simde_mm_storeu_pd(&rx_sig_re[ii][i + dd], simde_mm_mul_pd(rx_tmp128_re, pathloss128));
          simde_mm_storeu_pd(&rx_sig_im[ii][i + dd], simde_mm_mul_pd(rx_tmp128_im, pathloss128));
        }
      }
    } // ii
  } // i

  // Handle the final sample if the length is odd
  if ((length - dd) % 2) {
    int i_tail = length - dd - 1;
    for (ii = 0; ii < desc->nb_rx; ii++) {
      struct complexd rx_tmp = {0};
      for (j = 0; j < desc->nb_tx; j++) {
        struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];
        for (l = 0; l < (int)desc->channel_length; l++) {
          if ((i_tail - l) >= 0) {
            struct complexd tx;
            tx.r = tx_sig_re[j][i_tail - l];
            tx.i = tx_sig_im[j][i_tail - l];
            rx_tmp.r += (tx.r * chan[l].r) - (tx.i * chan[l].i);
            rx_tmp.i += (tx.i * chan[l].r) + (tx.r * chan[l].i);
          }
        }
      }
      rx_sig_re[ii][i_tail + dd] = rx_tmp.r * path_loss;
      rx_sig_im[ii][i_tail + dd] = rx_tmp.i * path_loss;
    }
  }
}

#else

void __attribute__((no_sanitize_address)) multipath_channel(channel_desc_t *desc,
                                                            double **tx_sig_re,
                                                            double **tx_sig_im,
                                                            double **rx_sig_re,
                                                            double **rx_sig_im,
                                                            uint32_t length,
                                                            uint8_t keep_channel,
                                                            int log_channel)
{
  double path_loss = pow(10, desc->path_loss_dB / 20);
  uint64_t dd = desc->channel_offset;

#ifdef DEBUG_CH
  printf("[CHANNEL] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %lu, len %d \n",
         keep_channel,
         path_loss,
         desc->path_loss_dB,
         desc->nb_rx,
         desc->nb_tx,
         dd,
         desc->channel_length);
#endif

  if (keep_channel) {
    // do nothing - keep channel
  } else {
    random_channel(desc, 0);
  }

#ifdef DEBUG_CH
  for (int l = 0; l < (int)desc->channel_length; l++) {
    printf("ch[%i] = (%f, %f)\n", l, desc->ch[0][l].r, desc->ch[0][l].i);
  }
#endif

  struct complexd cexp_doppler[length];
  if (desc->max_Doppler != 0.0) {
    get_cexp_doppler(cexp_doppler, desc, length);
  }

  for (int i = 0; i < ((int)length - dd); i++) {
    for (int ii = 0; ii < desc->nb_rx; ii++) {
      struct complexd rx_tmp = {0};
      for (int j = 0; j < desc->nb_tx; j++) {
        struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];
        for (int l = 0; l < (int)desc->channel_length; l++) {
          if ((i >= 0) && (i - l) >= 0) {
            struct complexd tx;
            tx.r = tx_sig_re[j][i - l];
            tx.i = tx_sig_im[j][i - l];
            rx_tmp.r += (tx.r * chan[l].r) - (tx.i * chan[l].i);
            rx_tmp.i += (tx.i * chan[l].r) + (tx.r * chan[l].i);
          }
#if 0
          if (i==0 && log_channel == 1) {
            printf("channel[%d][%d][%d] = %f dB \t(%e, %e)\n",
                   ii, j, l, 10 * log10(pow(chan[l].r, 2.0) + pow(chan[l].i, 2.0)), chan[l].r, chan[l].i);
	        }
#endif
        } // l
      } // j
#if 0
      if (desc->max_Doppler != 0.0)
        rx_tmp = cdMul(rx_tmp, cexp_doppler[i]);
#endif

#ifdef DOPPLER_DEBUG
      printf("[k %2i] cexp_doppler = (%7.4f, %7.4f), abs(cexp_doppler) = %.4f\n",
             i,
             cexp_doppler[i].r,
             cexp_doppler[i].i,
             sqrt(cexp_doppler[i].r * cexp_doppler[i].r + cexp_doppler[i].i * cexp_doppler[i].i));
#endif

      rx_sig_re[ii][i + dd] = rx_tmp.r * path_loss;
      rx_sig_im[ii][i + dd] = rx_tmp.i * path_loss;
#ifdef DEBUG_CH
      if ((i % 32) == 0) {
        printf("rx aa %d: %f, %f  =>  %e, %e\n", ii, rx_tmp.r, rx_tmp.i, rx_sig_re[ii][i - dd], rx_sig_im[ii][i - dd]);
      }
#endif
      // rx_sig_re[ii][i] = sqrt(.5)*(tx_sig_re[0][i] + tx_sig_re[1][i]);
      // rx_sig_im[ii][i] = sqrt(.5)*(tx_sig_im[0][i] + tx_sig_im[1][i]);

    } // ii
  } // i
}
#endif

#ifdef CHANNEL_SSE
void __attribute__((no_sanitize_address)) multipath_channel_float(channel_desc_t *desc,
                                                                  float **tx_sig_interleaved,
                                                                  float **rx_sig_re,
                                                                  float **rx_sig_im,
                                                                  uint32_t length,
                                                                  uint8_t keep_channel,
                                                                  int log_channel)
{
  int i, ii, j, l;
  simde__m128 rx_tmp128_re, rx_tmp128_im, tx128_re, tx128_im, ch128_r, ch128_i, pathloss128;
  uint64_t dd = desc->channel_offset;

  if (keep_channel == 0) {
    random_channel(desc, 0);
  }

  struct complexd cexp_doppler[length];
  if (desc->max_Doppler != 0.0) {
    get_cexp_doppler(cexp_doppler, desc, length);
  }

  if (dd >= length) {
    return; // No samples to process
  }

  for (i = 0; i <= (int)(length - dd) - 4; i += 4) {
    for (ii = 0; ii < desc->nb_rx; ii++) {
      rx_tmp128_re = simde_mm_setzero_ps();
      rx_tmp128_im = simde_mm_setzero_ps();

      for (j = 0; j < desc->nb_tx; j++) {
        for (l = 0; l < (int)desc->channel_length; l++) {
          int idx = i - l;
          if (idx >= 0) {
            // Load 2 complex numbers (4 floats) => {I0, Q0, I1, Q1}
            simde__m128 vec0 = simde_mm_loadu_ps(&tx_sig_interleaved[j][2 * idx]);
            simde__m128 vec1 = simde_mm_loadu_ps(&tx_sig_interleaved[j][2 * (idx + 2)]);
            tx128_re = simde_mm_shuffle_ps(vec0, vec1, SIMDE_MM_SHUFFLE(2, 0, 2, 0)); // {I0,I1,I2,I3}
            tx128_im = simde_mm_shuffle_ps(vec0, vec1, SIMDE_MM_SHUFFLE(3, 1, 3, 1)); // {Q0,Q1,Q2,Q3}
          } else {
            tx128_re = simde_mm_setzero_ps();
            tx128_im = simde_mm_setzero_ps();
          }

          ch128_r = simde_mm_set1_ps((float)desc->ch[ii + (j * desc->nb_rx)][l].r);
          ch128_i = simde_mm_set1_ps((float)desc->ch[ii + (j * desc->nb_rx)][l].i);

          // re = (tx_re * ch_r) - (tx_im * ch_i)
          rx_tmp128_re = simde_mm_add_ps(rx_tmp128_re,
                                         simde_mm_sub_ps(simde_mm_mul_ps(tx128_re, ch128_r), simde_mm_mul_ps(tx128_im, ch128_i)));
          // im = (tx_re * ch_i) + (tx_im * ch_r)
          rx_tmp128_im = simde_mm_add_ps(rx_tmp128_im,
                                         simde_mm_add_ps(simde_mm_mul_ps(tx128_re, ch128_i), simde_mm_mul_ps(tx128_im, ch128_r)));
        }
      }

#if 0
            if (desc->max_Doppler != 0.0) {
                float doppler_re[4] = {(float)cexp_doppler[i].r, (float)cexp_doppler[i+1].r, (float)cexp_doppler[i+2].r, (float)cexp_doppler[i+3].r};
                float doppler_im[4] = {(float)cexp_doppler[i].i, (float)cexp_doppler[i+1].i, (float)cexp_doppler[i+2].i, (float)cexp_doppler[i+3].i};
                
                simde__m128 doppler128_r = simde_mm_loadu_ps(doppler_re);
                simde__m128 doppler128_i = simde_mm_loadu_ps(doppler_im);

                simde__m128 temp_re = rx_tmp128_re;
                simde__m128 temp_im = rx_tmp128_im;

                rx_tmp128_re = simde_mm_sub_ps(simde_mm_mul_ps(temp_re, doppler128_r), simde_mm_mul_ps(temp_im, doppler128_i));
                rx_tmp128_im = simde_mm_add_ps(simde_mm_mul_ps(temp_im, doppler128_r), simde_mm_mul_ps(temp_re, doppler128_i));
            }
#endif
      simde_mm_storeu_ps(&rx_sig_re[ii][i + dd], rx_tmp128_re);
      simde_mm_storeu_ps(&rx_sig_im[ii][i + dd], rx_tmp128_im);
    }
  }

  // Handle remaining samples (tail loop) that are not a multiple of 4
  for (; i < (int)(length - dd); i++) {
    for (ii = 0; ii < desc->nb_rx; ii++) {
      struct complexf rx_tmp = {0.0f, 0.0f};
      for (j = 0; j < desc->nb_tx; j++) {
        struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];
        for (l = 0; l < (int)desc->channel_length; l++) {
          if ((i - l) >= 0) {
            struct complexf tx;
            tx.r = tx_sig_interleaved[j][2 * (i - l)];
            tx.i = tx_sig_interleaved[j][2 * (i - l) + 1];
            rx_tmp.r += (tx.r * (float)chan[l].r) - (tx.i * (float)chan[l].i);
            rx_tmp.i += (tx.i * (float)chan[l].r) + (tx.r * (float)chan[l].i);
          }
        }
      }
#if 0
            if (desc->max_Doppler != 0.0) {
                struct complexf doppler_factor = {(float)cexp_doppler[i].r, (float)cexp_doppler[i].i};
                struct complexf temp = rx_tmp;
                rx_tmp.r = (temp.r * doppler_factor.r) - (temp.i * doppler_factor.i);
                rx_tmp.i = (temp.i * doppler_factor.r) + (temp.r * doppler_factor.i);
            }
#endif
      rx_sig_re[ii][i + dd] = rx_tmp.r;
      rx_sig_im[ii][i + dd] = rx_tmp.i;
    }
  }
}

#else
void multipath_channel_float(channel_desc_t *desc,
                             float **tx_sig_interleaved,
                             float **rx_sig_re,
                             float **rx_sig_im,
                             uint32_t length,
                             uint8_t keep_channel,
                             int log_channel)
{
  uint64_t dd = desc->channel_offset;

  if (keep_channel == 0) {
    random_channel(desc, 0);
  }

  struct complexd cexp_doppler[length];
  if (desc->max_Doppler != 0.0) {
    get_cexp_doppler(cexp_doppler, desc, length);
  }

  if (dd >= length) {
    return;
  }

  for (int i = 0; i < ((int)length - dd); i++) {
    for (int ii = 0; ii < desc->nb_rx; ii++) {
      struct complexf rx_tmp = {0.0f, 0.0f};

      for (int j = 0; j < desc->nb_tx; j++) {
        struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];

        for (int l = 0; l < (int)desc->channel_length; l++) {
          if ((i - l) >= 0) {
            struct complexf tx;
            tx.r = tx_sig_interleaved[j][2 * (i - l)];
            tx.i = tx_sig_interleaved[j][2 * (i - l) + 1];
            rx_tmp.r += (tx.r * (float)chan[l].r) - (tx.i * (float)chan[l].i);
            rx_tmp.i += (tx.i * (float)chan[l].r) + (tx.r * (float)chan[l].i);
          }
        } // l (channel_length)
      } // j (nb_tx)

#if 0
            if (desc->max_Doppler != 0.0) {
                // Perform complex multiplication: rx_tmp = rx_tmp * cexp_doppler[i]
                struct complexf doppler_factor = {(float)cexp_doppler[i].r, (float)cexp_doppler[i].i};
                struct complexf temp = rx_tmp;
                rx_tmp.r = (temp.r * doppler_factor.r) - (temp.i * doppler_factor.i);
                rx_tmp.i = (temp.i * doppler_factor.r) + (temp.r * doppler_factor.i);
            }
#endif
      rx_sig_re[ii][i + dd] = rx_tmp.r;
      rx_sig_im[ii][i + dd] = rx_tmp.i;

    } // ii (nb_rx)
  } // i (length)
}

#endif

void channel_convolution(channel_desc_t *desc, c16_t **input, c16_t **output, uint32_t length)
{
  float pathloss = powf(10.0f, (float)desc->path_loss_dB / 20.0f);
  uint L = desc->channel_length;
  uint64_t offset = desc->channel_offset;

  // Reverse channel and bake in pathloss
  cf_t rev_channel[desc->nb_rx][desc->nb_tx][L];
  for (int rx = 0; rx < desc->nb_rx; rx++) {
    for (int tx = 0; tx < desc->nb_tx; tx++) {
      struct complexd *chan = desc->ch[rx + (tx * desc->nb_rx)];
      for (uint i = 0U; i < L; i++) {
        rev_channel[rx][tx][i].r = (float)chan[L - 1 - i].r * pathloss;
        rev_channel[rx][tx][i].i = (float)chan[L - 1 - i].i * pathloss;
      }
    }
  }

  for (int rx = 0; rx < desc->nb_rx; rx++) {
    for (uint i = 0U; i < length - offset; i++) {
      cf_t rx_sum = {0, 0};

      for (int tx = 0; tx < desc->nb_tx; tx++) {
        // We look at tx_sig[i - offset] and go 'backwards' for L_eff samples.
        // To make it a forward-crawl for speed, we start at the oldest
        // sample in the window and move forward.

        int oldest_idx = i - (L - 1);
        int current_L = L;
        int start_c = 0;

        // Handle the "Past" boundary (nothing before index 0 exists)
        if (oldest_idx < 0) {
          start_c = -oldest_idx; // Skip the taps that point to negative time
          current_L -= start_c;
          oldest_idx = 0;
        }

        if (current_L > 0) {
          c16_t *tx_ptr = &input[tx][oldest_idx];
          cf_t *c_ptr = &rev_channel[rx][tx][start_c];

          // Steady State inner loop: no branches, purely linear
          for (int l = 0; l < current_L; l++) {
            rx_sum.r += (tx_ptr->r * c_ptr->r) - (tx_ptr->i * c_ptr->i);
            rx_sum.i += (tx_ptr->i * c_ptr->r) + (tx_ptr->r * c_ptr->i);
            tx_ptr++;
            c_ptr++;
          }
        }
      }
      output[rx][i + offset].r = rx_sum.r;
      output[rx][i + offset].i = rx_sum.i;
    }
  }
}

void channel_convolution_avx2(channel_desc_t *desc, c16_t **input, c16_t **output, uint32_t length)
{
  float pathloss = powf(10.0f, (float)desc->path_loss_dB / 20.0f);
  uint L = desc->channel_length;
  uint64_t offset = desc->channel_offset;

  cf_t rev_channel[desc->nb_rx][desc->nb_tx][L];

  for (int rx = 0; rx < desc->nb_rx; rx++) {
    for (int tx = 0; tx < desc->nb_tx; tx++) {
      struct complexd *chan = desc->ch[rx + (tx * desc->nb_rx)];
      for (uint i = 0U; i < L; i++) {
        rev_channel[rx][tx][i].r = (float)chan[L - 1 - i].r * pathloss;
        rev_channel[rx][tx][i].i = (float)chan[L - 1 - i].i * pathloss;
      }
    }
  }

  int end_idx = length - offset;

  for (int rx = 0; rx < desc->nb_rx; rx++) {
    int i = 0;

    // 1. Transient part (scalar)
    for (; i < (int)L - 1 && i < end_idx; i++) {
      cf_t rx_sum = {0, 0};
      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int oldest_idx = i - (L - 1);
        int current_L = L;
        int start_c = 0;
        if (oldest_idx < 0) {
          start_c = -oldest_idx;
          current_L -= start_c;
          oldest_idx = 0;
        }
        if (current_L > 0) {
          c16_t *tx_ptr = &input[tx][oldest_idx];
          cf_t *c_ptr = &rev_channel[rx][tx][start_c];
          for (int l = 0; l < current_L; l++) {
            rx_sum.r += (tx_ptr->r * c_ptr->r) - (tx_ptr->i * c_ptr->i);
            rx_sum.i += (tx_ptr->i * c_ptr->r) + (tx_ptr->r * c_ptr->i);
            tx_ptr++;
            c_ptr++;
          }
        }
      }
      output[rx][i + offset].r = (int16_t)rx_sum.r;
      output[rx][i + offset].i = (int16_t)rx_sum.i;
    }

    // 2. Steady state (SIMD)
    for (; i <= end_idx - 4; i += 4) {
      simde__m256 rx_sum_vec = simde_mm256_setzero_ps();

      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int start_idx = i - (L - 1);
        c16_t *tx_ptr = &input[tx][start_idx];
        cf_t *c_ptr = &rev_channel[rx][tx][0];

        for (int l = 0; l < (int)L; l++) {
          simde__m128i in_128 = simde_mm_loadu_si128((simde__m128i const *)(tx_ptr + l)); // 4 complex samples
          simde__m256i in_32 = simde_mm256_cvtepi16_epi32(in_128);
          simde__m256 in_ps = simde_mm256_cvtepi32_ps(in_32);

          float tr = c_ptr[l].r;
          float ti = c_ptr[l].i;
          simde__m256 tap = simde_mm256_set_ps(ti, tr, ti, tr, ti, tr, ti, tr);

          simde__m256 in_re = simde_mm256_shuffle_ps(in_ps, in_ps, SIMDE_MM_SHUFFLE(2, 2, 0, 0));
          simde__m256 in_im = simde_mm256_shuffle_ps(in_ps, in_ps, SIMDE_MM_SHUFFLE(3, 3, 1, 1));

          simde__m256 term1 = simde_mm256_mul_ps(in_re, tap);
          simde__m256 tap_swap = simde_mm256_set_ps(tr, -ti, tr, -ti, tr, -ti, tr, -ti);
          simde__m256 term2 = simde_mm256_mul_ps(in_im, tap_swap);

          rx_sum_vec = simde_mm256_add_ps(rx_sum_vec, simde_mm256_add_ps(term1, term2));
        }
      }
      simde__m256i out_32 = simde_mm256_cvtps_epi32(rx_sum_vec);
      simde__m128i out_16_low = simde_mm256_castsi256_si128(out_32);
      simde__m128i out_16_high = simde_mm256_extracti128_si256(out_32, 1);
      simde__m128i out_16 = simde_mm_packs_epi32(out_16_low, out_16_high);
      simde_mm_storeu_si128((simde__m128i *)&output[rx][i + offset], out_16);
    }

    // 3. Remainder (scalar)
    for (; i < end_idx; i++) {
      cf_t rx_sum = {0, 0};
      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int oldest_idx = i - (L - 1);
        c16_t *tx_ptr = &input[tx][oldest_idx];
        cf_t *c_ptr = &rev_channel[rx][tx][0];
        for (int l = 0; l < (int)L; l++) {
          rx_sum.r += (tx_ptr->r * c_ptr->r) - (tx_ptr->i * c_ptr->i);
          rx_sum.i += (tx_ptr->i * c_ptr->r) + (tx_ptr->r * c_ptr->i);
          tx_ptr++;
          c_ptr++;
        }
      }
      output[rx][i + offset].r = (int16_t)rx_sum.r;
      output[rx][i + offset].i = (int16_t)rx_sum.i;
    }
  }
}

void channel_convolution_avx512(channel_desc_t *desc, c16_t **input, c16_t **output, uint32_t length)
{
  float pathloss = powf(10.0f, (float)desc->path_loss_dB / 20.0f);
  uint L = desc->channel_length;
  uint64_t offset = desc->channel_offset;

  cf_t rev_channel[desc->nb_rx][desc->nb_tx][L];

  for (int rx = 0; rx < desc->nb_rx; rx++) {
    for (int tx = 0; tx < desc->nb_tx; tx++) {
      struct complexd *chan = desc->ch[rx + (tx * desc->nb_rx)];
      for (uint i = 0U; i < L; i++) {
        rev_channel[rx][tx][i].r = (float)chan[L - 1 - i].r * pathloss;
        rev_channel[rx][tx][i].i = (float)chan[L - 1 - i].i * pathloss;
      }
    }
  }

  int end_idx = length - offset;

  for (int rx = 0; rx < desc->nb_rx; rx++) {
    int i = 0;

    // 1. Transient part (scalar)
    for (; i < (int)L - 1 && i < end_idx; i++) {
      cf_t rx_sum = {0, 0};
      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int oldest_idx = i - (L - 1);
        int current_L = L;
        int start_c = 0;
        if (oldest_idx < 0) {
          start_c = -oldest_idx;
          current_L -= start_c;
          oldest_idx = 0;
        }
        if (current_L > 0) {
          c16_t *tx_ptr = &input[tx][oldest_idx];
          cf_t *c_ptr = &rev_channel[rx][tx][start_c];
          for (int l = 0; l < current_L; l++) {
            rx_sum.r += (tx_ptr->r * c_ptr->r) - (tx_ptr->i * c_ptr->i);
            rx_sum.i += (tx_ptr->i * c_ptr->r) + (tx_ptr->r * c_ptr->i);
            tx_ptr++;
            c_ptr++;
          }
        }
      }
      output[rx][i + offset].r = (int16_t)rx_sum.r;
      output[rx][i + offset].i = (int16_t)rx_sum.i;
    }

    // 2. Steady state (AVX512)
    for (; i <= end_idx - 8; i += 8) {
      simde__m512 rx_sum_vec = simde_mm512_setzero_ps();

      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int start_idx = i - (L - 1);
        c16_t *tx_ptr = &input[tx][start_idx];
        cf_t *c_ptr = &rev_channel[rx][tx][0];

        for (int l = 0; l < (int)L; l++) {
          simde__m256i in_256i = simde_mm256_loadu_si256((simde__m256i const *)(tx_ptr + l));
          simde__m512i in_512i = simde_mm512_cvtepi16_epi32(in_256i);
          simde__m512 in_ps = simde_mm512_cvtepi32_ps(in_512i);

          float tr = c_ptr[l].r;
          float ti = c_ptr[l].i;
          simde__m512 v_tr = simde_mm512_set1_ps(tr);
          simde__m512 v_ti = simde_mm512_set1_ps(ti);
          simde__m512 v_nti = simde_mm512_set1_ps(-ti);

          simde__m512 tap = simde_mm512_mask_blend_ps(0xAAAA, v_tr, v_ti);
          simde__m512 tap_swap = simde_mm512_mask_blend_ps(0xAAAA, v_nti, v_tr);

          simde__m512 in_re = simde_mm512_shuffle_ps(in_ps, in_ps, 0xA0);
          simde__m512 in_im = simde_mm512_shuffle_ps(in_ps, in_ps, 0xF5);

          rx_sum_vec = simde_mm512_fmadd_ps(in_re, tap, rx_sum_vec);
          rx_sum_vec = simde_mm512_fmadd_ps(in_im, tap_swap, rx_sum_vec);
        }
      }
      simde__m512i out_32 = simde_mm512_cvtps_epi32(rx_sum_vec);
      simde__m256i out_16 = simde_mm512_cvtsepi32_epi16(out_32);
      simde_mm256_storeu_si256((simde__m256i *)&output[rx][i + offset], out_16);
    }

    // 3. Remainder (scalar)
    for (; i < end_idx; i++) {
      cf_t rx_sum = {0, 0};
      for (int tx = 0; tx < desc->nb_tx; tx++) {
        int oldest_idx = i - (L - 1);
        c16_t *tx_ptr = &input[tx][oldest_idx];
        cf_t *c_ptr = &rev_channel[rx][tx][0];
        for (int l = 0; l < (int)L; l++) {
          rx_sum.r += (tx_ptr->r * c_ptr->r) - (tx_ptr->i * c_ptr->i);
          rx_sum.i += (tx_ptr->i * c_ptr->r) + (tx_ptr->r * c_ptr->i);
          tx_ptr++;
          c_ptr++;
        }
      }
      output[rx][i + offset].r = (int16_t)rx_sum.r;
      output[rx][i + offset].i = (int16_t)rx_sum.i;
    }
  }
}

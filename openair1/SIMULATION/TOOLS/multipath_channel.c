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

//#define DEBUG_CH
//#define DOPPLER_DEBUG

uint8_t multipath_channel_nosigconv(channel_desc_t *desc)
{
  random_channel(desc,0);
  return(1);
}

//#define CHANNEL_SSE

#ifdef CHANNEL_SSE
void __attribute__ ((no_sanitize_address)) multipath_channel(channel_desc_t *desc,
                       double **tx_sig_re,
                       double **tx_sig_im,
                       double **rx_sig_re,
                       double **rx_sig_im,
                       uint32_t length,
                       uint8_t keep_channel,
             		       int log_channel)
{
  int i,ii,j,l;
  // int simd_length;
  // int simd_length, tail_start;
  simde__m128d rx_tmp128_re, rx_tmp128_im, tx128_re, tx128_im, ch128_r, ch128_i, pathloss128;

  double path_loss = pow(10,desc->path_loss_dB/20);
  uint64_t dd = desc->channel_offset;

  pathloss128 = simde_mm_set1_pd(path_loss);

  if (keep_channel == 0) {
    random_channel(desc,0);
  }

  // simd_length = (length - dd) / 2;
  // tail_start = simd_length * 2;

  for (i = 0; i < (int)(length - dd); i+=2) {
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
                  if ((i - l) >= 0) tx_re[0] = tx_sig_re[j][i - l];
                  if ((i + 1 - l) >= 0) tx_re[1] = tx_sig_re[j][i + 1 - l];
                  if ((i - l) >= 0) tx_im[0] = tx_sig_im[j][i - l];
                  if ((i + 1 - l) >= 0) tx_im[1] = tx_sig_im[j][i + 1 - l];
                  tx128_re = simde_mm_loadu_pd(tx_re);
                  tx128_im = simde_mm_loadu_pd(tx_im);
                  ch128_r = simde_mm_set1_pd(desc->ch[ii + (j * desc->nb_rx)][l].r);
                  ch128_i = simde_mm_set1_pd(desc->ch[ii + (j * desc->nb_rx)][l].i);
                  rx_tmp128_re = simde_mm_add_pd(rx_tmp128_re, simde_mm_sub_pd(simde_mm_mul_pd(tx128_re, ch128_r), simde_mm_mul_pd(tx128_im, ch128_i)));
                  rx_tmp128_im = simde_mm_add_pd(rx_tmp128_im, simde_mm_add_pd(simde_mm_mul_pd(tx128_re, ch128_i), simde_mm_mul_pd(tx128_im, ch128_r)));
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

void __attribute__ ((no_sanitize_address)) multipath_channel(channel_desc_t *desc,
                       double **tx_sig_re,
                       double **tx_sig_im,
                       double **rx_sig_re,
                       double **rx_sig_im,
                       uint32_t length,
                       uint8_t keep_channel,
		                   int log_channel)
{

  double path_loss = pow(10,desc->path_loss_dB/20);
  uint64_t dd = desc->channel_offset;

#ifdef DEBUG_CH
  printf("[CHANNEL] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %lu, len %d \n",
         keep_channel, path_loss, desc->path_loss_dB, desc->nb_rx, desc->nb_tx, dd, desc->channel_length);
#endif

  if (keep_channel) {
    // do nothing - keep channel
  } else {
    random_channel(desc,0);
  }

#ifdef DEBUG_CH
  for (l = 0; l<(int)desc->channel_length; l++) {
    printf("ch[%i] = (%f, %f)\n", l, desc->ch[0][l].r, desc->ch[0][l].i);
  }
#endif

  struct complexd cexp_doppler[length];
  if (desc->max_Doppler != 0.0) {
    get_cexp_doppler(cexp_doppler, desc, length);
  }

  for (int i=0; i<((int)length-dd); i++) {
    for (int ii=0; ii<desc->nb_rx; ii++) {
      struct complexd rx_tmp={0};
      for (int j=0; j<desc->nb_tx; j++) {
        struct complexd *chan=desc->ch[ii+(j*desc->nb_rx)];
        for (int l = 0; l<(int)desc->channel_length; l++) {
          if ((i>=0) && (i-l)>=0) {
            struct complexd tx;
            tx.r = tx_sig_re[j][i-l];
            tx.i = tx_sig_im[j][i-l];
            rx_tmp.r += (tx.r * chan[l].r) - (tx.i * chan[l].i);
            rx_tmp.i += (tx.i * chan[l].r) + (tx.r * chan[l].i);
          }
          #if 0
          if (i==0 && log_channel == 1) {
            printf("channel[%d][%d][%d] = %f dB \t(%e, %e)\n",
                   ii, j, l, 10 * log10(pow(chan[l].r, 2.0) + pow(chan[l].i, 2.0)), chan[l].r, chan[l].i);
	        }
          #endif
        } //l
      }  // j
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

      rx_sig_re[ii][i+dd] = rx_tmp.r*path_loss;
      rx_sig_im[ii][i+dd] = rx_tmp.i*path_loss;
#ifdef DEBUG_CH
      if ((i%32)==0) {
	       printf("rx aa %d: %f, %f  =>  %e, %e\n",
                ii,  rx_tmp.r, rx_tmp.i, rx_sig_re[ii][i-dd], rx_sig_im[ii][i-dd]);
      }	
#endif      
      //rx_sig_re[ii][i] = sqrt(.5)*(tx_sig_re[0][i] + tx_sig_re[1][i]);
      //rx_sig_im[ii][i] = sqrt(.5)*(tx_sig_im[0][i] + tx_sig_im[1][i]);

    } // ii
  } // i
}
#endif




#ifdef CHANNEL_SSE
void __attribute__ ((no_sanitize_address)) multipath_channel_float(
                             channel_desc_t *desc,
                             float **tx_sig_re,
                             float **tx_sig_im,
                             float **rx_sig_re,
                             float **rx_sig_im,
                             uint32_t length,
                             uint8_t keep_channel,
                             int log_channel)
{
    int i, ii, j, l;
    // Use __m128 for single-precision (float) operations
    simde__m128 rx_tmp128_re, rx_tmp128_im, tx128_re, tx128_im, ch128_r, ch128_i, pathloss128;

    float path_loss = (float)pow(10, desc->path_loss_dB / 20.0);
    uint64_t dd = desc->channel_offset;

    // Use _ps (packed single) intrinsics
    pathloss128 = simde_mm_set1_ps(path_loss);

    if (keep_channel == 0) {
        random_channel(desc, 0);
    }

    struct complexd cexp_doppler[length];
    if (desc->max_Doppler != 0.0) {
        get_cexp_doppler(cexp_doppler, desc, length);
    }

    // Process 4 floats at a time
    for (i = 0; i <= (int)(length - dd) - 4; i += 4) {
        for (ii = 0; ii < desc->nb_rx; ii++) {
            rx_tmp128_re = simde_mm_setzero_ps();
            rx_tmp128_im = simde_mm_setzero_ps();

            for (j = 0; j < desc->nb_tx; j++) {
                for (l = 0; l < (int)desc->channel_length; l++) {
                    
                    // Safely load 4 floats, zeroing out-of-bounds values
                    float tx_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    float tx_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    if ((i - l) >= 0)     tx_re[0] = tx_sig_re[j][i - l];
                    if ((i + 1 - l) >= 0) tx_re[1] = tx_sig_re[j][i + 1 - l];
                    if ((i + 2 - l) >= 0) tx_re[2] = tx_sig_re[j][i + 2 - l];
                    if ((i + 3 - l) >= 0) tx_re[3] = tx_sig_re[j][i + 3 - l];

                    if ((i - l) >= 0)     tx_im[0] = tx_sig_im[j][i - l];
                    if ((i + 1 - l) >= 0) tx_im[1] = tx_sig_im[j][i + 1 - l];
                    if ((i + 2 - l) >= 0) tx_im[2] = tx_sig_im[j][i + 2 - l];
                    if ((i + 3 - l) >= 0) tx_im[3] = tx_sig_im[j][i + 3 - l];

                    tx128_re = simde_mm_loadu_ps(tx_re);
                    tx128_im = simde_mm_loadu_ps(tx_im);

                    // Broadcast channel coefficients to all 4 lanes
                    ch128_r = simde_mm_set1_ps((float)desc->ch[ii + (j * desc->nb_rx)][l].r);
                    ch128_i = simde_mm_set1_ps((float)desc->ch[ii + (j * desc->nb_rx)][l].i);

                    // Perform complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
                    // re = (tx_re * ch_r) - (tx_im * ch_i)
                    rx_tmp128_re = simde_mm_add_ps(rx_tmp128_re, simde_mm_sub_ps(simde_mm_mul_ps(tx128_re, ch128_r), simde_mm_mul_ps(tx128_im, ch128_i)));
                    // im = (tx_re * ch_i) + (tx_im * ch_r)
                    rx_tmp128_im = simde_mm_add_ps(rx_tmp128_im, simde_mm_add_ps(simde_mm_mul_ps(tx128_re, ch128_i), simde_mm_mul_ps(tx128_im, ch128_r)));
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

            // Store results after applying path loss
            simde_mm_storeu_ps(&rx_sig_re[ii][i + dd], simde_mm_mul_ps(rx_tmp128_re, pathloss128));
            simde_mm_storeu_ps(&rx_sig_im[ii][i + dd], simde_mm_mul_ps(rx_tmp128_im, pathloss128));
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
                        tx.r = tx_sig_re[j][i - l];
                        tx.i = tx_sig_im[j][i - l];
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
            rx_sig_re[ii][i + dd] = rx_tmp.r * path_loss;
            rx_sig_im[ii][i + dd] = rx_tmp.i * path_loss;
        }
    }
}

#else

void multipath_channel_float(channel_desc_t *desc,
                             float **tx_sig_re,
                             float **tx_sig_im,
                             float **rx_sig_re,
                             float **rx_sig_im,
                             uint32_t length,
                             uint8_t keep_channel,
                             int log_channel)
{
    // --- Initialization ---
    float path_loss = (float)pow(10, desc->path_loss_dB / 20.0);
    uint64_t dd = desc->channel_offset;

    // --- Handle keep_channel flag ---
    if (keep_channel) {
        // do nothing - keep the existing channel
    } else {
        random_channel(desc, 0);
    }

    // --- Doppler Effect Preparation ---
    struct complexd cexp_doppler[length];
    if (desc->max_Doppler != 0.0) {
        get_cexp_doppler(cexp_doppler, desc, length);
    }

    // --- Core Convolution Loop ---
    for (int i = 0; i < ((int)length - dd); i++) {
        for (int ii = 0; ii < desc->nb_rx; ii++) {
            struct complexf rx_tmp = {0.0f, 0.0f};

            for (int j = 0; j < desc->nb_tx; j++) {
                struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];

                for (int l = 0; l < (int)desc->channel_length; l++) {
                    if ((i - l) >= 0) {
                        // 1. Get the past transmitted signal (float)
                        struct complexf tx;
                        tx.r = tx_sig_re[j][i - l];
                        tx.i = tx_sig_im[j][i - l];

                        // 2. Perform complex multiplication with mixed precision.
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

            // --- Finalization and Storage ---
            rx_sig_re[ii][i + dd] = rx_tmp.r * path_loss;
            rx_sig_im[ii][i + dd] = rx_tmp.i * path_loss;

        } // ii (nb_rx)
    } // i (length)
}

#endif
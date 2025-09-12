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

#include "oai_cuda.h"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

typedef struct complexd {
  double r;
  double i;
} complexd;

#define CHECK_CUDA(val)                                                                          \
  {                                                                                              \
    if (val != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(val)); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

static void prepare_channel_coeffs(float2* h_out, channel_desc_t* desc, int nb_tx, int nb_rx)
{
  int idx = 0;
  for (int rx = 0; rx < nb_rx; ++rx) {
    for (int tx = 0; tx < nb_tx; ++tx) {
      struct complexd* channel_model = desc->ch[rx + (tx * nb_rx)];
      for (int l = 0; l < desc->channel_length; ++l) {
        h_out[idx].x = (float)channel_model[l].r;
        h_out[idx].y = (float)channel_model[l].i;
        idx++;
      }
    }
  }
}

struct GpuContext {
  cudaStream_t stream;
  curandState_t* d_curand_states;

  float* h_pinned_input;
  c16_t* h_pinned_output;

  float* d_input_padded;
  float2* d_channel_output;
  short2* d_final_output;
  float2* d_channel_coeffs;

  c16_t* cpu_history_buffer;
  int channel_length;
  int nb_tx;
};

extern "C" void vrtsim_cuda_init(void** context_handle, int max_samples, int nb_tx, int nb_rx, int channel_length)
{
  printf("[CUDA] Initializing vrtsim GPU context...\n");

  GpuContext* ctx = new GpuContext();
  ctx->channel_length = channel_length;
  ctx->nb_tx = nb_tx;

  CHECK_CUDA(cudaStreamCreate(&ctx->stream));

  const int max_padded_samples = max_samples + channel_length - 1;
  CHECK_CUDA(cudaMalloc(&ctx->d_input_padded, max_padded_samples * nb_tx * sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&ctx->d_channel_output, max_samples * nb_rx * sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&ctx->d_final_output, max_samples * nb_rx * sizeof(short2)));
  CHECK_CUDA(cudaMalloc(&ctx->d_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2)));

  CHECK_CUDA(cudaMallocHost(&ctx->h_pinned_input, max_padded_samples * nb_tx * sizeof(float2)));
  CHECK_CUDA(cudaMallocHost(&ctx->h_pinned_output, max_samples * nb_rx * sizeof(c16_t)));

  const int num_rand_elements = max_samples * nb_rx;
  ctx->d_curand_states = (curandState_t*)create_and_init_curand_states_cuda(num_rand_elements, time(NULL));

  const int history_size = (channel_length > 0) ? (channel_length - 1) * nb_tx : 0;
  ctx->cpu_history_buffer = new c16_t[history_size];
  memset(ctx->cpu_history_buffer, 0, history_size * sizeof(c16_t));

  *context_handle = ctx;

  printf("[CUDA] GPU context initialized successfully.\n");
}

extern "C" void vrtsim_cuda_shutdown(void* context_handle)
{
  printf("[CUDA] Shutting down vrtsim GPU context...\n");

  if (context_handle == nullptr) {
    return;
  }

  GpuContext* ctx = (GpuContext*)context_handle;

  CHECK_CUDA(cudaFree(ctx->d_input_padded));
  CHECK_CUDA(cudaFree(ctx->d_channel_output));
  CHECK_CUDA(cudaFree(ctx->d_final_output));
  CHECK_CUDA(cudaFree(ctx->d_channel_coeffs));

  CHECK_CUDA(cudaFreeHost(ctx->h_pinned_input));
  CHECK_CUDA(cudaFreeHost(ctx->h_pinned_output));

  destroy_curand_states_cuda(ctx->d_curand_states);

  CHECK_CUDA(cudaStreamDestroy(ctx->stream));

  delete[] ctx->cpu_history_buffer;

  delete ctx;

  printf("[CUDA] GPU context shut down successfully.\n");
}

static inline float* get_tx_ptr(float* base, int tx_ant, int padded_len)
{
  return base + (tx_ant * padded_len * 2);
}

static inline c16_t* get_hist_ptr(c16_t* base, int tx_ant, int hist_len)
{
  return base + (tx_ant * hist_len);
}

extern "C" void vrtsim_cuda_process(void* context_handle,
                                    c16_t** input_samples,
                                    int nsamps,
                                    int nb_tx,
                                    int nb_rx,
                                    channel_desc_t* channel_desc,
                                    float sigma2,
                                    double ts,
                                    uint16_t pdu_bit_map,
                                    uint16_t ptrs_bit_map,
                                    c16_t* final_output_buffer)
{
  nvtxRangePushA("vrtsim_cuda_process_complete");

  GpuContext* ctx = (GpuContext*)context_handle;
  const int hist_len = ctx->channel_length > 0 ? ctx->channel_length - 1 : 0;
  const int padded_len = nsamps + hist_len;

  nvtxRangePushA("CPU_data_preparation");

  for (int i = 0; i < nb_tx; ++i) {
    float* h_in_ptr = get_tx_ptr(ctx->h_pinned_input, i, padded_len);
    c16_t* hist_ptr = get_hist_ptr(ctx->cpu_history_buffer, i, hist_len);
    c16_t* input_ptr = input_samples[i];

    for (int j = 0; j < hist_len; ++j) {
      h_in_ptr[j * 2] = (float)hist_ptr[j].r;
      h_in_ptr[j * 2 + 1] = (float)hist_ptr[j].i;
    }

    for (int j = 0; j < nsamps; ++j) {
      h_in_ptr[(hist_len + j) * 2] = (float)input_ptr[j].r;
      h_in_ptr[(hist_len + j) * 2 + 1] = (float)input_ptr[j].i;
    }

    if (hist_len > 0) {
      memcpy(hist_ptr, input_ptr + (nsamps - hist_len), hist_len * sizeof(c16_t));
    }
  }
  nvtxRangePop(); // End CPU_data_preparation

  nvtxRangePushA("channel_coeffs_transfer");

  const int coeffs_size = nb_tx * nb_rx * ctx->channel_length;
  float2 h_temp_coeffs[coeffs_size];
  prepare_channel_coeffs(h_temp_coeffs, channel_desc, nb_tx, nb_rx);

  size_t channel_bytes = coeffs_size * sizeof(float2);
  CHECK_CUDA(cudaMemcpyAsync(ctx->d_channel_coeffs, h_temp_coeffs, channel_bytes, cudaMemcpyHostToDevice, ctx->stream));

  nvtxRangePop(); // End channel_coeffs_transfer

  nvtxRangePushA("GPU_pipeline_execution");

  nvtxRangePushA("input_data_H2D");
  size_t input_bytes = padded_len * nb_tx * sizeof(float2);
  CHECK_CUDA(cudaMemcpyAsync(ctx->d_input_padded, ctx->h_pinned_input, input_bytes, cudaMemcpyHostToDevice, ctx->stream));
  nvtxRangePop(); // End input_data_H2D

  nvtxRangePushA("multipath_kernel");
  dim3 threads_multipath(512, 1);
  dim3 blocks_multipath((nsamps + threads_multipath.x - 1) / threads_multipath.x, nb_rx);
  size_t sharedMemSize = (threads_multipath.x + ctx->channel_length - 1) * sizeof(float2);
  multipath_channel_kernel<<<blocks_multipath, threads_multipath, sharedMemSize, ctx->stream>>>(ctx->d_channel_coeffs,
                                                                                                ctx->d_input_padded,
                                                                                                ctx->d_channel_output,
                                                                                                nsamps,
                                                                                                ctx->channel_length,
                                                                                                nb_tx,
                                                                                                nb_rx);
  nvtxRangePop(); // End multipath_kernel

  nvtxRangePushA("noise_kernel");
  float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;
  dim3 threads_noise(256, 1);
  dim3 blocks_noise((nsamps + threads_noise.x - 1) / threads_noise.x, nb_rx);
  add_noise_and_phase_noise_kernel<<<blocks_noise, threads_noise, 0, ctx->stream>>>(ctx->d_channel_output,
                                                                                    ctx->d_final_output,
                                                                                    ctx->d_curand_states,
                                                                                    nsamps,
                                                                                    sqrtf(sigma2 / 2.0f),
                                                                                    sqrtf(pn_variance),
                                                                                    pdu_bit_map,
                                                                                    ptrs_bit_map);
  nvtxRangePop(); // End noise_kernel

  nvtxRangePushA("output_data_D2H");
  size_t output_bytes = nsamps * nb_rx * sizeof(c16_t);
  CHECK_CUDA(cudaMemcpyAsync(ctx->h_pinned_output, ctx->d_final_output, output_bytes, cudaMemcpyDeviceToHost, ctx->stream));
  nvtxRangePop(); // End output_data_D2H

  nvtxRangePop(); // End GPU_pipeline_execution

  nvtxRangePushA("synchronization_and_copy");
  CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

  memcpy(final_output_buffer, ctx->h_pinned_output, output_bytes);
  nvtxRangePop(); // End synchronization_and_copy
  nvtxRangePop(); // End vrtsim_cuda_process_complete
}

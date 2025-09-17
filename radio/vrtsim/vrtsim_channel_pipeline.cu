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

  c16_t* cpu_history_buffer;
  c16_t* h_pinned_history;
  c16_t* h_pinned_new_samples;
  c16_t* h_pinned_output;

  c16_t* d_history_buffer;
  c16_t* d_new_samples_buffer;
  float* d_input_padded;
  float2* d_channel_coeffs;
  float2* d_channel_output;
  short2* d_final_output;

  int channel_length;
  int nb_tx;
};

__global__ void prepare_input_kernel(float* d_out_padded,
                                     const c16_t* d_in_history,
                                     const c16_t* d_in_new,
                                     int nsamps,
                                     int hist_len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int padded_len = nsamps + hist_len;

  if (idx >= padded_len)
    return;

  c16_t sample;
  if (idx < hist_len) {
    sample = d_in_history[idx];
  } else {
    sample = d_in_new[idx - hist_len];
  }

  d_out_padded[idx * 2] = (float)sample.r;
  d_out_padded[idx * 2 + 1] = (float)sample.i;
}

__global__ void prepare_input_kernel_bulk(float* d_out_padded,
                                          const c16_t* d_in_history,
                                          const c16_t* d_in_new,
                                          int nsamps,
                                          int hist_len)
{
  const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_idx = blockIdx.y;

  const int padded_len = nsamps + hist_len;
  if (sample_idx >= padded_len)
    return;

  const c16_t* history_ptr = d_in_history + antenna_idx * hist_len;
  const c16_t* new_samples_ptr = d_in_new + antenna_idx * nsamps;
  float* output_ptr = d_out_padded + antenna_idx * padded_len * 2;

  c16_t sample;
  if (sample_idx < hist_len) {
    sample = history_ptr[sample_idx];
  } else {
    sample = new_samples_ptr[sample_idx - hist_len];
  }

  output_ptr[sample_idx * 2] = (float)sample.r;
  output_ptr[sample_idx * 2 + 1] = (float)sample.i;
}

extern "C" c16_t* vrtsim_cuda_get_pinned_output_buffer(void* context_handle)
{
  GpuContext* ctx = (GpuContext*)context_handle;
  return ctx->h_pinned_output;
}

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
  CHECK_CUDA(cudaMalloc(&ctx->d_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2)));

  const int history_size_per_ant = (channel_length > 0) ? channel_length - 1 : 0;
  CHECK_CUDA(cudaMalloc(&ctx->d_history_buffer, history_size_per_ant * nb_tx * sizeof(c16_t)));
  CHECK_CUDA(cudaMalloc(&ctx->d_new_samples_buffer, max_samples * nb_tx * sizeof(c16_t)));

  printf("[CUDA] Allocating pinned staging buffers...\n");
  CHECK_CUDA(cudaMallocHost(&ctx->h_pinned_history, history_size_per_ant * nb_tx * sizeof(c16_t)));
  CHECK_CUDA(cudaMallocHost(&ctx->h_pinned_new_samples, max_samples * nb_tx * sizeof(c16_t)));

  printf("[CUDA] Allocating output buffers...\n");
  CHECK_CUDA(cudaMalloc(&ctx->d_final_output, max_samples * nb_rx * sizeof(short2)));
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
  CHECK_CUDA(cudaFree(ctx->d_channel_coeffs));
  CHECK_CUDA(cudaFreeHost(ctx->h_pinned_history));
  CHECK_CUDA(cudaFreeHost(ctx->h_pinned_new_samples));
  CHECK_CUDA(cudaFree(ctx->d_final_output));
  CHECK_CUDA(cudaFreeHost(ctx->h_pinned_output));
  CHECK_CUDA(cudaFree(ctx->d_history_buffer));
  CHECK_CUDA(cudaFree(ctx->d_new_samples_buffer));

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
                                    uint16_t ptrs_bit_map)
{
  nvtxRangePushA("vrtsim_cuda_process_complete");

  GpuContext* ctx = (GpuContext*)context_handle;
  const int hist_len = ctx->channel_length > 0 ? ctx->channel_length - 1 : 0;
  const int padded_len = nsamps + hist_len;

  nvtxRangePushA("CPU_data_preparation");

  nvtxRangePushA("CPU_Memcpy_To_Pinned");
  for (int i = 0; i < nb_tx; ++i) {
    c16_t* hist_ptr = get_hist_ptr(ctx->cpu_history_buffer, i, hist_len);
    c16_t* input_ptr = input_samples[i];

    c16_t* h_pinned_hist_offset = ctx->h_pinned_history + i * hist_len;
    c16_t* h_pinned_new_offset = ctx->h_pinned_new_samples + i * nsamps;

    memcpy(h_pinned_hist_offset, hist_ptr, hist_len * sizeof(c16_t));
    memcpy(h_pinned_new_offset, input_ptr, nsamps * sizeof(c16_t));
  }
  nvtxRangePop();

  nvtxRangePushA("Async_H2D_Copies");
  size_t history_bytes = hist_len * nb_tx * sizeof(c16_t);
  size_t new_samples_bytes = nsamps * nb_tx * sizeof(c16_t);
  CHECK_CUDA(cudaMemcpyAsync(ctx->d_history_buffer, ctx->h_pinned_history, history_bytes, cudaMemcpyHostToDevice, ctx->stream));
  CHECK_CUDA(cudaMemcpyAsync(ctx->d_new_samples_buffer,
                             ctx->h_pinned_new_samples,
                             new_samples_bytes,
                             cudaMemcpyHostToDevice,
                             ctx->stream));
  nvtxRangePop();

  nvtxRangePushA("prepare_input_kernel_launch");
  dim3 threads(256, 1);
  dim3 blocks((padded_len + threads.x - 1) / threads.x, nb_tx);
  prepare_input_kernel_bulk<<<blocks, threads, 0, ctx->stream>>>(ctx->d_input_padded,
                                                                 ctx->d_history_buffer,
                                                                 ctx->d_new_samples_buffer,
                                                                 nsamps,
                                                                 hist_len);
  nvtxRangePop();

  nvtxRangePushA("CPU_History_Update");
  for (int i = 0; i < nb_tx; ++i) {
    if (hist_len > 0) {
      c16_t* hist_ptr = get_hist_ptr(ctx->cpu_history_buffer, i, hist_len);
      c16_t* input_ptr = input_samples[i];
      memcpy(hist_ptr, input_ptr + (nsamps - hist_len), hist_len * sizeof(c16_t));
    }
  }
  nvtxRangePop();

  nvtxRangePop(); // End CPU_data_preparation

  nvtxRangePushA("channel_coeffs_transfer");

  const int coeffs_size = nb_tx * nb_rx * ctx->channel_length;
  float2 h_temp_coeffs[coeffs_size];
  prepare_channel_coeffs(h_temp_coeffs, channel_desc, nb_tx, nb_rx);

  size_t channel_bytes = coeffs_size * sizeof(float2);
  CHECK_CUDA(cudaMemcpyAsync(ctx->d_channel_coeffs, h_temp_coeffs, channel_bytes, cudaMemcpyHostToDevice, ctx->stream));

  nvtxRangePop(); // End channel_coeffs_transfer

  nvtxRangePushA("GPU_pipeline_execution");

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

  nvtxRangePushA("Stream_Synchronize");
  CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
  nvtxRangePop(); // End Stream_Synchronize

  nvtxRangePop(); // End Final_Memcpy_To_Output
  nvtxRangePop(); // End vrtsim_cuda_process_complete
}

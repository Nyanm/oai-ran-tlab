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

/*! \file nrLDPC_decoder_cuda.h
 * \brief Defines the CUDA version of nrLDPC decoder, including initialization and driver warmup mechanisms.
 * \author Qizhi Pan, Raymond Knopp
 * \company EURECOM
 * \email: qizhi.pan@eurecom.fr, raymond.knopp@eurecom.fr
 * \date 2025-12-30
 * \version 1.0
 * \note Optimized for NVIDIA GH200 (Grace Hopper) architecture using Zero-Copy access.
 * \warning
 */
#include <stdint.h>
#include "PHY/sse_intrin.h"
#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"
#include "nrLDPC_cnProc.h"
#include "nrLDPC_bnProc.h"
#include "openair1/PHY/CODING/coding_defs.h"

#include "openair1/PHY/CODING/nrLDPC_extern.h"

#ifdef NR_LDPC_DEBUG_MODE
#include "nrLDPC_tools/nrLDPC_debug.h"
#endif

// decoder interface
/**
   \brief LDPC decoder API type definition
   \param p_decParams LDPC decoder parameters
   \param p_llr Input LLRs
   \param p_llrOut Output vector
   \param p_profiler LDPC profiler statistics
*/

//--------------------------CUDA Area---------------------------
#include <cuda_runtime.h>
#include "nrLDPC_CUDA_shared_param.h"

static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
static bool decoder_streamsCreated = false;
cudaError_t Err;

int8_t* cnProcBuf_dev;
int8_t* bnProcBuf_dev;
int8_t* llrRes_dev;
int8_t* llrProcBuf_dev;

int cuda_support_init_decoder()
{
  // use cudaMalloc for all inner buffers
  cudaError_t err;

  err = cudaMalloc((void**)&cnProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF);
  AssertFatal(err == cudaSuccess, "CUDA Error (cnProcBuf_dev): %s\n", cudaGetErrorString(err));

  err = cudaMalloc((void**)&bnProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF);
  AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));

  err = cudaMalloc((void**)&llrRes_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
  AssertFatal(err == cudaSuccess, "CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));

  err = cudaMalloc((void**)&llrProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
  AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));

  printf("[CUDA] Intermediate buffers allocated in HBM3 (Device Memory).\n");

  return 0;
}

static ldpc_cuda_bridge_t* stream_bridges[8];

extern void nrLDPC_decoder_cuda_GraphRecord(ldpc_cuda_bridge_t* buffer,
                                            uint32_t numLLR,
                                            int8_t* cnProcBuf,
                                            int8_t* bnProcBuf,
                                            int8_t* llrRes,
                                            int8_t* llrProcBuf,
                                            uint32_t Z,
                                            uint32_t K,
                                            uint8_t BG,
                                            uint8_t R,
                                            uint8_t numMaxIter,
                                            uint8_t n_segments,
                                            e_nrLDPC_outMode outMode,
                                            cudaStream_t* streams,
                                            uint8_t CudaStreamIdx,
                                            cudaGraph_t* graphPtr,
                                            cudaGraphExec_t* graphExecPtr,
                                            uint8_t* isCreatedFlag);

extern cudaError_t nrLDPC_decoder_cuda_GraphExecute(cudaGraphExec_t graphExec,
                                                    cudaStream_t stream,
                                                    cudaEvent_t* doneEvent,
                                                    uint8_t CudaStreamIdx);

extern void nrLDPC_decoder_cuda_NormalExecute(ldpc_cuda_bridge_t* buffer,
                                              uint32_t numLLR,
                                              int8_t* cnProcBuf,
                                              int8_t* bnProcBuf,
                                              int8_t* llrRes,
                                              int8_t* llrProcBuf,
                                              uint32_t Z,
                                              uint32_t K,
                                              uint8_t BG,
                                              uint8_t R,
                                              uint8_t numMaxIter,
                                              uint8_t n_segments,
                                              e_nrLDPC_outMode outMode,
                                              cudaStream_t* streams,
                                              uint8_t CudaStreamIdx,
                                              cudaEvent_t* doneEvent);

static inline uint32_t nrLDPC_decoder_core_dynamic(int8_t* p_llr,
                                                   int8_t* p_out,
                                                   int n_segments,
                                                   t_nrLDPC_dec_params* p_decParams,
                                                   t_nrLDPC_time_stats* p_profiler,
                                                   decode_abort_t* ab);
#define MAX_GRAPH_CACHE_SIZE 16
#define PRE_RECORDED_COUNT 6
#define STATIC_SEG_SIZE 1 // n_segments in pre-record graphs, should be determined for real cases

typedef struct {
  uint32_t Z;
  uint32_t K;
  uint32_t numLLR;
  uint8_t R;
  uint8_t BG;
  uint8_t numMaxIter;
  uint16_t n_segments;
  e_nrLDPC_outMode outMode;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  ldpc_cuda_bridge_t* bridge_ptr;
  bool occupied;
} gpu_graph_node_t;

static gpu_graph_node_t gpu_graph_cache[MAX_GRAPH_CACHE_SIZE];
static int dynamic_cache_idx = PRE_RECORDED_COUNT;

void init_decoder_warmup()
{
  // =====================================================================
  // CUDA Driver Warm-up
  // Purpose: Execute a few representative graphs to trigger CUDA context 
  // initialization and driver-level JIT/lazy loading.
  // Note: These specific Z/R combinations might not match the actual 
  // run-time traffic, but running them ensures the GPU pipeline is ready.
  // =====================================================================
  
  // Sample configurations for warmup
  uint32_t Z_list[] = {320, 352, 384}; 
  uint8_t R_list[] = {13, 23};
  int node_idx = 0;

  int8_t* dummy_input_llr = NULL;
  int8_t* dummy_output_bits = NULL;
  uint32_t max_z = 384;
  uint32_t max_n_segs = STATIC_SEG_SIZE;

  size_t input_size_bytes = 68 * max_z * max_n_segs * sizeof(int8_t);
  size_t output_size_bytes = 8448 * max_n_segs * sizeof(int8_t);

  // Allocate Pinned/Mapped Memory for Zero-Copy access (GH200 friendly)
  cudaHostAlloc((void**)&dummy_input_llr, input_size_bytes, cudaHostAllocMapped);
  cudaHostAlloc((void**)&dummy_output_bits, output_size_bytes, cudaHostAllocMapped);

  memset(dummy_input_llr, 0, input_size_bytes);
  memset(dummy_output_bits, 0, output_size_bytes);

  printf("[CUDA] Initializing & Warming up Driver Pipeline...\n");

  for (int r_idx = 0; r_idx < 2; r_idx++) {
    for (int z_idx = 0; z_idx < 3; z_idx++) {
      uint32_t Z = Z_list[z_idx];
      uint8_t R = R_list[r_idx];
      uint8_t BG = 1;
      uint32_t K = 22 * Z;
      uint32_t numLLR = (R == 13) ? NR_LDPC_NCOL_BG1_R13 * Z : NR_LDPC_NCOL_BG1_R23 * Z;
      uint8_t numMaxIter = 2; // 2 iterations for faster warmup
      uint8_t n_segments = STATIC_SEG_SIZE;

      // Bind dummy buffers
      gpu_graph_cache[node_idx].bridge_ptr->p_llr_ptr = dummy_input_llr;
      gpu_graph_cache[node_idx].bridge_ptr->p_out_ptr = dummy_output_bits;

      // Record graph 
      nrLDPC_decoder_cuda_GraphRecord(gpu_graph_cache[node_idx].bridge_ptr,
                                      numLLR,
                                      cnProcBuf_dev,
                                      bnProcBuf_dev,
                                      llrRes_dev,
                                      llrProcBuf_dev,
                                      Z,
                                      K,
                                      BG,
                                      R,
                                      numMaxIter,
                                      n_segments,
                                      nrLDPC_outMode_BIT,
                                      decoderStreams,
                                      0,
                                      &gpu_graph_cache[node_idx].graph,
                                      &gpu_graph_cache[node_idx].exec,
                                      (uint8_t*)&gpu_graph_cache[node_idx].occupied);

      // Save metadata
      gpu_graph_cache[node_idx].Z = Z;
      gpu_graph_cache[node_idx].R = R;
      gpu_graph_cache[node_idx].K = K;
      gpu_graph_cache[node_idx].numLLR = numLLR;
      gpu_graph_cache[node_idx].BG = BG;
      gpu_graph_cache[node_idx].numMaxIter = numMaxIter;
      gpu_graph_cache[node_idx].n_segments = n_segments;
      gpu_graph_cache[node_idx].outMode = nrLDPC_outMode_BIT;

      node_idx++;
    }
  }
  dynamic_cache_idx = node_idx;

  // Execute to trigger driver initialization
  if (dynamic_cache_idx > 0) {
    for (int i = 0; i < dynamic_cache_idx; i++) {
      if (gpu_graph_cache[i].occupied) {
        nrLDPC_decoder_cuda_GraphExecute(gpu_graph_cache[i].exec, decoderStreams[0], NULL, 0);
      }
    }
    cudaDeviceSynchronize();
    printf("[CUDA] Driver warm-up complete. Executed %d dummy graphs.\n", dynamic_cache_idx);

    // Mark slots as free and reset index so real traffic starts from slot 0
    for (int i = 0; i < dynamic_cache_idx; i++) {
        gpu_graph_cache[i].occupied = false;
    }
    dynamic_cache_idx = 0;
    printf("[CUDA] Cache cleared. Ready for dynamic recording.\n");
  }

  cudaFreeHost(dummy_input_llr);
  cudaFreeHost(dummy_output_bits);
}


void init_decoder_gpu_structures()
{
  printf("[CUDA] Initializing Global GPU Structures...\n");
  // Bridge for graphs
  for (int i = 0; i < MAX_GRAPH_CACHE_SIZE; i++) {
    if (gpu_graph_cache[i].bridge_ptr == NULL) {
      cudaHostAlloc((void**)&gpu_graph_cache[i].bridge_ptr, sizeof(ldpc_cuda_bridge_t), cudaHostAllocMapped);

      gpu_graph_cache[i].bridge_ptr->p_llr_ptr = NULL;
      gpu_graph_cache[i].bridge_ptr->p_out_ptr = NULL;
      gpu_graph_cache[i].occupied = false;
    }
  }
  printf("[CUDA] Allocated %d Graph Bridges.\n", MAX_GRAPH_CACHE_SIZE);
  // Bridge for normal execute
  for (int i = 0; i < 8; i++) {
    if (stream_bridges[i] == NULL) {
      cudaHostAlloc((void**)&stream_bridges[i], sizeof(ldpc_cuda_bridge_t), cudaHostAllocMapped);
      stream_bridges[i]->p_llr_ptr = NULL;
      stream_bridges[i]->p_out_ptr = NULL;
    }
  }
  printf("[CUDA] Allocated %d Stream Bridges for Fallback case.\n", 8);
}

void init_decoder_graphs()
{
  for (int i = 0; i < MAX_GRAPH_CACHE_SIZE; i++) {
    gpu_graph_cache[i].occupied = false;
    gpu_graph_cache[i].graph = NULL;
    gpu_graph_cache[i].exec = NULL;
    gpu_graph_cache[i].bridge_ptr = NULL;
    gpu_graph_cache[i].Z = 0;
    gpu_graph_cache[i].R = 0;
  }

  dynamic_cache_idx = 0;

  printf("[decoder_graphs] initialized %d dynamic cache slots\n", MAX_GRAPH_CACHE_SIZE);
}

void free_graphs()
{
  for (int i = 0; i < MAX_GRAPH_CACHE_SIZE; i++) {
    if (gpu_graph_cache[i].occupied) {
      if (gpu_graph_cache[i].exec)
        cudaGraphExecDestroy(gpu_graph_cache[i].exec);
      if (gpu_graph_cache[i].graph)
        cudaGraphDestroy(gpu_graph_cache[i].graph);
      gpu_graph_cache[i].occupied = false;
    }
  }
  printf("[decoder_graphs] shutdown complete (Dynamic Cache Cleared)\n");
}

extern int cuda_support_set;

bool encoder_streamsCreated = false;
cudaStream_t encoderStreams[4];

int32_t LDPCinit_cuda()
{
  if (cuda_support_set == 0) {
    printf("Calling encoder initializations\n");
    cuda_support_init();
  }
  if (!decoder_streamsCreated) {
    for (int s = 0; s < 8; ++s) {
      cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
      cudaEventCreate(&decoderDoneEvents[s]);
    }
    decoder_streamsCreated = true;
  }

  if (!encoder_streamsCreated) {
    for (int s = 0; s < 4; ++s) {
      cudaStreamCreateWithFlags(&encoderStreams[s], cudaStreamNonBlocking);
    }
    encoder_streamsCreated = true;
  }
  printf("CUDA LDPC decoder initiating\n");
  cuda_support_init_decoder();
  init_decoder_graphs();
  init_decoder_gpu_structures();
  init_decoder_warmup();
  return 0;
}

int32_t LDPCshutdown_cuda()
{
  for (int s = 0; s < 8; ++s) {
    if (decoder_streamsCreated) {
      cudaEventDestroy(decoderDoneEvents[s]);
      cudaStreamDestroy(decoderStreams[s]);
    }
  }

  for (int s = 0; s < 4; s++) {
    if (encoder_streamsCreated) {
      cudaStreamDestroy(encoderStreams[s]);
    }
  }
  free_graphs();

  decoder_streamsCreated = false;
  encoder_streamsCreated = false;

  return 0;
}

int32_t LDPCdecoder_cuda(t_nrLDPC_dec_params* p_decParams,
                         int8_t* p_llr,
                         uint8_t* p_out,
                         t_nrLDPC_time_stats* p_profiler,
                         decode_abort_t* ab)
{
  if (!((p_decParams->R == 23 || p_decParams->R == 13) && p_decParams->BG == 1 && p_decParams->Z % 4 == 0 && p_decParams->Z >= 128
        && p_decParams->Z <= 384)) { // format check
    printf("Current format: BG = %d, R = %d, Zc = %d\n", p_decParams->BG, p_decParams->R, p_decParams->Z);
    AssertFatal(false, "Format cuda not support, only support BG = 1, Zc >= 128 and R = 13, 23 right now\n");
    return 0;
  }

  // Launch LDPC decoder core for all segments
  int n_segments = p_decParams->n_segments;

  int numIter = nrLDPC_decoder_core_dynamic(p_llr, p_out, n_segments, p_decParams, p_profiler, ab);

  set_abort(ab, false);

  return numIter;
}

/**
   \brief PerformsnrLDPC decoding of one code block
   \param p_llr Input LLRs
   \param p_out Output vector
   \param numLLR Number of LLRs
   \param p_decParamsnrLDPC decoder parameters
   \param p_profilernrLDPC profiler statistics
*/

static inline uint32_t nrLDPC_decoder_core_dynamic(int8_t* p_llr,
                                                   int8_t* p_out,
                                                   int n_segments,
                                                   t_nrLDPC_dec_params* p_decParams,
                                                   t_nrLDPC_time_stats* p_profiler,
                                                   decode_abort_t* ab)
{
  extern int pageable_uses_host;

  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R;
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  uint32_t K = Z * 22;
  // Calculate LLR size per segment based on Rate
  uint32_t numLLR = (R == 13) ? NR_LDPC_NCOL_BG1_R13 * Z : NR_LDPC_NCOL_BG1_R23 * Z;

  // =====================================================================================
  // Architecture Support: PCIe vs GH200 (Zero-Copy)
  // If pageable_uses_host is 0 (PCIe), we must use explicit cudaMemcpy.
  // If pageable_uses_host is 1 (GH200/Jetson), we use Direct/Zero-Copy access.
  // =====================================================================================

  int8_t* p_llr_dev = p_llr; // Default to host pointer (Zero-Copy path)
  int8_t* p_out_dev = p_out; // Default to host pointer (Zero-Copy path)

  // Calculate total buffer sizes
  size_t total_input_size = n_segments * numLLR * sizeof(int8_t);
  // Output size safety: assume worst-case unpacked bytes (K * n_segments)
  size_t total_output_size = n_segments * K * sizeof(int8_t);

  // Flag to track if explicit allocation/copy is needed
  int need_explicit_copy = (!pageable_uses_host);

  if (need_explicit_copy) {
    // Allocate device memory for Discrete GPU
    cudaError_t err_alloc_in = cudaMalloc((void**)&p_llr_dev, total_input_size);
    cudaError_t err_alloc_out = cudaMalloc((void**)&p_out_dev, total_output_size);

    if (err_alloc_in != cudaSuccess || err_alloc_out != cudaSuccess) {
      // TODO: Handle allocation failure gracefully (e.g., fallback to CPU or Assert)
    }

    // Copy Input data from Host to Device
    cudaMemcpyAsync(p_llr_dev, p_llr, total_input_size, cudaMemcpyHostToDevice, decoderStreams[0]);
  }

  int found_idx = -1;

  // Search in Graph Cache
  for (int i = 0; i < dynamic_cache_idx; i++) {
    if (gpu_graph_cache[i].occupied && gpu_graph_cache[i].Z == Z && gpu_graph_cache[i].R == R && gpu_graph_cache[i].BG == BG
        && gpu_graph_cache[i].K == K && gpu_graph_cache[i].numLLR == numLLR && gpu_graph_cache[i].numMaxIter == numMaxIter
        && gpu_graph_cache[i].n_segments == n_segments && gpu_graph_cache[i].outMode == outMode) {
      found_idx = i;
      break;
    }
  }

  if (found_idx >= 0) {
    // === Cache HIT: Execute Recorded Graph ===
    gpu_graph_cache[found_idx].bridge_ptr->p_llr_ptr = p_llr_dev;
    gpu_graph_cache[found_idx].bridge_ptr->p_out_ptr = p_out_dev;

    nrLDPC_decoder_cuda_GraphExecute(gpu_graph_cache[found_idx].exec,
                                     decoderStreams[0],
                                     NULL, // doneEvent
                                     0); // Stream Index
  } else if (dynamic_cache_idx < MAX_GRAPH_CACHE_SIZE) {
    // === Cache MISS: Record New Graph and Execute ===
    int new_idx = dynamic_cache_idx;

    gpu_graph_cache[new_idx].occupied = true;
    gpu_graph_cache[new_idx].Z = Z;
    gpu_graph_cache[new_idx].R = R;
    gpu_graph_cache[new_idx].BG = BG;
    gpu_graph_cache[new_idx].K = K;
    gpu_graph_cache[new_idx].numLLR = numLLR;
    gpu_graph_cache[new_idx].numMaxIter = numMaxIter;
    gpu_graph_cache[new_idx].n_segments = n_segments;
    gpu_graph_cache[new_idx].outMode = outMode;

    // Use the determined pointers (Device ptrs for PCIe, Host ptrs for GH200)
    gpu_graph_cache[new_idx].bridge_ptr->p_llr_ptr = p_llr_dev;
    gpu_graph_cache[new_idx].bridge_ptr->p_out_ptr = p_out_dev;

    nrLDPC_decoder_cuda_GraphRecord(gpu_graph_cache[new_idx].bridge_ptr,
                                    numLLR,
                                    cnProcBuf_dev,
                                    bnProcBuf_dev,
                                    llrRes_dev,
                                    llrProcBuf_dev,
                                    Z,
                                    K,
                                    BG,
                                    R,
                                    numMaxIter,
                                    n_segments,
                                    outMode,
                                    decoderStreams,
                                    0, // CudaStreamIdx
                                    &gpu_graph_cache[new_idx].graph,
                                    &gpu_graph_cache[new_idx].exec,
                                    (uint8_t*)&gpu_graph_cache[new_idx].occupied);

    nrLDPC_decoder_cuda_GraphExecute(gpu_graph_cache[new_idx].exec, decoderStreams[0], NULL, 0);

    dynamic_cache_idx++;

  } else {
    // === Cache FULL: Fallback to Normal Execution ===
    // If the cache is full, we cannot record new graphs. 
    // Execute kernel directly using standard launch.

    ldpc_cuda_bridge_t* perpack_buffer = stream_bridges[0];
    perpack_buffer->p_llr_ptr = p_llr_dev;
    perpack_buffer->p_out_ptr = p_out_dev;

    nrLDPC_decoder_cuda_NormalExecute(perpack_buffer,
                                      numLLR,
                                      cnProcBuf_dev,
                                      bnProcBuf_dev,
                                      llrRes_dev,
                                      llrProcBuf_dev,
                                      Z,
                                      K,
                                      BG,
                                      R,
                                      numMaxIter,
                                      n_segments,
                                      outMode,
                                      decoderStreams,
                                      0,
                                      NULL);
  }

  // Copy back and Cleanup for Discrete GPU
  if (need_explicit_copy) {
    // Copy Output from Device to Host
    cudaMemcpyAsync(p_out, p_out_dev, total_output_size, cudaMemcpyDeviceToHost, decoderStreams[0]);

    cudaStreamSynchronize(decoderStreams[0]);

    cudaFree(p_llr_dev);
    cudaFree(p_out_dev);
  } else {
    // For GH200/Zero-Copy, just wait for kernel completion
    cudaDeviceSynchronize();
  }

  return numMaxIter;
}
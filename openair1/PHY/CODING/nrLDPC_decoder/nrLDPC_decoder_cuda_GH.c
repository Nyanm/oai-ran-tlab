

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

/*!\file nrLDPC_decoder.c
 * \brief Defines thenrLDPC decoder
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

#define USE_STATIC_ALLOC
static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
static bool streamsCreated = false;
static bool SegmentPacked = false;
static int NumSegPacks = 0;
cudaError_t Err;

#define CUDAMALLOC \
  0 // set 1 to use gpu memory via HBM,
    //     0 to use cpu memory via NvLink C-C
#define USE_STATIC_ALLOC

//--------------------------------------------------------------
// debug function
void dumpASS(int8_t* cnProcBufRes, const char* filename)
{
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL * 8448; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}

//--------------------------------------------------------------

#ifdef USE_STATIC_ALLOC

static int8_t cnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t bnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t llrRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static int8_t llrProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
#else

int8_t* cnProcBuf_dev;
int8_t* bnProcBuf_dev;
int8_t* llrRes_dev;
int8_t* llrProcBuf_dev;

int8_t* cnProcBuf_host;
int8_t* bnProcBuf_host;
int8_t* llrRes_host;
int8_t* llrProcBuf_host;

extern int pageable, register_host;

int cuda_support_init_decoder()
{
  if (!pageable && !register_host) {
    cudaError_t err =
        cudaMalloc((void**)&cnProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF);
    AssertFatal(err == cudaSuccess, "CUDA Error (cnProcBuf_dev): %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&bnProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&llrRes_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&llrProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));

  } else {
    cudaError_t err = cudaHostAlloc((void**)&cnProcBuf_host,
                                    sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF,
                                    cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (c_dev): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&cnProcBuf_dev, cnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (cnProcBuf_host): %s\n", cudaGetErrorString(err));

    err = cudaHostAlloc((void**)&bnProcBuf_host,
                        sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF,
                        cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&bnProcBuf_dev, bnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));

    err = cudaHostAlloc((void**)&llrRes_host,
                        sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,
                        cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrRes_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&llrRes_dev, llrRes_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));

    err = cudaHostAlloc((void**)&llrProcBuf_host,
                        sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,
                        cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&llrProcBuf_dev, llrProcBuf_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));

    printf("All cudaHostAlloc done\n");
  }
  return 0;
}
#endif

extern void nrLDPC_decoder_scheduler_BG1_cuda_core(int8_t* p_out,
                                                   uint32_t numLLR,
                                                   int8_t* llr,
                                                   int8_t* cnProcBuf,
                                                   int8_t* bnProcBuf,
                                                   int8_t* llrRes,
                                                   int8_t* llrProcBuf,
                                                   int Z,
                                                   uint32_t K,
                                                   uint8_t BG,
                                                   uint8_t R,
                                                   uint8_t numMaxIter,
                                                   e_nrLDPC_outMode outMode,
                                                   cudaStream_t* streams,
                                                   uint8_t CudaStreamIdx,
                                                   cudaEvent_t* doneEvent);

extern void nrLDPC_decoder_cuda_GraphRecord(int8_t* p_out,
                                            uint32_t numLLR,
                                            int8_t* llr,
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
extern void nrLDPC_decoder_cuda_NormalExecute(int8_t* p_out,
                                              uint32_t numLLR,
                                              int8_t* llr,
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

static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab);

static inline uint32_t nrLDPC_decoder_core_dynamic(int8_t* p_llr,
                                                   int8_t* p_out,
                                                   int n_segments,
                                                   t_nrLDPC_dec_params* p_decParams,
                                                   t_nrLDPC_time_stats* p_profiler,
                                                   decode_abort_t* ab);
#define MAX_GRAPH_CACHE_SIZE 16
#define PRE_RECORDED_COUNT 6
#define STATIC_SEG_SIZE 1 // n_segments in pre-record graphs

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
  bool occupied;
} gpu_graph_node_t;

static gpu_graph_node_t gpu_graph_cache[MAX_GRAPH_CACHE_SIZE];
static int dynamic_cache_idx = PRE_RECORDED_COUNT;

void init_decoder_warmup(int8_t* input_llr, int8_t* output_bits)
{
  uint32_t Z_list[] = {320, 352, 384};
  uint8_t R_list[] = {13, 23};
  int node_idx = 0;

  printf("[CUDA] Starting pre-recording for 6 standard formats...\n");
  printf("input_llr: %p, output_bits: %p\n", input_llr, output_bits);
  for (int r_idx = 0; r_idx < 2; r_idx++) {
    for (int z_idx = 0; z_idx < 3; z_idx++) {
      uint32_t Z = Z_list[z_idx];
      uint8_t R = R_list[r_idx];
      uint8_t BG = 1;
      uint32_t K = 22 * Z;
      uint32_t numLLR = (R == 13) ? NR_LDPC_NCOL_BG1_R13 * Z : NR_LDPC_NCOL_BG1_R23 * Z;
      uint8_t numMaxIter = 4;
      uint8_t n_segments = STATIC_SEG_SIZE;

      nrLDPC_decoder_cuda_GraphRecord(output_bits,
                                      numLLR,
                                      input_llr,
                                      cnProcBuf,
                                      bnProcBuf,
                                      llrRes,
                                      llrProcBuf,
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

      // save parameters
      gpu_graph_cache[node_idx].Z = Z;
      gpu_graph_cache[node_idx].R = R;
      gpu_graph_cache[node_idx].K = K;
      gpu_graph_cache[node_idx].numLLR = numLLR;
      gpu_graph_cache[node_idx].BG = BG;
      gpu_graph_cache[node_idx].numMaxIter = numMaxIter;
      gpu_graph_cache[node_idx].n_segments = n_segments;
      gpu_graph_cache[node_idx].outMode = nrLDPC_outMode_BIT;

      printf("  - Recorded: Slot %d, Z=%d, R=%d, K=%d\n", node_idx, Z, R, K);
      node_idx++;
    }
  }
  dynamic_cache_idx = node_idx;

  if (dynamic_cache_idx > 0) {
    printf("[CUDA] Warming up the GPU pipeline with ALL %d recorded graphs...\n", dynamic_cache_idx);

    for (int i = 0; i < dynamic_cache_idx; i++) {
      if (gpu_graph_cache[i].occupied) {
        cudaError_t err = nrLDPC_decoder_cuda_GraphExecute(gpu_graph_cache[i].exec, decoderStreams[0], NULL, 0);
        cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          printf("[CUDA] Warm-up failed at slot %d (Z=%d, R=%d): %s\n",
                 i,
                 gpu_graph_cache[i].Z,
                 gpu_graph_cache[i].R,
                 cudaGetErrorString(err));
        }
      }
    }

    cudaDeviceSynchronize();
    printf("[CUDA] Warm-up complete. All templates validated.\n");
  }
}
void init_decoder_graphs()
{
  // This one should be removed
  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL; i++) {
    decoderGraphs[i] = NULL;
    decoderGraphExec[i] = NULL;
    graphCreated[i] = false;
  }

  for (int i = 0; i < MAX_GRAPH_CACHE_SIZE; i++) {
    gpu_graph_cache[i].occupied = false;
    gpu_graph_cache[i].graph = NULL;
    gpu_graph_cache[i].exec = NULL;
    gpu_graph_cache[i].Z = 0;
    gpu_graph_cache[i].R = 0;
  }

  dynamic_cache_idx = 0;

  printf("[decoder_graphs] initialized old slots and cleared %d dynamic cache slots\n", MAX_GRAPH_CACHE_SIZE);
}

void free_graphs()
{
  // This one should be removed as well
  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL; i++) {
    if (graphCreated[i]) {
      cudaGraphExecDestroy(decoderGraphExec[i]);
      cudaGraphDestroy(decoderGraphs[i]);
      graphCreated[i] = false;
    }
  }

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

int32_t LDPCinit_cuda(int8_t* input_llr, int8_t* output_bits)
{
  if (cuda_support_set == 0) {
    // printf("Are we here in the new init function?input:%p, output:%p\n", input_llr, output_bits);
    printf("Calling encoder initializations\n");
    cuda_support_init();
    printf("CUDA LDPC decoder initiating\n");
#ifndef USE_STATIC_ALLOC
    cuda_support_init_decoder();
#endif
  }
  if (!streamsCreated) {
    for (int s = 0; s < 8; ++s) {
      cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
      cudaEventCreate(&decoderDoneEvents[s]);
    }
    streamsCreated = true;
  }
  if (!encoder_streamsCreated) {
    for (int s = 0; s < 4; ++s) {
      cudaStreamCreateWithFlags(&encoderStreams[s], cudaStreamNonBlocking);
    }
    encoder_streamsCreated = true;
  }
  init_decoder_graphs();
  init_decoder_warmup(input_llr, output_bits);
  return 0;
}

int32_t LDPCshutdown_cuda()
{
  for (int s = 0; s < 8; ++s) {
    if (streamsCreated) {
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

  streamsCreated = false;
  encoder_streamsCreated = false;
  SegmentPacked = false;
  // d_mem_exist = false;

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

  // Launch LDPC decoder core for one segment
  int n_segments = p_decParams->n_segments;

  int numIter = nrLDPC_decoder_core_dynamic(p_llr, p_out, n_segments, p_decParams, p_profiler, ab);

  // int numIter = nrLDPC_decoder_core(p_llr, p_out, n_segments, p_decParams, p_profiler, ab); //old decoder module

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
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R;
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  uint32_t K = Z * 22;
  uint32_t numLLR = (R == 13) ? NR_LDPC_NCOL_BG1_R13 * Z : NR_LDPC_NCOL_BG1_R23 * Z;

  int found_idx = -1;

  // loop Cache to find the fit Graph
  for (int i = 0; i < dynamic_cache_idx; i++) {
    if (gpu_graph_cache[i].occupied && gpu_graph_cache[i].Z == Z && gpu_graph_cache[i].R == R && gpu_graph_cache[i].BG == BG
        && gpu_graph_cache[i].K == K && gpu_graph_cache[i].numLLR == numLLR && gpu_graph_cache[i].numMaxIter == numMaxIter
        && gpu_graph_cache[i].n_segments == n_segments && gpu_graph_cache[i].outMode == outMode) {
      found_idx = i;
      break;
    }
  }

  if (found_idx >= 0) {
    // === HIT: execute Graph ===
    nrLDPC_decoder_cuda_GraphExecute(gpu_graph_cache[found_idx].exec,
                                     decoderStreams[0],
                                     NULL, // doneEvent
                                     0); // Stream Index
  } else if (dynamic_cache_idx < MAX_GRAPH_CACHE_SIZE) {
    // printf("We need to record new graph\n");
    //  === MISS : record new Graph and execute ===
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

    nrLDPC_decoder_cuda_GraphRecord(p_out,
                                    numLLR,
                                    p_llr,
                                    cnProcBuf,
                                    bnProcBuf,
                                    llrRes,
                                    llrProcBuf,
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

    // printf("[CUDA Dynamic] Recorded new graph at slot %d (Z=%d, R=%d)\n", new_idx, Z, R);
  } else {
    // === MISS : Fallback ===

    // printf("We need to use normal execution\n");
    nrLDPC_decoder_cuda_NormalExecute(p_out,
                                      numLLR,
                                      p_llr,
                                      cnProcBuf,
                                      bnProcBuf,
                                      llrRes,
                                      llrProcBuf,
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
  cudaDeviceSynchronize();
  return numMaxIter;
}

/**
   \brief PerformsnrLDPC decoding of one code block
   \param p_llr Input LLRs
   \param p_out Output vector
   \param numLLR Number of LLRs
   \param p_decParamsnrLDPC decoder parameters
   \param p_profilernrLDPC profiler statistics
*/

static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab)
{
//  int8_t temp_out[n_segments * 8448] __attribute__((aligned(64)));
#ifdef USE_STATIC_ALLOC
//  memcpy(temp_in , p_llr ,  n_segments * 68 * 384);
//  memset(temp_out, 0     ,  n_segments * 8448);
#endif

  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 13,23,... for code rates 1/3, 2/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter; // To match the actual iterations
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  uint32_t K = Z * 22;
  uint32_t numLLR = (R == 13) ? NR_LDPC_NCOL_BG1_R13 * Z : NR_LDPC_NCOL_BG1_R23 * Z;

  // Pack setting area
  if (!SegmentPacked) {
    int segPerPack = 0;
    switch (R) {
      case 13:
        segPerPack = 132; // It's quite free here, GPU can handle this
        break; // And also, the best practice should be only use one stream in the whole decoding
      case 23: // So we set the maximum threads in one pack to a large number
        segPerPack = 132;
        break;
      default:
        printf("Not supporting R");
        return 0;
        break;
    }
    NumSegPacks = (n_segments + segPerPack - 1) / segPerPack;

    for (int p = 0; p < NumSegPacks; ++p) {
      segmentPacks[p].packIdx = p;
      segmentPacks[p].startSeg = p * segPerPack;
      segmentPacks[p].nSeg = (n_segments - p * segPerPack > segPerPack) ? segPerPack : n_segments - p * segPerPack;

      segmentPacks[p].stream = decoderStreams[p];
      segmentPacks[p].doneEvt = decoderDoneEvents[p];
    }
    SegmentPacked = true;
  }

  for (int SegPackIdx = 0; SegPackIdx < NumSegPacks; SegPackIdx++) {
    int PackShiftIdx = segmentPacks[SegPackIdx].startSeg;
#ifdef USE_STATIC_ALLOC
    int8_t* perpack_llr = p_llr + PackShiftIdx * 68 * 384;
    int8_t* perpack_cnProcBuf = cnProcBuf + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* perpack_bnProcBuf = bnProcBuf + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_llrProcBuf = llrProcBuf + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrRes = llrRes + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_out = p_out + PackShiftIdx * K;
#else
    int8_t* perpack_llr = p_llr + PackShiftIdx * 68 * 384;
    int8_t* perpack_cnProcBuf = cnProcBuf_dev + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* perpack_bnProcBuf = bnProcBuf_dev + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_llrRes = llrRes_dev + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrProcBuf = llrProcBuf_dev + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_out = p_out + PackShiftIdx * K;
#endif
    //  Call scheduler for this segment and stream
    //  Launch decoder on stream
    nrLDPC_decoder_scheduler_BG1_cuda_core(perpack_out,
                                           numLLR,
                                           perpack_llr,
                                           perpack_cnProcBuf,
                                           perpack_bnProcBuf,
                                           perpack_llrRes,
                                           perpack_llrProcBuf,
                                           Z,
                                           K,
                                           BG,
                                           R,
                                           numMaxIter,
                                           outMode,
                                           decoderStreams,
                                           SegPackIdx, // Index for the whole pack
                                           decoderDoneEvents);
  }

  for (int s = 0; s < NumSegPacks; ++s) {
    cudaEventSynchronize(decoderDoneEvents[s]); // stop until segment decode
  }
  cudaDeviceSynchronize();

  // cudaDeviceSynchronize();
  // printf("p_out %p, temp_out %p\n",p_out,temp_out);

  // dumpASS(p_out, "Dump_Output_Stream_GH.txt");

  return numMaxIter;
}

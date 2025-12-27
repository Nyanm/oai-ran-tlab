

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
    // err=cudaMalloc((void **)&cnProcBufRes_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 *
    // NR_LDPC_SIZE_CN_PROC_BUF); AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_dev): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&bnProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));
    // err=cudaMalloc((void **)&bnProcBufRes_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 *
    // NR_LDPC_SIZE_BN_PROC_BUF); AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBufRes_dev): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&llrRes_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&llrProcBuf_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));
    //err = cudaMalloc((void**)&llrOut_dev, sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    //AssertFatal(err == cudaSuccess, "CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));
    // err=cudaMalloc((void **)&iter_ptr_array_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4);
    // AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_dev): %s\n", cudaGetErrorString(err));
    // err=cudaMalloc((void **)&PC_Flag_array_dev,sizeof(int)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4);
    // AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_dev): %s\n", cudaGetErrorString(err));
  } else {
    cudaError_t err = cudaHostAlloc((void**)&cnProcBuf_host,
                                    sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF,
                                    cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (c_dev): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&cnProcBuf_dev, cnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (cnProcBuf_host): %s\n", cudaGetErrorString(err));

    /// err=cudaHostAlloc((void **)&cnProcBufRes_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 *
    /// NR_LDPC_SIZE_CN_PROC_BUF,cudaHostAllocMapped);
    // AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_host): %s\n", cudaGetErrorString(err));
    // err = cudaHostGetDevicePointer((void**)&cnProcBufRes_dev, cnProcBufRes_host, 0);
    // AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_dev): %s\n", cudaGetErrorString(err));

    err = cudaHostAlloc((void**)&bnProcBuf_host,
                        sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF,
                        cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&bnProcBuf_dev, bnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess, "CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));

    // err=cudaHostAlloc((void **)&bnProcBufRes_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 *
    // NR_LDPC_SIZE_BN_PROC_BUF,cudaHostAllocMapped); AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBufRes_host): %s\n",
    // cudaGetErrorString(err)); err = cudaHostGetDevicePointer((void**)&bnProcBufRes_dev, bnProcBufRes_host, 0); AssertFatal(err ==
    // cudaSuccess,"CUDA Error (bnProcBufRes_dev): %s\n", cudaGetErrorString(err));

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

    //err = cudaHostAlloc((void**)&llrOut_host,
    //                    sizeof(int8_t) * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,
     //                   cudaHostAllocMapped);
    //AssertFatal(err == cudaSuccess, "CUDA Error (llrOut_host): %s\n", cudaGetErrorString(err));
    //err = cudaHostGetDevicePointer((void**)&llrOut_dev, llrOut_host, 0);
    //AssertFatal(err == cudaSuccess, "CUDA Error (llrOut_dev): %s\n", cudaGetErrorString(err));

    // err=cudaHostAlloc((void **)&iter_ptr_array_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4,cudaHostAllocMapped);
    // AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_host): %s\n", cudaGetErrorString(err));
    // err = cudaHostGetDevicePointer((void**)&iter_ptr_array_dev, iter_ptr_array_host, 0);
    // AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_dev): %s\n", cudaGetErrorString(err));

    // err=cudaHostAlloc((void **)&PC_Flag_array_host,sizeof(int)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4,cudaHostAllocMapped);
    // AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_host): %s\n", cudaGetErrorString(err));
    // err = cudaHostGetDevicePointer((void**)&PC_Flag_array_dev, PC_Flag_array_host, 0);
    // AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_dev): %s\n", cudaGetErrorString(err));

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

static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           uint32_t numLLR, 
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab);

void init_decoder_graphs()
{
  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL; i++) {
    decoderGraphs[i] = NULL;
    decoderGraphExec[i] = NULL;
    graphCreated[i] = false;
  }
  printf("[decoder_graphs] initialized %d slots\n", MAX_NUM_DLSCH_SEGMENTS_DL);
}

void free_graphs()
{
  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL; i++) {
    if (graphCreated[i]) {
      cudaGraphExecDestroy(decoderGraphExec[i]);
      cudaGraphDestroy(decoderGraphs[i]);
      graphCreated[i] = false;
    }
  }
  printf("[decoder_graphs] shutdown complete\n");
}

extern int cuda_support_set;

bool encoder_streamsCreated = false;
cudaStream_t encoderStreams[4];

int32_t LDPCinit_cuda()
{
  if (cuda_support_set == 0) {
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

  for (int s=0; s< 4; s++) {
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
  if (!((p_decParams->R == 23 || p_decParams->R == 13) && p_decParams->BG == 1 && p_decParams->Z%4 == 0 && p_decParams->Z >= 128 && p_decParams->Z <= 384)) { // format check
    printf("Current format: BG = %d, R = %d, Zc = %d\n", p_decParams->BG, p_decParams->R, p_decParams->Z);
    AssertFatal(false, "Format cuda not support, only support BG = 1, Zc >= 128 and R = 13, 23 right now\n");
    return 0;
  }
  uint32_t numLLR;
  if (p_decParams->R == 13) {
    numLLR = NR_LDPC_NCOL_BG1_R13 * p_decParams->Z;
  } else {
    numLLR = NR_LDPC_NCOL_BG1_R23 * p_decParams->Z;
  }
  // Launch LDPC decoder core for one segment
  int n_segments = p_decParams->n_segments;
  int numIter = nrLDPC_decoder_core(p_llr, (int8_t*)p_out, n_segments, numLLR, p_decParams, p_profiler, ab);

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

static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           uint32_t numLLR,
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
  uint32_t K = Z*22;

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

  return numMaxIter;
}

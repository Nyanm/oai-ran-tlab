

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
#include "decoder_graphs.h"

static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
static bool streamsCreated = false;

cudaError_t Err;

#define CUDAMALLOC \
  0 // set 1 to use gpu memory via HBM,
    //     0 to use cpu memory via NvLink C-C

#if CUDAMALLOC
static int8_t* d_cnProcBuf = NULL;
static int8_t* d_cnProcBufRes = NULL;
static int8_t* d_bnProcBuf = NULL;
static int8_t* d_bnProcBufRes = NULL;
static int8_t* d_llrRes = NULL;
static int8_t* d_llrProcBuf = NULL;
static int8_t* d_llrOut = NULL;
static int8_t* d_temp_out = NULL;
static int8_t* d_temp_in = NULL;
static int8_t* d_iter_ptr_array = NULL;
static int* d_PC_Flag_array = NULL;

static bool p_lutCreated = false;
t_nrLDPC_lut* p_lut_dev = NULL;

void check_lut_pointers(const t_nrLDPC_lut* lut)
{
  if (!lut) {
    printf("check_lut_pointers: lut is NULL\n");
    return;
  }

  printf("Checking LUT pointers:\n");
  printf("startAddrCnGroups       = %p\n", (void*)lut->startAddrCnGroups);
  printf("numCnInCnGroups         = %p\n", (void*)lut->numCnInCnGroups);
  printf("numBnInBnGroups         = %p\n", (void*)lut->numBnInBnGroups);
  printf("startAddrBnGroups       = %p\n", (void*)lut->startAddrBnGroups);
  printf("startAddrBnGroupsLlr    = %p\n", (void*)lut->startAddrBnGroupsLlr);
  printf("llr2llrProcBufAddr      = %p\n", (void*)lut->llr2llrProcBufAddr);
  printf("llr2llrProcBufBnPos     = %p\n", (void*)lut->llr2llrProcBufBnPos);

  printf("circShift               = %p\n", (void*)lut->circShift);
  printf("startAddrBnProcBuf       = %p\n", (void*)lut->startAddrBnProcBuf);
  printf("bnPosBnProcBuf           = %p\n", (void*)lut->bnPosBnProcBuf);
  printf("posBnInCnProcBuf         = %p\n", (void*)lut->posBnInCnProcBuf);
}
#else
static bool SegmentPacked = false;
static int NumSegPacks = 0;
//static int currentStreamCount = 0;
static int8_t iter_ptr_array[MAX_NUM_DLSCH_SEGMENTS_DL];
static int PC_Flag_array[MAX_NUM_DLSCH_SEGMENTS_DL];
static int8_t cnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t cnProcBufRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t bnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t bnProcBufRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
static int8_t llrRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static int8_t llrProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static int8_t llrOut[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static int8_t temp_out[MAX_NUM_DLSCH_SEGMENTS_DL * 8448] __attribute__((aligned(64)));
static int8_t temp_in[MAX_NUM_DLSCH_SEGMENTS_DL * 68 * 384] __attribute__((aligned(64)));
static int8_t iter_ptr_array[MAX_NUM_DLSCH_SEGMENTS_DL];
static int PC_Flag_array[MAX_NUM_DLSCH_SEGMENTS_DL];
#endif
extern void nrLDPC_decoder_scheduler_BG1_cuda_core(const t_nrLDPC_lut* p_lut,
                                                   int8_t* p_out,
                                                   uint32_t numLLR,
                                                   int8_t* llr,
                                                   int8_t* cnProcBuf,
                                                   int8_t* cnProcBufRes,
                                                   int8_t* bnProcBuf,
                                                   int8_t* bnProcBufRes,
                                                   int8_t* llrRes,
                                                   int8_t* llrProcBuf,
                                                   int8_t* llrOut,
                                                   int8_t* p_llrOut,
                                                   int Z,
                                                   uint8_t BG,
                                                   uint8_t R,
                                                   uint8_t numMaxIter,
                                                   e_nrLDPC_outMode outMode,
                                                   cudaStream_t* streams,
                                                   uint8_t CudaStreamIdx,
                                                   cudaEvent_t* doneEvent,
                                                   int8_t* iter_ptr,
                                                   int* PC_Flag);

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
static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           uint32_t numLLR,
                                           t_nrLDPC_lut* p_lut,
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

int32_t LDPCinit_cuda()
{
  printf("Calling encoder initializations\n");
  if (cuda_support_set == 0)
    cuda_support_init();
  printf("CUDA LDPC decoder initiating\n");
#if CUDAMALLOC
  size_t cn_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF * sizeof(int8_t);
  size_t bn_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF * sizeof(int8_t);
  size_t llr_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);
  size_t llrOut_bytes = NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);

  // cudaGetDevice(&gpuDeviceId); // get device id

  cudaError_t err;
  err = cudaMalloc((void**)&d_cnProcBuf, cn_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_cnProcBuf failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_cnProcBufRes, cn_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_cnProcBufRes failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_bnProcBuf, bn_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_bnProcBuf failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_bnProcBufRes, bn_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_bnProcBufRes failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_llrRes, llr_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_llrRes failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_llrProcBuf, llr_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMallocManaged d_llrProcBuf failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMallocManaged((void**)&d_iter_ptr_array, MAX_NUM_DLSCH_SEGMENTS_DL * sizeof(int8_t), cudaMemAttachGlobal);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMallocManaged iter_ptr_array failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMallocManaged((void**)&d_PC_Flag_array, MAX_NUM_DLSCH_SEGMENTS_DL * sizeof(int), cudaMemAttachGlobal);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMallocManaged PC_Flag_array failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_llrOut, MAX_NUM_DLSCH_SEGMENTS_DL * llrOut_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_pp_llrOut failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_temp_out, MAX_NUM_DLSCH_SEGMENTS_DL * 8448 * sizeof(uint8_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_temp_in, MAX_NUM_DLSCH_SEGMENTS_DL * 68 * 384 * sizeof(uint8_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_in failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
#endif
  if (!streamsCreated) {
    for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
      cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
      cudaEventCreate(&decoderDoneEvents[s]);
    }
    streamsCreated = true;
  }
  init_decoder_graphs();
  return 0;
}

int32_t LDPCshutdown_cuda()
{
#if CUDAMALLOC
  if (d_cnProcBuf)
    cudaFree(d_cnProcBuf);
  if (d_cnProcBufRes)
    cudaFree(d_cnProcBufRes);
  if (d_bnProcBuf)
    cudaFree(d_bnProcBuf);
  if (d_bnProcBufRes)
    cudaFree(d_bnProcBufRes);
  if (d_llrRes)
    cudaFree(d_llrRes);
  if (d_llrProcBuf)
    cudaFree(d_llrProcBuf);
  if (d_llrOut)
    cudaFree(d_llrOut);
  if (d_temp_in)
    cudaFree(d_temp_in);
  if (d_temp_out)
    cudaFree(d_temp_out);
#endif
  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
    if (streamsCreated) {
      cudaEventDestroy(decoderDoneEvents[s]);
      cudaStreamDestroy(decoderStreams[s]);
    }
  }

  free_graphs();

  streamsCreated = false;
#if CUDAMALLOC
  p_lutCreated = false;
#endif
  // d_mem_exist = false;

  return 0;
}

int32_t LDPCdecoder_cuda(t_nrLDPC_dec_params* p_decParams,
                         // uint8_t harq_pid,
                         // uint8_t ulsch_id,
                         // uint8_t C,
                         int8_t* p_llr,
                         uint8_t* p_out,
                         t_nrLDPC_time_stats* p_profiler,
                         decode_abort_t* ab)
{
  if (!((p_decParams->R == 23 || p_decParams->R == 13) && p_decParams->BG == 1 && p_decParams->Z == 384)) { // format check
    printf("Current format: BG = %d, R = %d, Zc = %d\n", p_decParams->BG, p_decParams->R, p_decParams->Z);
    AssertFatal(false, "Format cuda not support, only support BG = 1, Zc = 384 and R = 13, 23 right now\n");
    return 0;
  }
  uint32_t numLLR;

#if CUDAMALLOC
  if (!p_lutCreated) {
    // P_lut = p_lut;
    printf("Start to create p_lut_dev\n");

    Err = cudaMallocManaged((void**)&p_lut_dev, sizeof(t_nrLDPC_lut), cudaMemAttachGlobal);
    if (Err != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed for lut: %s\n", cudaGetErrorString(Err));
      exit(EXIT_FAILURE);
    }
    // p_lut_dev = &lut;
    numLLR = nrLDPC_init(p_decParams, p_lut_dev);
    printf("p_lut_dev Created\n");
    // check p_lut
    // check_lut_pointers(p_lut);

    p_lutCreated = true;
  } else {
    numLLR = nrLDPC_init(p_decParams, p_lut_dev);
  }
  // Launch LDPC decoder core for one segment
  int n_segments = p_decParams->n_segments;
  int numIter = nrLDPC_decoder_core(p_llr, p_out, n_segments, numLLR, p_lut_dev, p_decParams, p_profiler, ab);
#else
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;

  // Initialize decoder core(s) with correct LUTs
  numLLR = nrLDPC_init(p_decParams, p_lut);
  // Launch LDPC decoder core for one segment
  int n_segments = p_decParams->n_segments;
  int numIter = nrLDPC_decoder_core(p_llr, p_out, n_segments, numLLR, p_lut, p_decParams, p_profiler, ab);

#endif

  // printf("6.1: It works here\n");
  if (numIter >= p_decParams->numMaxIter) {
    LOG_D(PHY, "set abort: %d, %d\n", numIter, p_decParams->numMaxIter);
    set_abort(ab, true);
  }
  // printf("6.2: It works here\n");
  return numIter;
}

/**
   \brief PerformsnrLDPC decoding of one code block
   \param p_llr Input LLRs
   \param p_out Output vector
   \param numLLR Number of LLRs
   \param p_lut Pointer to decoder LUTs
   \param p_decParamsnrLDPC decoder parameters
   \param p_profilernrLDPC profiler statistics
*/

static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           uint32_t numLLR,
                                           t_nrLDPC_lut* p_lut,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab)
{

#if CUDAMALLOC
  Err = cudaMemcpy(d_temp_in, p_llr, n_segments * 68 * 384 * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (Err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy d_temp_in failed: %s\n", cudaGetErrorString(Err));
    return -1;
  }
  Err = cudaMemset(d_temp_out, 0, n_segments * 8448 * sizeof(uint8_t));
  if (Err != cudaSuccess) {
    fprintf(stderr, "cudaMemset d_temp_out failed: %s\n", cudaGetErrorString(Err));
    return -1;
  }
#else
  memcpy(temp_in, p_llr, n_segments * 68 * 384);
  memset(temp_out, 0, n_segments * 8448);
#endif
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter; // To match the actual iterations
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;


  if (!SegmentPacked) {
    int segPerPack = 0;
    switch (R) {
      case 13:
        segPerPack = 30; // It's quite free here, GPU can handle this
        threadSize.NumThreads = 384;//maximum 1024
        threadSize.NumBlocks = (num_TotalThreads_BG1_R13 + threadSize.NumThreads - 1) / threadSize.NumThreads;
        break; // GH200 has 132 SMs, 264 blocks available, one R13 segment needs 30 blocks,
               // so maximent it can run 264/30 = 8 segments at one time
      case 23:
        segPerPack = 18;
        threadSize.NumThreads = 1024;//maximum 1024
        threadSize.NumBlocks = (num_TotalThreads_BG1_R23 + threadSize.NumThreads - 1) / threadSize.NumThreads;
        break; // For R23, it's 264/14 = 18

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
      /*
          printf("Pack %d -> startSeg=%d, nSeg=%d\n",
                 segmentPacks[p].packIdx,
                 segmentPacks[p].startSeg,
                 segmentPacks[p].nSeg);
      */
    }
    SegmentPacked = true;
  }
#if CUDAMALLOC
for (int s = 0; s < n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/; s++) {
    d_iter_ptr_array[s] = 0;
    d_PC_Flag_array[s] = 1;
  }


  for (int SegPackIdx = 0; SegPackIdx < NumSegPacks; SegPackIdx++) {
int PackShiftIdx = segmentPacks[SegPackIdx].startSeg;

    int8_t* perpack_llr = d_temp_in + PackShiftIdx * 68 * 384;
    int8_t* perpack_cnProcBuf = d_cnProcBuf + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* perpack_cnProcBufRes = d_cnProcBufRes + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;    
    int8_t* perpack_bnProcBuf = d_bnProcBuf + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_bnProcBufRes = d_bnProcBufRes + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_llrProcBuf = d_llrProcBuf + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrRes = d_llrRes + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrOut = d_llrOut + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_out = d_temp_out + PackShiftIdx * 8448; // use temp_out rather than p_out
    int8_t* perpack_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? perpack_out : perpack_llrOut;

    //  Call scheduler for this segment and stream
    //  Launch decoder on stream
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut,
                                           perpack_out,
                                           numLLR,
                                           perpack_llr,
                                           perpack_cnProcBuf,
                                           perpack_cnProcBufRes,
                                           perpack_bnProcBuf,
                                           perpack_bnProcBufRes,
                                           perpack_llrRes,
                                           perpack_llrProcBuf,
                                           perpack_llrOut,
                                           perpack_p_llrOut,
                                           Z,
                                           BG,
                                           R,
                                           numMaxIter,
                                           outMode,
                                           decoderStreams,
                                           SegPackIdx, // Index for the whole pack
                                           decoderDoneEvents,
                                           &iter_ptr_array[PackShiftIdx],
                                           &PC_Flag_array[PackShiftIdx]); // stream index passed in
  }
#else
  for (int s = 0; s < n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/; s++) {
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
  }

for (int SegPackIdx = 0; SegPackIdx < NumSegPacks; SegPackIdx++) {

    int PackShiftIdx = segmentPacks[SegPackIdx].startSeg;

    int8_t* perpack_llr = temp_in + PackShiftIdx * 68 * 384;
    int8_t* perpack_cnProcBuf = cnProcBuf + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* perpack_cnProcBufRes = cnProcBufRes + PackShiftIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* perpack_bnProcBuf = bnProcBuf + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_bnProcBufRes = bnProcBufRes + PackShiftIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* perpack_llrProcBuf = llrProcBuf + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrRes = llrRes + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_llrOut = llrOut + PackShiftIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* perpack_out = temp_out + PackShiftIdx * 8448; // use temp_out rather than p_out
    int8_t* perpack_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? perpack_out : perpack_llrOut;

    //  Call scheduler for this segment and stream
    //  Launch decoder on stream
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut,
                                           perpack_out,
                                           numLLR,
                                           perpack_llr,
                                           perpack_cnProcBuf,
                                           perpack_cnProcBufRes,
                                           perpack_bnProcBuf,
                                           perpack_bnProcBufRes,
                                           perpack_llrRes,
                                           perpack_llrProcBuf,
                                           perpack_llrOut,
                                           perpack_p_llrOut,
                                           Z,
                                           BG,
                                           R,
                                           numMaxIter,
                                           outMode,
                                           decoderStreams,
                                           SegPackIdx, // Index for the whole pack
                                           decoderDoneEvents,
                                           &iter_ptr_array[PackShiftIdx],
                                           &PC_Flag_array[PackShiftIdx]); // stream index passed in
  }
#endif

    for (int s = 0; s < NumSegPacks; ++s) {
      // printf("Synchronizing segment %d \n",s);
      cudaEventSynchronize(decoderDoneEvents[s]); // stop until segment decode
    }
    cudaDeviceSynchronize();

    // cudaDeviceSynchronize();
    // printf("p_out %p, temp_out %p\n",p_out,temp_out);
#if CUDAMALLOC
    cudaMemcpy(p_out, d_temp_out, n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/ * 8448, cudaMemcpyDeviceToHost);
#else
    memcpy(p_out, temp_out, n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/ * 8448);
#endif
     //dumpASS(p_out, "Dump_Output_Stream_GH.txt");

    return numMaxIter;
  }

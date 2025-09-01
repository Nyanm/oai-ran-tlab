

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
#define UNROLL_CN_PROC 1
#define UNROLL_BN_PROC 1
#define UNROLL_BN_PROC_PC 1
#define UNROLL_BN2CN_PROC 1
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
/*----------------------------------------------------------------------
|                  cn processing files -->AVX512
/----------------------------------------------------------------------*/

// BG1-------------------------------------------------------------------
#if defined(__AVX512BW__)

#include "cnProc_avx512/nrLDPC_cnProc_BG1_R13_AVX512.h"
#include "cnProc_avx512/nrLDPC_cnProc_BG1_R23_AVX512.h"
#include "cnProc_avx512/nrLDPC_cnProc_BG1_R89_AVX512.h"
// BG2-------------------------------------------------------------------
#include "cnProc_avx512/nrLDPC_cnProc_BG2_R15_AVX512.h"
#include "cnProc_avx512/nrLDPC_cnProc_BG2_R13_AVX512.h"
#include "cnProc_avx512/nrLDPC_cnProc_BG2_R23_AVX512.h"

#elif defined(__AVX2__)

/*----------------------------------------------------------------------
|                  cn Processing files -->AVX2
/----------------------------------------------------------------------*/

// BG1------------------------------------------------------------------
#include "cnProc/nrLDPC_cnProc_BG1_R13_AVX2.h"
#include "cnProc/nrLDPC_cnProc_BG1_R23_AVX2.h"
#include "cnProc/nrLDPC_cnProc_BG1_R89_AVX2.h"
// BG2 --------------------------------------------------------------------
#include "cnProc/nrLDPC_cnProc_BG2_R15_AVX2.h"
#include "cnProc/nrLDPC_cnProc_BG2_R13_AVX2.h"
#include "cnProc/nrLDPC_cnProc_BG2_R23_AVX2.h"

#else

// BG1------------------------------------------------------------------
#include "cnProc128/nrLDPC_cnProc_BG1_R13_128.h"
#include "cnProc128/nrLDPC_cnProc_BG1_R23_128.h"
#include "cnProc128/nrLDPC_cnProc_BG1_R89_128.h"
// BG2 --------------------------------------------------------------------
#include "cnProc128/nrLDPC_cnProc_BG2_R15_128.h"
#include "cnProc128/nrLDPC_cnProc_BG2_R13_128.h"
#include "cnProc128/nrLDPC_cnProc_BG2_R23_128.h"
#endif

/*----------------------------------------------------------------------
|                 bn Processing files -->AVX2
/----------------------------------------------------------------------*/

// bnProcPc-------------------------------------------------------------
#ifdef __AVX2__
// BG1------------------------------------------------------------------
#include "bnProcPc/nrLDPC_bnProcPc_BG1_R13_AVX2.h"
#include "bnProcPc/nrLDPC_bnProcPc_BG1_R23_AVX2.h"
#include "bnProcPc/nrLDPC_bnProcPc_BG1_R89_AVX2.h"
// BG2 --------------------------------------------------------------------
#include "bnProcPc/nrLDPC_bnProcPc_BG2_R15_AVX2.h"
#include "bnProcPc/nrLDPC_bnProcPc_BG2_R13_AVX2.h"
#include "bnProcPc/nrLDPC_bnProcPc_BG2_R23_AVX2.h"
#else
#include "bnProcPc128/nrLDPC_bnProcPc_BG1_R13_128.h"
#include "bnProcPc128/nrLDPC_bnProcPc_BG1_R23_128.h"
#include "bnProcPc128/nrLDPC_bnProcPc_BG1_R89_128.h"
#include "bnProcPc128/nrLDPC_bnProcPc_BG2_R15_128.h"
#include "bnProcPc128/nrLDPC_bnProcPc_BG2_R13_128.h"
#include "bnProcPc128/nrLDPC_bnProcPc_BG2_R23_128.h"
#endif

// bnProc----------------------------------------------------------------

#if defined(__AVX512BW__)
// BG1-------------------------------------------------------------------
#include "bnProc_avx512/nrLDPC_bnProc_BG1_R13_AVX512.h"
#include "bnProc_avx512/nrLDPC_bnProc_BG1_R23_AVX512.h"
#include "bnProc_avx512/nrLDPC_bnProc_BG1_R89_AVX512.h"
// BG2 --------------------------------------------------------------------
#include "bnProc_avx512/nrLDPC_bnProc_BG2_R15_AVX512.h"
#include "bnProc_avx512/nrLDPC_bnProc_BG2_R13_AVX512.h"
#include "bnProc_avx512/nrLDPC_bnProc_BG2_R23_AVX512.h"

#elif defined(__AVX2__)
#include "bnProc/nrLDPC_bnProc_BG1_R13_AVX2.h"
#include "bnProc/nrLDPC_bnProc_BG1_R23_AVX2.h"
#include "bnProc/nrLDPC_bnProc_BG1_R89_AVX2.h"
// BG2 --------------------------------------------------------------------
#include "bnProc/nrLDPC_bnProc_BG2_R15_AVX2.h"
#include "bnProc/nrLDPC_bnProc_BG2_R13_AVX2.h"
#include "bnProc/nrLDPC_bnProc_BG2_R23_AVX2.h"
#else
#include "bnProc128/nrLDPC_bnProc_BG1_R13_128.h"
#include "bnProc128/nrLDPC_bnProc_BG1_R23_128.h"
#include "bnProc128/nrLDPC_bnProc_BG1_R89_128.h"
// BG2 --------------------------------------------------------------------
#include "bnProc128/nrLDPC_bnProc_BG2_R15_128.h"
#include "bnProc128/nrLDPC_bnProc_BG2_R13_128.h"
#include "bnProc128/nrLDPC_bnProc_BG2_R23_128.h"
#endif

// #define NR_LDPC_PROFILER_DETAIL(a) a
#define NR_LDPC_PROFILER_DETAIL(a)

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
static int currentStreamCount = 0;
static int8_t iter_ptr_array[MAX_NUM_DLSCH_SEGMENTS_DL];
static  int PC_Flag_array[MAX_NUM_DLSCH_SEGMENTS_DL];
static  int8_t cnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
static  int8_t cnProcBufRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
static  int8_t bnProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
static  int8_t bnProcBufRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
static  int8_t llrRes[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static  int8_t llrProcBuf[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
static  int8_t llrOut[MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};



extern void nrLDPC_cnProc_BG1_cuda(const t_nrLDPC_lut* p_lut,
                                   int8_t* cnProcBuf,
                                   int8_t* cnProcBufRes,
                                   int8_t* bnProcBuf,
                                   uint16_t Z);

extern void nrLDPC_bnProc_BG1_cuda(const t_nrLDPC_lut* p_lut,
                                   int8_t* bnProcBuf,
                                   int8_t* bnProcBufRes,
                                   int8_t* llrProcBuf,
                                   int8_t* llrRes,
                                   uint16_t Z);

extern void nrLDPC_BnToCnPC_BG1_cuda(const t_nrLDPC_lut* p_lut,
                                     int8_t* bnProcBufRes,
                                     int8_t* cnProcBuf,
                                     int8_t* cnProcBufRes,
                                     int8_t* bnProcBuf,
                                     uint16_t Z,
                                     int* PC_Flag);

extern void nrLDPC_decoder_scheduler_BG1_cuda_core(const t_nrLDPC_lut* p_lut,
                                                   int8_t* p_out,
                                                   uint32_t numLLR,
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

//-------------------------Debug Function-----------------------
void dump_cnProcBufRes_to_file(const int8_t* cnProcBufRes, const char* filename)
{
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < NR_LDPC_SIZE_CN_PROC_BUF; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}

void dumpASS(int8_t* cnProcBufRes, const char* filename)
{
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL * 68 * 384; i++) {
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

void init_decoder_graphs() {
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

int32_t LDPCinit_cuda()
{
  printf("CUDA LDPC decoder initiating\n");
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

int32_t LDPCinit()
{
  printf("initialling\n");
  LDPCinit_cuda();
  return 0;
}

int32_t LDPCshutdown_cuda()
{
  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
    if (streamsCreated) {
      cudaEventDestroy(decoderDoneEvents[s]);
      cudaStreamDestroy(decoderStreams[s]);
    }
  }

  free_graphs();

  streamsCreated = false;
  //d_mem_exist = false;

  return 0;
}

int32_t LDPCshutdown()
{

  LDPCshutdown_cuda();
  return 0;
}

int32_t LDPCdecoder(t_nrLDPC_dec_params* p_decParams,
                    uint8_t harq_pid,
                    uint8_t ulsch_id,
                    uint8_t C,
                    int8_t* p_llr,
                    int8_t* p_out,
                    t_nrLDPC_time_stats* p_profiler,
                    decode_abort_t* ab)
{
  uint32_t numLLR;
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;

  // Initialize decoder core(s) with correct LUTs
  numLLR = nrLDPC_init(p_decParams, p_lut);

  // Launch LDPC decoder core for one segment
  int n_segments = p_decParams->n_segments;
  int numIter = nrLDPC_decoder_core(p_llr, p_out, n_segments, numLLR, p_lut, p_decParams, p_profiler, ab);
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
  // printf("n_segments = %d\n", n_segments);
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;
  int LastTrial = p_decParams->LastTrial;
  // printf("Kprime = %d\n", Kprime);
  //  int8_t* cnProcBuf=  cnProcBuf;
  //  int8_t* cnProcBufRes= cnProcBufRes;
  // printf("1: It works here\n");


  // printf("2: It works here\n");
  //  Minimum number of iterations is 1
  //  0 iterations means hard-decision on input LLRs
  //  Initialize with parity check fail != 0
  // printf("3: It works here\n");
  //  Initialization
  //cudaStream_t streams[MAX_NUM_DLSCH_SEGMENTS];
  //cudaEvent_t done[MAX_NUM_DLSCH_SEGMENTS]; // MAX_NUM_SEGMENTS = stream num

  for (int s = 0; s < n_segments; s++) {
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
  }
  // printf("3.1: It works here\n");
  if (!streamsCreated) {
  for (int s = 0; s < n_segments; ++s) {
    cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
    cudaEventCreate(&decoderDoneEvents[s]);
  }
  streamsCreated = true;
  currentStreamCount = n_segments;
}

  // printf("3.2: It works here\n");
  for (int CudaStreamIdx = 0; CudaStreamIdx < n_segments; CudaStreamIdx++) {
    int8_t* pp_llr = p_llr + CudaStreamIdx * 68 * 384;
    int8_t* pp_out = p_out + CudaStreamIdx * Kprime;
    // printf("Stream %d: pp_out = %p\n", CudaStreamIdx, pp_out);
    int8_t* pp_cnProcBuf = cnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = cnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = bnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = bnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrOut = llrOut + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    // printf("4: It works here\n");
    //  LLR preprocessing
    //NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    //NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
    //NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
    if (BG == 1)
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
    //NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));
    // Call scheduler for this segment and stream
    int8_t* pp_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : pp_llrOut;
    // printf("5: It works here\n");
    //  Launch decoder on stream s
    
    //cudaEventCreate(&decoderDoneEvents[CudaStreamIdx]);
    //printf("Launching segment %d \n",CudaStreamIdx);
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut,
                                           pp_out,
                                           numLLR,
                                           pp_cnProcBuf,
                                           pp_cnProcBufRes,
                                           pp_bnProcBuf,
                                           pp_bnProcBufRes,
                                           pp_llrRes,
                                           pp_llrProcBuf,
                                           pp_llrOut,
                                           pp_p_llrOut,
                                           Z,
                                           BG,
                                           R,
                                           numMaxIter,
                                           outMode,
                                           decoderStreams,
                                           CudaStreamIdx,
                                           decoderDoneEvents,
                                           &iter_ptr_array[CudaStreamIdx],
                                           &PC_Flag_array[CudaStreamIdx]); // stream index passed in
  
  }
for (int s = 0; s < n_segments; ++s) {
 // printf("Synchronizing segment %d \n",s);
    cudaEventSynchronize(decoderDoneEvents[s]);  // stop until segment decode
}
cudaDeviceSynchronize();

if(LastTrial == 1){
  //printf("Now is the last trial\n");
  for (int s = 0; s < n_segments; s++) {
    cudaEventDestroy(decoderDoneEvents[s]);
    cudaStreamSynchronize(decoderStreams[s]);
    cudaStreamDestroy(decoderStreams[s]);
    streamsCreated = false;
  }
}
//cudaDeviceSynchronize();
  // dumpASS(p_out, "Dump_Output_Stream.txt");
  // printf("6: It works here\n");

  return numMaxIter;
}

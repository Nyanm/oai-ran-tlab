

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

  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS * 68 * 384; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}
//--------------------------------------------------------------
#ifdef PARALLEL_STREAM
static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           int n_segments,
                                           uint32_t numLLR,
                                           t_nrLDPC_lut* p_lut,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab);

int32_t LDPCinit()
{
  return 0;
}

int32_t LDPCshutdown()
{
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
  // printf("Kprime = %d\n", Kprime);
  //  int8_t* cnProcBuf=  cnProcBuf;
  //  int8_t* cnProcBufRes= cnProcBufRes;
  // printf("1: It works here\n");

  int8_t cnProcBuf[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t cnProcBufRes[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t bnProcBuf[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t bnProcBufRes[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t llrRes[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
  int8_t llrProcBuf[MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
  // printf("2: It works here\n");
  //  Minimum number of iterations is 1
  //  0 iterations means hard-decision on input LLRs
  //  Initialize with parity check fail != 0
  // printf("3: It works here\n");
  //  Initialization
  cudaStream_t streams[MAX_NUM_DLSCH_SEGMENTS];
  cudaEvent_t done[MAX_NUM_DLSCH_SEGMENTS]; // MAX_NUM_SEGMENTS = stream数量
  int8_t iter_ptr_array[MAX_NUM_DLSCH_SEGMENTS];
  int PC_Flag_array[MAX_NUM_DLSCH_SEGMENTS];
  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS; s++) {
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
  }
  // printf("3.1: It works here\n");
  for (int s = 0; s < n_segments; ++s) {
    cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
     cudaEventCreate(&done[s]);
    // printf("stream: It works here\n");
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
    // printf("4: It works here\n");
    //  LLR preprocessing
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
    if (BG == 1)
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));
    // Call scheduler for this segment and stream
    int8_t pp_llrOut[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
    int8_t* pp_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : pp_llrOut;
    // printf("5: It works here\n");
    //  Launch decoder on stream s
    
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
                                           streams,
                                           CudaStreamIdx,
                                           &iter_ptr_array[CudaStreamIdx],
                                           &PC_Flag_array[CudaStreamIdx]); // stream index passed in
  
    
    
                                           //cudaEventRecord(done[CudaStreamIdx], streams[CudaStreamIdx]);
    // printf("5: It works here\n");
  }

  // Wait for all streams
 // for (int s = 0; s < n_segments; s++) {
 //   cudaEventSynchronize(done[s]); // 等待stream[i]完成
 //   // 可安全访问对应的解码输出结果 p_llrOut[i]
//}

  for (int s = 0; s < n_segments; s++) {
    cudaStreamSynchronize(streams[s]);
    cudaStreamDestroy(streams[s]);
  }
//cudaDeviceSynchronize();
  // dumpASS(p_out, "Dump_Output_Stream.txt");
  // printf("6: It works here\n");

  return numMaxIter;
}

#else
static inline uint32_t nrLDPC_decoder_core(int8_t* p_llr,
                                           int8_t* p_out,
                                           uint32_t numLLR,
                                           t_nrLDPC_lut* p_lut,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab);

int32_t LDPCinit()
{
  return 0;
}

int32_t LDPCshutdown()
{
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
  int numIter = nrLDPC_decoder_core(p_llr, p_out, numLLR, p_lut, p_decParams, p_profiler, ab);
  if (numIter >= p_decParams->numMaxIter) {
    LOG_D(PHY, "set abort: %d, %d\n", numIter, p_decParams->numMaxIter);
    set_abort(ab, true);
  }
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
                                           uint32_t numLLR,
                                           t_nrLDPC_lut* p_lut,
                                           t_nrLDPC_dec_params* p_decParams,
                                           t_nrLDPC_time_stats* p_profiler,
                                           decode_abort_t* ab)
{
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  // int8_t* cnProcBuf=  cnProcBuf;
  // int8_t* cnProcBufRes= cnProcBufRes;

  int8_t cnProcBuf[NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t cnProcBufRes[NR_LDPC_SIZE_CN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t bnProcBuf[NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t bnProcBufRes[NR_LDPC_SIZE_BN_PROC_BUF] __attribute__((aligned(64))) = {0};
  int8_t llrRes[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
  int8_t llrProcBuf[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
  // Minimum number of iterations is 1
  // 0 iterations means hard-decision on input LLRs
  // Initialize with parity check fail != 0
//printf("cnProcBuf address: %p\n",cnProcBuf);
  // Initialization
  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
  nrLDPC_llr2llrProcBuf(p_lut, p_llr, llrProcBuf, Z, BG);
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_LLR_PROC);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_LLR_PROC, llrProcBuf);
#endif

  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
  if (BG == 1)
    nrLDPC_llr2CnProcBuf_BG1(p_lut, p_llr, cnProcBuf, Z);
  else
    nrLDPC_llr2CnProcBuf_BG2(p_lut, p_llr, cnProcBuf, Z);
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_CN_PROC);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_CN_PROC, cnProcBuf);
#endif
  // First iteration
  // CN processing
  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->cnProc));
#ifdef USE_CUDA
  // printf("We're not here when CUDA stream enabled ^ ^ (but nothing here yet)\n");
  //       printf("\nHere we use CUDA\n");
  //  dump_cnProcBufRes_to_file(cnProcBuf, "First_cnProcBuf_dump_cuda.txt");
  nrLDPC_cnProc_BG1_cuda(p_lut, cnProcBuf, cnProcBufRes, bnProcBuf, Z);
  // dump_cnProcBufRes_to_file(bnProcBuf, "First_bnProcBuf_dump_cuda.txt");

  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cnProc));
  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->cn2bnProcBuf));
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cn2bnProcBuf));
  // dump_cnProcBufRes_to_file(cnProcBufRes, "First_cnProcBufRes_dump_cuda.txt");

#else
  if (BG == 1) {
    // printf("\ncheck point 1\n");

#ifndef UNROLL_CN_PROC
    // printf("\nCheck point 2\n");
    nrLDPC_cnProc_BG1(p_lut, cnProcBuf, cnProcBufRes, Z);
#else
    // printf("\nCheckpoint 3\n ");
    switch (R) {
      case 13: {
#if defined(__AVX512BW__)
        // printf("\nCheckpoint 4\n ");
        nrLDPC_cnProc_BG1_R13_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        // printf("\nCheckpoint 5\n ");
        nrLDPC_cnProc_BG1_R13_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        // printf("\nCheckpoint 6\n ");
        // dump_cnProcBufRes_to_file(cnProcBuf, "First_cnProcBuf_dump_128.txt");
        nrLDPC_cnProc_BG1_R13_128(cnProcBuf, cnProcBufRes, Z);
        // dump_cnProcBufRes_to_file(cnProcBufRes, "First_cnProcBufRes_dump_128.txt");
#endif
        break;
      }

      case 23: {
#if defined(__AVX512BW__)
        nrLDPC_cnProc_BG1_R23_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        nrLDPC_cnProc_BG1_R23_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        nrLDPC_cnProc_BG1_R23_128(cnProcBuf, cnProcBufRes, Z);
#endif
        break;
      }

      case 89: {
#if defined(__AVX512BW__)
        nrLDPC_cnProc_BG1_R89_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        nrLDPC_cnProc_BG1_R89_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        nrLDPC_cnProc_BG1_R89_128(cnProcBuf, cnProcBufRes, Z);
#endif
        break;
      }
    }
#endif

  } else {
#ifndef UNROLL_CN_PROC
    nrLDPC_cnProc_BG2(p_lut, cnProcBuf, cnProcBufRes, Z);
#else
    switch (R) {
      case 15: {
#if defined(__AVX512BW__)
        nrLDPC_cnProc_BG2_R15_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        nrLDPC_cnProc_BG2_R15_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        nrLDPC_cnProc_BG2_R15_128(cnProcBuf, cnProcBufRes, Z);
#endif
        break;
      }
      case 13: {
#if defined(__AVX512BW__)
        nrLDPC_cnProc_BG2_R13_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        nrLDPC_cnProc_BG2_R13_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        nrLDPC_cnProc_BG2_R13_128(cnProcBuf, cnProcBufRes, Z);
#endif
        break;
      }
      case 23: {
#if defined(__AVX512BW__)
        nrLDPC_cnProc_BG2_R23_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
        nrLDPC_cnProc_BG2_R23_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
        nrLDPC_cnProc_BG2_R23_128(cnProcBuf, cnProcBufRes, Z);
#endif
        break;
      }
    }
#endif
  }
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cnProc));

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_CN_PROC_RES);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_CN_PROC_RES, cnProcBufRes);
#endif

  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->cn2bnProcBuf));
  if (BG == 1) {
    nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
    // dump_cnProcBufRes_to_file(bnProcBuf, "First_bnProcBuf_dump_128.txt");
  } else
    nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cn2bnProcBuf));

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_BN_PROC);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_BN_PROC, bnProcBuf);
#endif
#endif
  // BN processing
#ifdef USE_CUDA

  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProcPc));
  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProcPc));

  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProc));
  // printf("I'm here:1\n");
  nrLDPC_bnProc_BG1_cuda(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
  // dump_cnProcBufRes_to_file(bnProcBufRes, "First_bnProcBufRes_dump_cuda.txt");
  // dump_cnProcBufRes_to_file(llrRes, "First_llrRes_dump_cuda.txt");

  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProc));

#ifdef NR_LDPC_PROFILER_DETAIL
  start_meas(&p_profiler->bn2cnProcBuf);
#endif
  // printf("I'm here:2\n");
  int Temp = 0;
  int* Not_use_Flag = &Temp;
  nrLDPC_BnToCnPC_BG1_cuda(p_lut, bnProcBufRes, cnProcBuf, cnProcBufRes, bnProcBuf, Z, Not_use_Flag);
  // dump_cnProcBufRes_to_file(cnProcBuf, "First_cnProcBuf_New_dump_cuda.txt");
  // dump_cnProcBufRes_to_file(cnProcBufRes, "First_cnProcBufRes_dump_cuda.txt");
  // dump_cnProcBufRes_to_file(cnProcBuf, "First_Packed_cnProcBuf_dump_cuda.txt");

#ifdef NR_LDPC_PROFILER_DETAIL
  stop_meas(&p_profiler->bn2cnProcBuf);
#endif

#else
  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProcPc));

#ifndef UNROLL_BN_PROC_PC
  nrLDPC_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
#else
  if (BG == 1) {
    switch (R) {
      case 13: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG1_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG1_R13_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }
      case 23: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG1_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG1_R23_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }
      case 89: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG1_R89_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG1_R89_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }
    }
  } else {
    switch (R) {
      case 15: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG2_R15_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG2_R15_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }
      case 13: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG2_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG2_R13_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }

      case 23: {
#ifdef __AVX2__
        nrLDPC_bnProcPc_BG2_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
        nrLDPC_bnProcPc_BG2_R23_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
        break;
      }
    }
  }
#endif

  NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProcPc));

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_LLR_RES);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_LLR_RES, llrRes);
#endif

  NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProc));

  if (BG == 1) {
#ifndef UNROLL_BN_PROC
    nrLDPC_bnProc(p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);
#else
    switch (R) {
      case 13: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG1_R13_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG1_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG1_R13_128(bnProcBuf, bnProcBufRes, llrRes, Z);
        // dump_cnProcBufRes_to_file(bnProcBufRes, "First_bnProcBufRes_dump_128.txt");
        // dump_cnProcBufRes_to_file(llrRes, "First_llrRes_dump_128.txt");
#endif
        break;
      }
      case 23: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG1_R23_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG1_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG1_R23_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
        break;
      }
      case 89: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG1_R89_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG1_R89_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG1_R89_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
        break;
      }
    }
#endif
  } else {
#ifndef UNROLL_BN2CN_PROC
    nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);
#else
    switch (R) {
      case 15: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG2_R15_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG2_R15_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG2_R15_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
        break;
      }
      case 13: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG2_R13_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG2_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG2_R13_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
        break;
      }

      case 23: {
#if defined(__AVX512BW__)
        nrLDPC_bnProc_BG2_R23_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
        nrLDPC_bnProc_BG2_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
        nrLDPC_bnProc_BG2_R23_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
        break;
      }
    }
#endif
  }

#ifdef NR_LDPC_PROFILER_DETAIL
  stop_meas(&p_profiler->bnProc);
#endif

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_initBuffer2File(nrLDPC_buffers_BN_PROC_RES);
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_BN_PROC_RES, bnProcBufRes);
#endif

  // BN results to CN processing buffer
#ifdef NR_LDPC_PROFILER_DETAIL
  start_meas(&p_profiler->bn2cnProcBuf);
#endif
  if (BG == 1) {
    nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
    // dump_cnProcBufRes_to_file(cnProcBuf, "First_cnProcBuf_New_dump_128.txt");
    // dump_cnProcBufRes_to_file(cnProcBufRes, "First_cnProcBufRes_dump_128.txt");
  } else
    nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);
#ifdef NR_LDPC_PROFILER_DETAIL
  stop_meas(&p_profiler->bn2cnProcBuf);
#endif

#ifdef NR_LDPC_DEBUG_MODE
  nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_CN_PROC, cnProcBuf);
#endif
// cuda ends here
#endif

  // Parity Check not necessary here since it will fail
  // because first 2 cols/BNs in BG are punctured and cannot be
  // estimated after only one iteration

  // First iteration finished
  uint32_t numIter = 0;
  int32_t pcRes = 1; // pcRes is 0 if the ldpc decoder is succesful
  while ((numIter < numMaxIter) && (pcRes != 0)) {
    if (check_abort(ab)) {
      numIter = numMaxIter;
      break;
    }
    // CN processing
#ifdef NR_LDPC_PROFILER_DETAIL
    start_meas(&p_profiler->cnProc);
#endif
#ifdef USE_CUDA
    // dump_cnProcBufRes_to_file(cnProcBufRes, "cnProcBufRes_last_dump.txt");
    nrLDPC_cnProc_BG1_cuda(p_lut, cnProcBuf, cnProcBufRes, bnProcBuf, Z);
    if (numIter == 0) {
      // dump_cnProcBufRes_to_file(bnProcBuf, "Second_bnProcBuf_dump_cuda.txt");
    }
    // dump_cnProcBufRes_to_file(bnProcBuf, "First_bnProcBuf_dump_cuda.txt");
#ifdef NR_LDPC_PROFILER_DETAIL
    stop_meas(&p_profiler->cnProc);
    start_meas(&p_profiler->cn2bnProcBuf);
    stop_meas(&p_profiler->cn2bnProcBuf);
#endif
    // dump_cnProcBufRes_to_file(cnProcBufRes, "cnProcBufRes_dump_cuda.txt");
    // dump_cnProcBufRes_to_file(cnProcBuf, "cnProcBuf_last_dump.txt");

#else

    if (BG == 1) {

#ifndef UNROLL_CN_PROC
      nrLDPC_cnProc_BG1(p_lut, cnProcBuf, cnProcBufRes, Z);
#else
      switch (R) {
        case 13: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG1_R13_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG1_R13_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG1_R13_128(cnProcBuf, cnProcBufRes, Z);
          // dump_cnProcBufRes_to_file(cnProcBufRes, "cnProcBufRes_dump_128.txt");

#endif
          break;
        }
        case 23: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG1_R23_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG1_R23_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG1_R23_128(cnProcBuf, cnProcBufRes, Z);
#endif
          break;
        }
        case 89: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG1_R89_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG1_R89_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG1_R89_128(cnProcBuf, cnProcBufRes, Z);
#endif
          break;
        }
      }
#endif

    } else {
#ifndef UNROLL_CN_PROC
      nrLDPC_cnProc_BG2(p_lut, cnProcBuf, cnProcBufRes, Z);
#else
      switch (R) {
        case 15: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG2_R15_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG2_R15_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG2_R15_128(cnProcBuf, cnProcBufRes, Z);
#endif
          break;
        }
        case 13: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG2_R13_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG2_R13_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG2_R13_128(cnProcBuf, cnProcBufRes, Z);
#endif
          break;
        }
        case 23: {
#if defined(__AVX512BW__)
          nrLDPC_cnProc_BG2_R23_AVX512(cnProcBuf, cnProcBufRes, Z);
#elif defined(__AVX2__)
          nrLDPC_cnProc_BG2_R23_AVX2(cnProcBuf, cnProcBufRes, Z);
#else
          nrLDPC_cnProc_BG2_R23_128(cnProcBuf, cnProcBufRes, Z);
#endif
          break;
        }
      }
#endif
    }
#ifdef NR_LDPC_PROFILER_DETAIL
    stop_meas(&p_profiler->cnProc);
#endif

#ifdef NR_LDPC_DEBUG_MODE
    nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_CN_PROC_RES, cnProcBufRes);
#endif

    // Send CN results back to BNs
#ifdef NR_LDPC_PROFILER_DETAIL
    start_meas(&p_profiler->cn2bnProcBuf);
#endif
    if (BG == 1)
      nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
    // dump_cnProcBufRes_to_file(cnProcBufRes, "cnProcBufRes_dump_inBn.txt");
    // dump_cnProcBufRes_to_file(bnProcBuf, "have_a_look_Bn_buffer.txt");}// for debug
    else
      nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);
#ifdef NR_LDPC_PROFILER_DETAIL
    stop_meas(&p_profiler->cn2bnProcBuf);
#endif

#ifdef NR_LDPC_DEBUG_MODE
    nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_BN_PROC, bnProcBuf);
#endif
// cuda end
#endif
    // BN Processing

#ifdef USE_CUDA

    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProcPc));
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProcPc));

    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProc));

    nrLDPC_bnProc_BG1_cuda(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);

    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProc));

#ifdef NR_LDPC_PROFILER_DETAIL
    start_meas(&p_profiler->bn2cnProcBuf);
#endif
    int Flag_Value = 0;
    int* PC_Flag = &Flag_Value;
    nrLDPC_BnToCnPC_BG1_cuda(p_lut, bnProcBufRes, cnProcBuf, cnProcBufRes, bnProcBuf, Z, PC_Flag);

    // Parity Check
    if (!p_decParams->check_crc) {
      NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->cnProcPc));
      if (BG == 1) {
        pcRes &= *(int32_t*)PC_Flag;
        // printf("PC_Flag = %d\n",*PC_Flag);
      } else
        pcRes = nrLDPC_cnProcPc_BG2(p_lut, cnProcBuf, cnProcBufRes, Z);
      NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cnProcPc));
    } else {
      if (numIter > 0) {
        int8_t llrOut[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
        int8_t* p_llrOut = outMode == nrLDPC_outMode_LLRINT8 ? p_out : llrOut;
        nrLDPC_llrRes2llrOut(p_lut, p_llrOut, llrRes, Z, BG);
        if (outMode == nrLDPC_outMode_BIT)
          nrLDPC_llr2bitPacked(p_out, p_llrOut, numLLR);
        else // if (outMode == nrLDPC_outMode_BITINT8)
          nrLDPC_llr2bit(p_out, p_llrOut, numLLR);
        if (p_decParams->check_crc((uint8_t*)p_out, p_decParams->Kprime, p_decParams->crc_type)) {
          LOG_D(PHY, "Segment CRC OK, exiting LDPC decoder\n");
          break;
        }
      }
    }
    // dump_cnProcBufRes_to_file(cnProcBuf, "First_Packed_cnProcBuf_dump_cuda.txt");

#ifdef NR_LDPC_PROFILER_DETAIL
    stop_meas(&p_profiler->bn2cnProcBuf);
#endif

#else
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProcPc));

#ifndef UNROLL_BN_PROC_PC
    nrLDPC_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
#else
    if (BG == 1) {
      switch (R) {
        case 13: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG1_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG1_R13_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
        case 23: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG1_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG1_R23_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
        case 89: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG1_R89_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG1_R89_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
      }
    } else {
      switch (R) {
        case 15: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG2_R15_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG2_R15_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
        case 13: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG2_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG2_R13_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
        case 23: {
#ifdef __AVX2__
          nrLDPC_bnProcPc_BG2_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#else
          nrLDPC_bnProcPc_BG2_R23_128(bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, Z);
#endif
          break;
        }
      }
    }
#endif
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProcPc));

#ifdef NR_LDPC_DEBUG_MODE
    nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_LLR_RES, llrRes);
#endif

    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bnProc));
#ifndef UNROLL_BN_PROC
    nrLDPC_bnProc(p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);
#else
    if (BG == 1) {
      switch (R) {
        case 13: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG1_R13_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG1_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG1_R13_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          //                printf("\nThere's a cat in BG1 R13:\n");
          //		printf(" /\\_/\\\n");
          //              printf("( o.o )\n");
          //            printf(" > ^ <\n");
          break;
        }
        case 23: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG1_R23_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG1_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG1_R23_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          break;
        }
        case 89: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG1_R89_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG1_R89_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG1_R89_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          break;
        }
      }
    } else {
      switch (R) {
        case 15: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG2_R15_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG2_R15_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG2_R15_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          break;
        }
        case 13: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG2_R13_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG2_R13_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG2_R13_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          break;
        }
        case 23: {
#if defined(__AVX512BW__)
          nrLDPC_bnProc_BG2_R23_AVX512(bnProcBuf, bnProcBufRes, llrRes, Z);
#elif defined(__AVX2__)
          nrLDPC_bnProc_BG2_R23_AVX2(bnProcBuf, bnProcBufRes, llrRes, Z);
#else
          nrLDPC_bnProc_BG2_R23_128(bnProcBuf, bnProcBufRes, llrRes, Z);
#endif
          break;
        }
      }
    }
#endif

    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bnProc));

#ifdef NR_LDPC_DEBUG_MODE
    nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_BN_PROC_RES, bnProcBufRes);
#endif

    // BN results to CN processing buffer
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->bn2cnProcBuf));
    if (BG == 1) {
      nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
      // dump_cnProcBufRes_to_file(cnProcBuf, "First_Packed_cnProcBuf_dump_128.txt");
    } else
      nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->bn2cnProcBuf));

#ifdef NR_LDPC_DEBUG_MODE
    nrLDPC_debug_writeBuffer2File(nrLDPC_buffers_CN_PROC, cnProcBuf);
#endif

    // Parity Check
    if (!p_decParams->check_crc) {
      NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->cnProcPc));
      if (BG == 1)
        pcRes = nrLDPC_cnProcPc_BG1(p_lut, cnProcBuf, cnProcBufRes, Z);
      else
        pcRes = nrLDPC_cnProcPc_BG2(p_lut, cnProcBuf, cnProcBufRes, Z);
      NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->cnProcPc));
    } else {
      if (numIter > 0) {
        int8_t llrOut[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
        int8_t* p_llrOut = outMode == nrLDPC_outMode_LLRINT8 ? p_out : llrOut;
        nrLDPC_llrRes2llrOut(p_lut, p_llrOut, llrRes, Z, BG);
        if (outMode == nrLDPC_outMode_BIT)
          nrLDPC_llr2bitPacked(p_out, p_llrOut, numLLR);
        else // if (outMode == nrLDPC_outMode_BITINT8)
          nrLDPC_llr2bit(p_out, p_llrOut, numLLR);
        if (p_decParams->check_crc((uint8_t*)p_out, p_decParams->Kprime, p_decParams->crc_type)) {
          LOG_D(PHY, "Segment CRC OK, exiting LDPC decoder\n");
          break;
        }
      }
    }
// CUDA ends here
#endif

    // Increase iteration counter
    numIter++;
  }

  if (!p_decParams->check_crc) {
    int8_t llrOut[NR_LDPC_MAX_NUM_LLR] __attribute__((aligned(64))) = {0};
    int8_t* p_llrOut = outMode == nrLDPC_outMode_LLRINT8 ? p_out : llrOut;
    // Assign results from processing buffer to output
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llrRes2llrOut));
    nrLDPC_llrRes2llrOut(p_lut, p_llrOut, llrRes, Z, BG);

    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llrRes2llrOut));
    // Hard-decision
    NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2bit));
    if (outMode == nrLDPC_outMode_BIT) {
      nrLDPC_llr2bitPacked(p_out, p_llrOut, numLLR);
    } else { // if (outMode == nrLDPC_outMode_BITINT8)
      nrLDPC_llr2bit(p_out, p_llrOut, numLLR);
    }
    NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2bit));
  }
  //    #ifdef USE_CUDA
  //  printf("Using CUDA decoder\n");
  // #else
  // printf("Using CPU decoder\n");
  // #endif
  return numIter;
}
#endif
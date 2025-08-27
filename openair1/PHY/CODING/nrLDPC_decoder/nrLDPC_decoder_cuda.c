

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

#define STATIC_LUT 1

#if STATIC_LUT
static bool p_lutCreated = false;
static uint32_t numLLR;
static t_nrLDPC_lut lut;
static t_nrLDPC_lut* p_lut = &lut;
#endif

#if USE_CUDA
#include <cuda_runtime.h>
#endif


#define COPY_ARR_MEMBER(member, type, groups) do { \
    for (int i = 0; i < (groups); i++) { \
        type* tmp_dev; \
        if (h_lut->member[i].d != NULL && h_lut->member[i].dim1 > 0 && h_lut->member[i].dim2 > 0) { \
            size_t sz = h_lut->member[i].dim1 * h_lut->member[i].dim2 * sizeof(type); \
            err = cudaMalloc((void**)&tmp_dev, sz); \
            if (err != cudaSuccess) { \
                fprintf(stderr, "cudaMalloc failed for " #member "[%d]: %s\n", i, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
            cudaMemcpy(tmp_dev, h_lut->member[i].d, sz, cudaMemcpyHostToDevice); \
            /* updtae d_lut->member[i].d pointer */ \
            cudaMemcpy(&(d_lut->member[i].d), &tmp_dev, sizeof(type*), cudaMemcpyHostToDevice); \
            /* copy dim1 and dim2 */ \
            cudaMemcpy(&(d_lut->member[i].dim1), &(h_lut->member[i].dim1), sizeof(int), cudaMemcpyHostToDevice); \
            cudaMemcpy(&(d_lut->member[i].dim2), &(h_lut->member[i].dim2), sizeof(int), cudaMemcpyHostToDevice); \
        } \
    } \
} while(0)

#define COPY_POINTER_MEMBER(member, type, count) do { \
    type* tmp_dev; \
    printf("tmp_dev = %p\n", (void*)tmp_dev);\
    err = cudaMalloc((void**)&tmp_dev, (count) * sizeof(type)); \
    printf("malloc tmp_dev = %p\n", (void*)tmp_dev);\
    if (err != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc failed for " #member ": %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    printf("h_lut->member = %p\n", (void*)h_lut->member);\
    cudaMemcpy(tmp_dev, h_lut->member, (count) * sizeof(type), cudaMemcpyHostToDevice); \
    printf("d_lut->member");\
    printf(" = %p\n", (void*)d_lut->member);\
    cudaMemcpy(&(d_lut->member), &tmp_dev, sizeof(type*), cudaMemcpyHostToDevice); \
} while(0)

#include "decoder_graphs.h"
static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
static bool streamsCreated = false;
static bool d_mem_exist = false;
static int currentStreamCount = 0;
static int8_t* iter_ptr_array;//size of [MAX_NUM_DLSCH_SEGMENTS];
static int* PC_Flag_array;// size of[MAX_NUM_DLSCH_SEGMENTS];
 t_nrLDPC_lut* p_lut_dev = NULL;
static t_nrLDPC_lut* P_lut = NULL;

// device buffers (allocated in LDPCinit)
static int8_t* d_cnProcBuf = NULL;
static int8_t* d_cnProcBufRes = NULL;
static int8_t* d_bnProcBuf = NULL;
static int8_t* d_bnProcBufRes = NULL;
static int8_t* d_llrRes = NULL;
static int8_t* d_llrProcBuf = NULL;
static int8_t* d_llrOut = NULL;
static int8_t* d_out = NULL; // optional if needed per-seg
int gpuDeviceId;


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

extern void run_test_kernel();

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

void check_lut_pointers(const t_nrLDPC_lut* lut) {
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

t_nrLDPC_lut* copy_lut_to_device(const t_nrLDPC_lut* h_lut) {
    cudaError_t err;
    t_nrLDPC_lut* d_lut;
//printf("Inside copy 1\n");
    // malloc device end struct
    err = cudaMallocManaged((void**)&d_lut, sizeof(t_nrLDPC_lut), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_lut: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
//printf("Inside copy 2\n");
    // ---------------------------
    // copy all the member pointers
    // ---------------------------

    COPY_POINTER_MEMBER(startAddrCnGroups, uint32_t, 9);
    printf("Inside copy 3\n");
    COPY_POINTER_MEMBER(numCnInCnGroups, uint8_t, 9);
    printf("Inside copy 4\n");
    printf("host ptr = %p\n", (void*)d_lut->numBnInBnGroups);
    COPY_POINTER_MEMBER(numBnInBnGroups, uint8_t, 30);
    printf("Inside copy 5\n");
    printf("host ptr = %p\n", (void*)d_lut->startAddrBnGroups);
    printf("Inside copy 5.1\n");
    COPY_POINTER_MEMBER(startAddrBnGroups, uint32_t, 30);
    printf("Inside copy 6\n");
    COPY_POINTER_MEMBER(startAddrBnGroupsLlr, uint16_t, 30);
    printf("Inside copy 7\n");
    COPY_POINTER_MEMBER(llr2llrProcBufAddr, uint16_t, 26);
    printf("Inside copy 8\n");
    COPY_POINTER_MEMBER(llr2llrProcBufBnPos, uint8_t, 26);
    printf("Inside copy 9\n");
    //  COPY_POINTER_MEMBER
    // COPY_POINTER_MEMBER(numCnInCnGroups,  uint8_t,  X);
    // COPY_POINTER_MEMBER(numBnInBnGroups,  uint8_t,  Y);
    // ...

    // ---------------------------
    // cope with arr8_t/16_t/32_t
    // ---------------------------


    COPY_ARR_MEMBER(circShift,uint16_t, 9);
    COPY_ARR_MEMBER(startAddrBnProcBuf,uint32_t, 9);
    COPY_ARR_MEMBER(bnPosBnProcBuf,uint8_t, 9);
    COPY_ARR_MEMBER(posBnInCnProcBuf,uint8_t, 9);

    return d_lut;
}


extern void check_ptr_host(const void* p, const char* name);

#ifdef __cplusplus
extern "C" {
#endif

bool is_device_pointer(const void* p);

#ifdef __cplusplus
}
#endif

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

bool check_kernel_args_for_graph(const void* p_lut, // device
                                 const void* p_out, // may be host or device 
                                 const void* cnProcBuf, // device expected
                                 const void* cnProcBufRes, // device expected
                                 const void* bnProcBuf, // device expected
                                 const void* bnProcBufRes, // device expected
                                 const void* llrRes, // device expected
                                 const void* llrProcBuf, // device expected
                                 const void* llrOut, // device expected         // may be host or device
                                 const void* iter_ptr_array, // device expected (kernel iteration state)
                                 const void* iter_ptr_array2, // device expected 
                                 bool strict)
{
  bool ok = true;

  // check p_lut: should be dvice pointer
  if (is_device_pointer(p_lut)) {
    fprintf(stderr, "check_kernel_args_for_graph: p_lut should be HOST pointer: %p\n", p_lut);
    if (strict)
      return false;
    ok = false;
  }

  // check all the other buffer
  const void* device_ptrs[] =
      {cnProcBuf, cnProcBufRes, bnProcBuf, bnProcBufRes, llrRes, llrProcBuf, llrOut, iter_ptr_array, iter_ptr_array2};
  const char* device_names[] = {"cnProcBuf",
                                "cnProcBufRes",
                                "bnProcBuf",
                                "bnProcBufRes",
                                "llrRes",
                                "llrProcBuf",
                                "llrOut",
                                "iter_ptr_array",
                                "iter_ptr_array2"};
  for (int i = 0; i < (int)(sizeof(device_ptrs) / sizeof(device_ptrs[0])); i++) {
    if (!is_device_pointer(device_ptrs[i])) {
      fprintf(stderr, "check_kernel_args_for_graph: %s is NOT device pointer: %p\n", device_names[i], device_ptrs[i]);
      if (strict)
        return false;
      ok = false;
    }
  }

  if (!is_device_pointer(p_out)) {
    fprintf(stderr, "check_kernel_args_for_graph: p_out is NOT device pointer: %p\n", p_out);
    if (strict)
      return false;
    ok = false;
  }
  if (!is_device_pointer(d_llrOut)) {
    fprintf(stderr, "check_kernel_args_for_graph: p_llrOut is NOT device pointer: %p\n", d_llrOut);
    if (strict)
      return false;
    ok = false;
  }

  return ok;
}


int32_t LDPCinit_cuda()
{
  printf("CUDA LDPC decoder initiating\n");
  size_t cn_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_CN_PROC_BUF * sizeof(int8_t);
  size_t bn_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_SIZE_BN_PROC_BUF * sizeof(int8_t);
  size_t llr_bytes = MAX_NUM_DLSCH_SEGMENTS_DL * NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);
  size_t llrOut_bytes = NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);

  cudaGetDevice(&gpuDeviceId); //get device id

  cudaError_t err;
  err = cudaMallocManaged((void**)&d_cnProcBuf, cn_bytes, cudaMemAttachGlobal);
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
 err = cudaMallocManaged((void**)&d_llrProcBuf, llr_bytes, cudaMemAttachGlobal);
if (err != cudaSuccess) {
  fprintf(stderr, "cudaMallocManaged d_llrProcBuf failed: %s\n", cudaGetErrorString(err));
  return -1;
}
 err = cudaMallocManaged((void**)&iter_ptr_array, MAX_NUM_DLSCH_SEGMENTS_DL*sizeof(int8_t), cudaMemAttachGlobal);
if (err != cudaSuccess) {
  fprintf(stderr, "cudaMallocManaged iter_ptr_array failed: %s\n", cudaGetErrorString(err));
  return -1;
}

 err = cudaMallocManaged((void**)&PC_Flag_array, MAX_NUM_DLSCH_SEGMENTS_DL*sizeof(int), cudaMemAttachGlobal);
if (err != cudaSuccess) {
  fprintf(stderr, "cudaMallocManaged PC_Flag_array failed: %s\n", cudaGetErrorString(err));
  return -1;
}
  err = cudaMalloc((void**)&d_llrOut, MAX_NUM_DLSCH_SEGMENTS_DL * llrOut_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_pp_llrOut failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_out, MAX_NUM_DLSCH_SEGMENTS_DL*8448*sizeof(uint8_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
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
  if (d_out)
    cudaFree(d_out);

  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
    if (streamsCreated) {
      cudaEventDestroy(decoderDoneEvents[s]);
      cudaStreamDestroy(decoderStreams[s]);
    }
  }

  free_graphs();

  streamsCreated = false;
  d_mem_exist = false;

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
{ // Initialize decoder core(s) with correct LUTs
  if(p_decParams->R != 13 || p_decParams->BG != 1){    //format check
    printf("Current format: BG = %d, R = %d\n", p_decParams->BG, p_decParams->R);
    AssertFatal(false, "Format cuda not support, only support BG = 1 and R = 13 right now\n");
    return 0;
  }
#if STATIC_LUT
  if (!p_lutCreated) {
    //P_lut = p_lut;
    printf("Start to create p_lut\n");
    numLLR = nrLDPC_init(p_decParams, p_lut);
    printf("p_lut Created\n");
    //check p_lut
    //check_lut_pointers(p_lut);

    printf("Start to create p_lut_dev\n");
    p_lut_dev = copy_lut_to_device(p_lut);
    printf("p_lut_dev Created\n");

    p_lutCreated = true;
  }
#else
  uint32_t numLLR;
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;
#endif

  // Launch LDPC decoder core for one segment
  //printf("11111\n");
  int n_segments = p_decParams->n_segments;
  //printf("22222\n");

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
  //run_test_kernel();//just for testing
  /*
printf("=== Host p_lut->startAddrBnProcBuf dump ===\n");
        for (int i = 0; i < 9; i++) {
            printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
                   i,
                   (void*)p_lut->startAddrBnProcBuf[i].d,
                   p_lut->startAddrBnProcBuf[i].dim1,
                   p_lut->startAddrBnProcBuf[i].dim2);
        }

        printf("=== Host p_lut->bnPosBnProcBuf dump ===\n");
        for (int i = 0; i < 9; i++) {
            printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
                   i,
                   (void*)p_lut->bnPosBnProcBuf[i].d,
                   p_lut->bnPosBnProcBuf[i].dim1,
                   p_lut->bnPosBnProcBuf[i].dim2);
        }
  printf("n_segments = %d, R = %d\n", n_segments, p_decParams->R);
*/
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;
  int LastTrial = p_decParams->LastTrial;
/* move this part to LDPC_init
  if (d_mem_exist == false) {
    //P_lut = p_lut_dev;
    printf("2.1\n");
    //printf("Check P_lut = %d\n", P_lut->posBnInCnProcBuf[0]);
    //printf("2.2\n");
    LDPCinit_cuda(); // allocate device memory for the first time
    printf("2.3\n");
    d_mem_exist = true;
  }
*/
  //check_ptr_host(iter_ptr_array, "iter_ptr_array");
  //check_ptr_host(PC_Flag_array, "PC_Flag_array");

    for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; s++) {
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
  }
    cudaMemPrefetchAsync(p_lut_dev, sizeof(p_lut_dev), gpuDeviceId,0);
    cudaMemPrefetchAsync(iter_ptr_array, MAX_NUM_DLSCH_SEGMENTS_DL*sizeof(int8_t), gpuDeviceId,0);
    cudaMemPrefetchAsync(PC_Flag_array, MAX_NUM_DLSCH_SEGMENTS_DL*sizeof(int), gpuDeviceId,0);
//printf("Flag_ptr = %p\n", PC_Flag_array);
   printf("3.2: It works here\n");
  for (int CudaStreamIdx = 0; CudaStreamIdx < n_segments; CudaStreamIdx++) {
    //printf("3.21\n");
    int8_t* pp_llr = p_llr + CudaStreamIdx * 68 * 384; // no need put it into device
    //printf("3.22\n");
    int8_t* pp_out = d_out + CudaStreamIdx * Kprime;
    //printf("2.4\n");
    // printf("Stream %d: pp_out = %p\n", CudaStreamIdx, pp_out);
    int8_t* pp_cnProcBuf = d_cnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = d_cnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = d_bnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = d_bnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = d_llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = d_llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrOut = d_llrOut + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    // printf("4: It works here\n");
    //  LLR preprocessing
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
    printf("2.5\n");
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    printf("2.51\n");
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
    if (BG == 1){
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
      //printf("2.6\n");
      }
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));
    //  Call scheduler for this segment and stream
    //printf("3\n");
    int8_t* PP_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : pp_llrOut;
    // printf("5: It works here\n");
    //  Launch decoder on stream s

    // cudaEventCreate(&decoderDoneEvents[CudaStreamIdx]);
    // printf("Launching segment %d \n",CudaStreamIdx);

    //-------------------------check device pointer-----------------------
    /*check_kernel_args_for_graph(p_lut_dev,
                                pp_out, // pointer that will be passed to kernel (可能 host)
                                pp_cnProcBuf, // device expected
                                pp_cnProcBufRes,
                                pp_bnProcBuf,
                                pp_bnProcBufRes,
                                pp_llrRes,
                                pp_llrProcBuf,
                                pp_llrOut,
                                iter_ptr_array,
                                PC_Flag_array,
                                false);*/
    //------------------------check area end-------------------------------
//printf("4\n");
    cudaMemPrefetchAsync(pp_cnProcBuf, NR_LDPC_SIZE_CN_PROC_BUF, gpuDeviceId, decoderStreams[CudaStreamIdx]);//fetch cn_proc_buf to GPU
    cudaMemPrefetchAsync(pp_llrProcBuf, NR_LDPC_MAX_NUM_LLR*sizeof(int8_t), gpuDeviceId, decoderStreams[CudaStreamIdx]);
//printf("5\n");
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut_dev,
                                           pp_out,
                                           numLLR,
                                           pp_cnProcBuf,
                                           pp_cnProcBufRes,
                                           pp_bnProcBuf,
                                           pp_bnProcBufRes,
                                           pp_llrRes,
                                           pp_llrProcBuf,
                                           pp_llrOut,
                                           PP_llrOut,
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
//printf("6\n");
    // cudaEventRecord(done[CudaStreamIdx], streams[CudaStreamIdx]);
    // printf("5: It works here\n");
  }
  for (int s = 0; s < n_segments; ++s) {
    // printf("Synchronizing segment %d \n",s);
    cudaEventSynchronize(decoderDoneEvents[s]); // stop it until segment finish
    }
  cudaDeviceSynchronize();
  cudaMemcpy(p_out, d_out, MAX_NUM_DLSCH_SEGMENTS_DL * Kprime * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // cudaDeviceSynchronize();
  //  Wait for all streams
/*
  if (LastTrial == 1) {
    // printf("Now is the last trial\n");
    LDPCshutdown_cuda();
  }
*/
  // cudaDeviceSynchronize();
  //  dumpASS(p_out, "Dump_Output_Stream.txt");
  //  printf("6: It works here\n");

  return numMaxIter;
}

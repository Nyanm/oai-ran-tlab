

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

#ifdef PARALLEL_STREAM
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
            /* 更新 d_lut->member[i].d 指针 */ \
            cudaMemcpy(&(d_lut->member[i].d), &tmp_dev, sizeof(type*), cudaMemcpyHostToDevice); \
            /* 拷贝 dim1 和 dim2 */ \
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
static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS];
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
#endif

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
#ifdef PARALLEL_STREAM
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
#endif

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

    // 对数组类型打印首地址
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

  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS * 68 * 384; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}
//--------------------------------------------------------------
#ifdef PARALLEL_STREAM

t_nrLDPC_lut* copy_lut_to_device(const t_nrLDPC_lut* h_lut) {
    cudaError_t err;
    t_nrLDPC_lut* d_lut;
printf("Inside copy 1\n");
    // 分配 device 端的结构体
    err = cudaMallocManaged((void**)&d_lut, sizeof(t_nrLDPC_lut), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_lut: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
printf("Inside copy 2\n");
    // ---------------------------
    // 处理普通指针成员 (host 静态数组)
    // ---------------------------


    // 举例：假设 BG1 的 startAddrCnGroups 长度是 9
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
    // 其他类似的成员也要按实际长度调用 COPY_POINTER_MEMBER
    // COPY_POINTER_MEMBER(numCnInCnGroups,  uint8_t,  X);
    // COPY_POINTER_MEMBER(numBnInBnGroups,  uint8_t,  Y);
    // ...

    // ---------------------------
    // 处理 arr8_t/16_t/32_t
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

void free_graphs()
{
  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS; i++) {
    if (graphCreated[i]) {
      cudaGraphExecDestroy(decoderGraphExec[i]);
      cudaGraphDestroy(decoderGraphs[i]);
      graphCreated[i] = false;
    }
  }
}

bool check_kernel_args_for_graph(const void* p_lut, // host expected (重要 host 常量 LUT)
                                 const void* p_out, // may be host or device (建议 device 或 managed)
                                 const void* cnProcBuf, // device expected
                                 const void* cnProcBufRes, // device expected
                                 const void* bnProcBuf, // device expected
                                 const void* bnProcBufRes, // device expected
                                 const void* llrRes, // device expected
                                 const void* llrProcBuf, // device expected
                                 const void* llrOut, // device expected         // may be host or device
                                 const void* iter_ptr_array, // device expected (kernel iteration state)
                                 const void* iter_ptr_array2, // device expected (第二份迭代指针)
                                 bool strict)
{
  bool ok = true;

  // 检查 p_lut: 必须是 Host pointer
  if (is_device_pointer(p_lut)) {
    fprintf(stderr, "check_kernel_args_for_graph: p_lut should be HOST pointer: %p\n", p_lut);
    if (strict)
      return false;
    ok = false;
  }

  // 检查 device 预期的 buffer
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

  // p_out / p_llrOut 可以是 host 也可以是 device（建议 managed 或 device）
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

int32_t LDPCinit()
{
  return 0;
}
int32_t LDPCinit_cuda()
{
  size_t cn_bytes = MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_CN_PROC_BUF * sizeof(int8_t);
  size_t bn_bytes = MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_SIZE_BN_PROC_BUF * sizeof(int8_t);
  size_t llr_bytes = MAX_NUM_DLSCH_SEGMENTS * NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);
  size_t llrOut_bytes = NR_LDPC_MAX_NUM_LLR * sizeof(int8_t);

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
 err = cudaMallocManaged((void**)&iter_ptr_array, MAX_NUM_DLSCH_SEGMENTS*sizeof(int8_t), cudaMemAttachGlobal);
if (err != cudaSuccess) {
  fprintf(stderr, "cudaMallocManaged iter_ptr_array failed: %s\n", cudaGetErrorString(err));
  return -1;
}

 err = cudaMallocManaged((void**)&PC_Flag_array, MAX_NUM_DLSCH_SEGMENTS*sizeof(int), cudaMemAttachGlobal);
if (err != cudaSuccess) {
  fprintf(stderr, "cudaMallocManaged PC_Flag_array failed: %s\n", cudaGetErrorString(err));
  return -1;
}



  err = cudaMalloc((void**)&d_llrOut, llrOut_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_pp_llrOut failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  err = cudaMalloc((void**)&d_out, 13*8448*sizeof(uint8_t));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  /*err = cudaMalloc((void**)&p_lut_dev, sizeof(t_nrLDPC_lut));
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc for p_lut_dev failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // 拷贝整个结构体到 device
  printf("Copying LUT from host %p to device %p, size=%zu\n", (void*)P_lut, (void*)p_lut_dev, sizeof(t_nrLDPC_lut));
  err = cudaMemcpy(p_lut_dev, P_lut, sizeof(t_nrLDPC_lut), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy for p_lut_dev failed: %s\n", cudaGetErrorString(err));
    return 0;
  }*/
  // create streams/events if not already
  if (!streamsCreated) {
    for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS; ++s) {
      cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
      cudaEventCreate(&decoderDoneEvents[s]);
    }
    streamsCreated = true;
  }
  

  return 0;
}

int32_t LDPCshutdown()
{
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

  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS; ++s) {
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

int32_t LDPCdecoder(t_nrLDPC_dec_params* p_decParams,
                    uint8_t harq_pid,
                    uint8_t ulsch_id,
                    uint8_t C,
                    int8_t* p_llr,
                    int8_t* p_out,
                    t_nrLDPC_time_stats* p_profiler,
                    decode_abort_t* ab)
{ // Initialize decoder core(s) with correct LUTs
#if STATIC_LUT
  if (!p_lutCreated) {
    //P_lut = p_lut;
    printf("Start to create p_lut\n");
    numLLR = nrLDPC_init(p_decParams, p_lut);
    printf("p_lut Created\n");
    //check p_lut
    check_lut_pointers(p_lut);

    #ifdef PARALLEL_STREAM
    printf("Start to create p_lut_dev\n");
    p_lut_dev = copy_lut_to_device(p_lut);
    printf("p_lut_dev Created\n");
    #endif
    p_lutCreated = true;
  }
#else
  uint32_t numLLR;
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;
#endif

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
  //run_test_kernel();//just for testing

  // printf("n_segments = %d\n", n_segments);
  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;
  int LastTrial = p_decParams->LastTrial;

  if (d_mem_exist == false) {
    P_lut = p_lut_dev;
    //printf("2.1\n");
    //printf("Check P_lut = %d\n", P_lut->posBnInCnProcBuf[0]);
    //printf("2.2\n");
    LDPCinit_cuda(); // allocate device memory for the first time
    //printf("2.3\n");
    d_mem_exist = true;
  }

  //check_ptr_host(iter_ptr_array, "iter_ptr_array");
  //check_ptr_host(PC_Flag_array, "PC_Flag_array");

    for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS; s++) {
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
  }

  // printf("3.2: It works here\n");
  for (int CudaStreamIdx = 0; CudaStreamIdx < n_segments; CudaStreamIdx++) {
    int8_t* pp_llr = p_llr + CudaStreamIdx * 68 * 384; // no need put it into device
    int8_t* pp_out = d_out + CudaStreamIdx * Kprime;
    //printf("2.4\n");
    // printf("Stream %d: pp_out = %p\n", CudaStreamIdx, pp_out);
    int8_t* pp_cnProcBuf = d_cnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = d_cnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = d_bnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = d_bnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = d_llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = d_llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    // printf("4: It works here\n");
    //  LLR preprocessing
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
    //printf("2.5\n");
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    //printf("2.51\n");
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
    if (BG == 1)
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
      //printf("2.6\n");}
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));
    //  Call scheduler for this segment and stream
    //printf("3\n");
    int8_t* pp_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : d_llrOut;
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
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut_dev,
                                           pp_out,
                                           numLLR,
                                           pp_cnProcBuf,
                                           pp_cnProcBufRes,
                                           pp_bnProcBuf,
                                           pp_bnProcBufRes,
                                           pp_llrRes,
                                           pp_llrProcBuf,
                                           d_llrOut,
                                           pp_llrOut,
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

    // cudaEventRecord(done[CudaStreamIdx], streams[CudaStreamIdx]);
    // printf("5: It works here\n");
  }
  for (int s = 0; s < n_segments; ++s) {
    // printf("Synchronizing segment %d \n",s);
    cudaEventSynchronize(decoderDoneEvents[s]); // 阻塞直到该 segment 解码完成
  }
  cudaDeviceSynchronize();
  cudaMemcpy(p_out, d_out, 13 * Kprime * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // cudaDeviceSynchronize();
  //  Wait for all streams
  // for (int s = 0; s < n_segments; s++) {
  //   cudaEventSynchronize(done[s]); // 等待stream[i]完成
  //   // 可安全访问对应的解码输出结果 p_llrOut[i]
  //}
  if (LastTrial == 1) {
    // printf("Now is the last trial\n");
    LDPCshutdown_cuda();
  }
  // cudaDeviceSynchronize();
  //  dumpASS(p_out, "Dump_Output_Stream.txt");
  //  printf("6: It works here\n");

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
#if STATIC_LUT
  if (!p_lutCreated) {
    numLLR = nrLDPC_init(p_decParams, p_lut);
    printf("I'm here everytime\n");
    p_lutCreated = true;
  }
#else
  uint32_t numLLR;
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;
#endif
  // Initialize decoder core(s) with correct LUTs

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
  // printf("cnProcBuf address: %p\n",cnProcBuf);
  //  Initialization
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
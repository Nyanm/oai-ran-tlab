

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

#define MAX_NUM_DLSCH_SEGMENTS_DL 132
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

#define USE_STATIC_ALLOC
static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
static bool streamsCreated = false;
static int currentStreamCount = 0;
#ifdef USE_STATIC_ALLOC
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
#else
int8_t *iter_ptr_array_dev;
int *PC_Flag_array_dev;
int8_t *cnProcBuf_dev;
int8_t *cnProcBufRes_dev;
int8_t *bnProcBuf_dev;
int8_t *bnProcBufRes_dev;
int8_t *llrRes_dev;
int8_t *llrProcBuf_dev;
int8_t *llrOut_dev;
t_nrLDPC_lut *d_lut_R13;
t_nrLDPC_lut *d_lut_R23;

int8_t *iter_ptr_array_host;
int *PC_Flag_array_host;
int8_t *cnProcBuf_host;
int8_t *cnProcBufRes_host;
int8_t *bnProcBuf_host;
int8_t *bnProcBufRes_host;
int8_t *llrRes_host;
int8_t *llrProcBuf_host;
int8_t *llrOut_host;
t_nrLDPC_lut *h_lut_R13;
t_nrLDPC_lut *h_lut_R23;


extern int pageable, register_host;

#define COPY_ARR_MEMBER(member, type, groups) do { \
    for (int i = 0; i < (groups); i++) { \
        type* tmp_dev; \
        if (h_lut->member[i].d != NULL && h_lut->member[i].dim1 > 0 && h_lut->member[i].dim2 > 0) { \
            size_t sz = h_lut->member[i].dim1 * h_lut->member[i].dim2 * sizeof(type); \
            err = cudaHostAlloc((void**)&tmp_dev, sz,cudaHostAllocMapped); \
            if (err != cudaSuccess) { \
                fprintf(stderr, "cudaMalloc failed for " #member "[%d]: %s\n", i, cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
            memcpy(tmp_dev, h_lut->member[i].d, sz); \
            /* updtae d_lut->member[i].d pointer */ \
            memcpy(&(d_lut->member[i].d), &tmp_dev, sizeof(type*)); \
            /* copy dim1 and dim2 */ \
            memcpy(&(d_lut->member[i].dim1), &(h_lut->member[i].dim1), sizeof(int)); \
            memcpy(&(d_lut->member[i].dim2), &(h_lut->member[i].dim2), sizeof(int)); \
        } \
    } \
} while(0)

#define COPY_POINTER_MEMBER(member, type, count) do { \
    type* tmp_dev; \
    printf("tmp_dev = %p\n", (void*)tmp_dev);\
    err = cudaHostAlloc((void**)&tmp_dev, (count) * sizeof(type),cudaHostAllocMapped); \
    printf("malloc tmp_dev = %p\n", (void*)tmp_dev);\
    if (err != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc failed for " #member ": %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    printf("h_lut->member = %p\n", (void*)h_lut->member);\
    memcpy(tmp_dev, h_lut->member, (count) * sizeof(type)); \
    printf("d_lut->member");\
    printf(" = %p, tmp_dev %p\n", (void*)d_lut->member,tmp_dev);\
    d_lut->member = tmp_dev; \
} while(0)

int numLLR_R13,numLLR_R23;

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

void copy_luts_to_pinned() {
    cudaError_t err;
    t_nrLDPC_dec_params params;
    // ---------------------------
    // copy all the member pointers
    // ---------------------------
    t_nrLDPC_lut *h_lut = h_lut_R13;
    t_nrLDPC_lut *d_lut = d_lut_R13;
    params.BG=1;
    params.Z=384;
    params.R=13;
    numLLR_R13 = nrLDPC_init(&params,h_lut);

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

    //check_lut_pointers(d_lut);

    h_lut = h_lut_R23;
    d_lut = d_lut_R23;
    params.BG=1;
    params.Z=384;
    params.R=23;
    numLLR_R23 = nrLDPC_init(&params,h_lut);

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
}
int cuda_support_init_decoder() {
  if (!pageable && !register_host) {
    cudaError_t err=cudaMalloc((void **)&cnProcBuf_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF);
    AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBuf_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&cnProcBufRes_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF);
    AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&bnProcBuf_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&bnProcBufRes_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBufRes_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&llrRes_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&llrProcBuf_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&llrOut_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&iter_ptr_array_dev,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4);
    AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&PC_Flag_array_dev,sizeof(int)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4);
    AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&d_lut_R13,sizeof(*d_lut_R13));
    AssertFatal(err == cudaSuccess,"CUDA Error (d_lut_R13): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void **)&d_lut_R23,sizeof(*d_lut_R23));
    AssertFatal(err == cudaSuccess,"CUDA Error (d_lut_R13): %s\n", cudaGetErrorString(err));
  }
  else {
    cudaError_t err=cudaHostAlloc((void **)&cnProcBuf_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (c_dev): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&cnProcBuf_dev, cnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBuf_host): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&cnProcBufRes_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_CN_PROC_BUF,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&cnProcBufRes_dev, cnProcBufRes_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (cnProcBufRes_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&bnProcBuf_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBuf_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&bnProcBuf_dev, bnProcBuf_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBuf_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&bnProcBufRes_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_SIZE_BN_PROC_BUF,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBufRes_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&bnProcBufRes_dev, bnProcBufRes_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (bnProcBufRes_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&llrRes_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrRes_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&llrRes_dev, llrRes_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrRes_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&llrProcBuf_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrProcBuf_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&llrProcBuf_dev, llrProcBuf_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrProcBuf_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&llrOut_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4 * NR_LDPC_MAX_NUM_LLR,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrOut_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&llrOut_dev, llrOut_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (llrOut_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&iter_ptr_array_host,sizeof(int8_t)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&iter_ptr_array_dev, iter_ptr_array_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (iter_ptr_array_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&PC_Flag_array_host,sizeof(int)* MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4,cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&PC_Flag_array_dev, PC_Flag_array_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (PC_Flag_array_dev): %s\n", cudaGetErrorString(err));

    err=cudaHostAlloc((void **)&h_lut_R13,sizeof(*h_lut_R13),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (h_lut_R13): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&h_lut_R23,sizeof(*h_lut_R23),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (h_lut_R23): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&d_lut_R13, h_lut_R13, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (d_lut_R13): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&d_lut_R23, h_lut_R23, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (d_lut_R23): %s\n", cudaGetErrorString(err));
    copy_luts_to_pinned();
  }
  return 0;
}
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
  if (cuda_support_set == 0 ) {
     cuda_support_init();
     printf("CUDA LDPC decoder initiating\n");
#ifndef USE_STATIC_ALLOC
    cuda_support_init_decoder(); 
#endif
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
#ifdef USE_STATIC_ALLOC
  t_nrLDPC_lut lut;
  t_nrLDPC_lut* p_lut = &lut;
  // Initialize decoder core(s) with correct LUTs
  numLLR = nrLDPC_init(p_decParams, p_lut);
#else
  t_nrLDPC_lut *p_lut;
  if (p_decParams->R == 13) {
	  numLLR = numLLR_R13;
	  p_lut = d_lut_R13;
  }
  else {
	  numLLR = numLLR_R23;
	  p_lut = d_lut_R23;
  }
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
  int8_t temp_out[n_segments * 8448] __attribute__((aligned(64))); 
#ifdef USE_STATIC_ALLOC
  memcpy(temp_in , p_llr ,  n_segments * 68 * 384);
#endif
  memset(temp_out, 0     ,  n_segments * 8448);

  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter; // To match the actual iterations
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;

  for (int s = 0; s < n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/; s++) {
#ifdef USE_STATIC_ALLOC
    iter_ptr_array[s] = 0;
    PC_Flag_array[s] = 1;
#else
    iter_ptr_array_dev[s] = 0;
    PC_Flag_array_dev[s] = 1;
#endif
  }
  for (int CudaStreamIdx = 0; CudaStreamIdx < n_segments; CudaStreamIdx++) {
#ifdef USE_STATIC_ALLOC
    int8_t* pp_llr = temp_in + CudaStreamIdx * 68 * 384 ;
    int8_t* pp_out = temp_out + CudaStreamIdx * 8448; // use temp_out rather than p_out
#else
    int8_t* pp_llr = p_llr + CudaStreamIdx * 68 * 384 ;
    int8_t* pp_out = p_out + CudaStreamIdx * 8448; // use temp_out rather than p_out
#endif
    // printf("Stream %d: pp_out = %p\n", CudaStreamIdx, pp_out);

#ifdef USE_STATIC_ALLOC
    int8_t* pp_cnProcBuf = cnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = cnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = bnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = bnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrOut = llrOut + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
#else
    int8_t* pp_cnProcBuf = cnProcBuf_dev + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = cnProcBufRes_dev + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = bnProcBuf_dev + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = bnProcBufRes_dev + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = llrRes_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = llrProcBuf_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrOut = llrOut_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;

#endif
/*
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    if (BG == 1)
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
*/

    //  Call scheduler for this segment and stream
    int8_t* pp_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : pp_llrOut;

    //  Launch decoder on stream
    nrLDPC_decoder_scheduler_BG1_cuda_core(p_lut,
                                           pp_out,
                                           numLLR,
                                           pp_llr,
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
#ifdef USE_STATIC_ALLOC
                                           &iter_ptr_array[CudaStreamIdx],
                                           &PC_Flag_array[CudaStreamIdx]
#else
                                           &iter_ptr_array_dev[CudaStreamIdx],
                                           &PC_Flag_array_dev[CudaStreamIdx]
#endif
					   ); // stream index passed in
  }
  for (int s = 0; s < n_segments; ++s) {
    // printf("Synchronizing segment %d \n",s);
    cudaEventSynchronize(decoderDoneEvents[s]); // stop until segment decode
  }
  cudaDeviceSynchronize();

  // cudaDeviceSynchronize();
  //printf("p_out %p, temp_out %p\n",p_out,temp_out);
#ifdef USE_STATIC_ALLOC 
  memcpy(p_out, temp_out, n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/ * 8448);
#endif
  //dumpASS(p_out, "Dump_Output_Stream_GH_new.txt");

  return numMaxIter;
}

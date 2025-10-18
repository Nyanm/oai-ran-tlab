

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
//#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"
#include "nrLDPC_cnProc.h"
#include "nrLDPC_bnProc.h"
#include "openair1/PHY/CODING/coding_defs.h"

#include "openair1/PHY/CODING/nrLDPC_extern.h"


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

static bool streamsCreated = false;
static cudaStream_t decoderStreams[MAX_NUM_DLSCH_SEGMENTS_DL];
static cudaEvent_t decoderDoneEvents[MAX_NUM_DLSCH_SEGMENTS_DL];
int8_t *iter_ptr_array_dev;
int *PC_Flag_array_dev;
int8_t *cnProcBuf_dev;
int8_t *cnProcBufRes_dev;
int8_t *bnProcBuf_dev;
int8_t *bnProcBufRes_dev;
int8_t *llrRes_dev;
int8_t *llrProcBuf_dev;
int8_t *llrOut_dev;
t_nrLDPC_lut *lut_dev;

int8_t *iter_ptr_array_host;
int *PC_Flag_array_host;
int8_t *cnProcBuf_host;
int8_t *cnProcBufRes_host;
int8_t *bnProcBuf_host;
int8_t *bnProcBufRes_host;
int8_t *llrRes_host;
int8_t *llrProcBuf_host;
int8_t *llrOut_host;
t_nrLDPC_lut *lut_host;

extern int pageable, register_host;
/* 
const uint32_t startAddrCnGroups_BG1[NR_LDPC_NUM_CN_GROUPS_BG1] = {0, 1152, 8832, 43392, 61824, 75264, 81408, 88320, 92160};
extern uint32_t *gpu_lut_startAddrCnGroups_BG1;

const uint32_t startAddrBnGroups_BG1_R13[NR_LDPC_NUM_BN_GROUPS_BG1_R13] = {0, 16128, 17664, 19584, 24192, 34944, 44160, 47616, 62976, 75648, 94080, 99072, 109824};
extern uint32_t *gpu_lut_startAddrBnGroups_BG1_R13;

const uint32_t startAddrBnGroups_BG1_R23[NR_LDPC_NUM_BN_GROUPS_BG1_R23] = {0, 3456, 4224, 9984, 14592, 28032, 46464, 50688};
extern uint32_t *gpu_lut_startAddrBnGroups_BG1_R23;

const uint32_t numBnInBnGroups_BG1_R13[NR_LDPC_NUM_BN_GROUPS_BG1_R13] = {42, 0, 0, 1, 1, 2, 4, 3, 1, 4, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};
extern uint32_t *gpu_lut_numBnInBnGroups_BG1_R13;

const uint32_t numBnInBnGroups_BG1_R23[NR_LDPC_NUM_BN_GROUPS_BG1_R13] = { 9, 1, 5, 3, 7, 8, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
extern uint32_t *gpu_lut_numBnInBnGroups_BG1_R23;

const uint32_t startAddrBnGroupsLlr_BG1_R13[NR_LDPC_NUM_BN_GROUPS_BG1_R13] = {0, 16128, 16512, 16896, 17664, 19200, 20352, 20736, 22272, 23424, 24960, 25344, 25728};
extern uint32_t *gpu_lut_startAddrBnGroupsLlr_BG1_R13;

const uint32_t startAddrBnGroupsLlr_BG1_R23[NR_LDPC_NUM_BN_GROUPS_BG1_R23] = {0, 3456, 3840, 5760, 6912, 9600, 12672, 13056};
extern uint32_t *gpu_lut_startAddrBnGroupsLlr_BG1_R23;
*/
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
    err=cudaMalloc((void **)&lut_dev,sizeof(*lut_dev));
    AssertFatal(err == cudaSuccess,"CUDA Error (lut_dev): %s\n", cudaGetErrorString(err));
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

    err=cudaHostAlloc((void **)&lut_host,sizeof(*lut_host),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (lut_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&lut_dev, lut_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (lut_dev): %s\n", cudaGetErrorString(err));
/*
    err=cudaHostAlloc((void **)gpu_lut_startAddrCnGroups_BG1,sizeof(startAddrCnGroups_BG1),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_startAddrCnGroups_BG1): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_startAddrBnGroups_BG1_R13,sizeof(startAddrBnGroups_BG1_R13),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_startAddrBnGroups_BG1_R13): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_startAddrBnGroups_BG1_R23,sizeof(startAddrBnGroups_BG1_R23),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_startAddrBnGroups_BG1_R23): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_numBnInBnGroups_BG1_R13,sizeof(numBnInBnGroups_BG1_R13),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_numBnInBnGroups_BG1_R13): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_numBnInBnGroups_BG1_R23,sizeof(numBnInBnGroups_BG1_R23),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_numBnInBnGroups_BG1_R23): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_startAddrBnGroupsLlr_BG1_R13,sizeof(startAddrBnGroupsLlr_BG1_R13),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_startAddrBnGroupsLlr_BG1_R13): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)gpu_lut_startAddrBnGroupsLlr_BG1_R23,sizeof(startAddrBnGroupsLlr_BG1_R23),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (gpu_lut_startAddrBnGroupsLlr_BG1_R23): %s\n", cudaGetErrorString(err));
    */
  }
  return 0;
}

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

  for (int i = 0; i < MAX_NUM_DLSCH_SEGMENTS_DL * 8448; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes_host[i]);
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
  if (cuda_support_set == 0 ) {
    printf("Calling encoder initializations\n");	
    cuda_support_init();
  }
  if (!streamsCreated) {
    printf("CUDA LDPC decoder initiating\n");
    for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
        cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
        cudaEventCreate(&decoderDoneEvents[s]);
    }
    streamsCreated = true;
    init_decoder_graphs();
  }
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
  if (!((p_decParams->R == 23 || p_decParams->R == 13)&&p_decParams->BG == 1)) { // format check
    printf("Current format: BG = %d, R = %d\n", p_decParams->BG, p_decParams->R);
    AssertFatal(false, "Format cuda not support, only support BG = 1 and R = 13, 23 right now\n");
    return 0;
  }
  uint32_t numLLR;
  t_nrLDPC_lut* p_lut = lut_host;

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

  int8_t temp_out[/*MAX_NUM_DLSCH_SEGMENTS_DL*/n_segments * 8448] __attribute__((aligned(64))); /* = {0};*/
  memset(temp_out,0,n_segments * 8448);

  uint16_t Z = p_decParams->Z;
  uint8_t BG = p_decParams->BG;
  uint8_t R = p_decParams->R; // Decoding rate: Format 15,13,... for code rates 1/5, 1/3,... */
  uint8_t numMaxIter = p_decParams->numMaxIter;
  e_nrLDPC_outMode outMode = p_decParams->outMode;
  int Kprime = p_decParams->Kprime;
  // int LastTrial = p_decParams->LastTrial;
  //  printf("Kprime = %d\n", Kprime);
  //   int8_t* cnProcBuf=  cnProcBuf;
  //   int8_t* cnProcBufRes= cnProcBufRes;
  //  printf("1: It works here\n");

  // printf("2: It works here\n");
  //  Minimum number of iterations is 1
  //  0 iterations means hard-decision on input LLRs
  //  Initialize with parity check fail != 0
  // printf("3: It works here\n");
  //  Initialization
  // cudaStream_t streams[MAX_NUM_DLSCH_SEGMENTS];
  // cudaEvent_t done[MAX_NUM_DLSCH_SEGMENTS]; // MAX_NUM_SEGMENTS = stream num

  for (int s = 0; s < n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/; s++) {
    iter_ptr_array_host[s] = 0;
    PC_Flag_array_host[s] = 1;
  }
  // printf("3.1: It works here\n");
  /*
  if (!streamsCreated) {
  for (int s = 0; s < MAX_NUM_DLSCH_SEGMENTS_DL; ++s) {
    cudaStreamCreateWithFlags(&decoderStreams[s], cudaStreamNonBlocking);
    cudaEventCreate(&decoderDoneEvents[s]);
  }
  streamsCreated = true;
  currentStreamCount = n_segments;
}
*/
  // printf("3.2: It works here\n");
  for (int CudaStreamIdx = 0; CudaStreamIdx < n_segments; CudaStreamIdx++) {
    int8_t* pp_llr = p_llr + CudaStreamIdx * 68 * 384 ;
    int8_t* pp_out = temp_out + CudaStreamIdx * 8448; // use temp_out rather than p_out
    // printf("Stream %d: pp_out = %p\n", CudaStreamIdx, pp_out);
        /*
    int8_t* pp_cnProcBuf ,pp_cnProcBufRes , pp_bnProcBuf ,pp_bnProcBufRes ,pp_llrRes ,pp_llrProcBuf ,pp_llrOut ;
    switch (R)
    {
    case 13:
    pp_cnProcBuf = cnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    pp_cnProcBufRes = cnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    pp_bnProcBuf = bnProcBuf + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    pp_bnProcBufRes = bnProcBufRes + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    pp_llrRes = llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    pp_llrProcBuf = llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    pp_llrOut = llrOut + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
      break;
    case 23:
      pp_cnProcBuf = cnProcBuf + CudaStreamIdx * 144*384;
    pp_cnProcBufRes = cnProcBufRes + CudaStreamIdx * 144*384;
    pp_bnProcBuf = bnProcBuf + CudaStreamIdx * 144*384;
    pp_bnProcBufRes = bnProcBufRes + CudaStreamIdx * 144*384;
    pp_llrRes = llrRes + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    pp_llrProcBuf = llrProcBuf + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    pp_llrOut = llrOut + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
      break;
    
    default:
      break;
    }
    */
    int8_t* pp_cnProcBuf = cnProcBuf_dev + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_cnProcBufRes = cnProcBufRes_dev + CudaStreamIdx * NR_LDPC_SIZE_CN_PROC_BUF;
    int8_t* pp_bnProcBuf = bnProcBuf_dev + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_bnProcBufRes = bnProcBufRes_dev + CudaStreamIdx * NR_LDPC_SIZE_BN_PROC_BUF;
    int8_t* pp_llrRes = llrRes_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrProcBuf = llrProcBuf_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    int8_t* pp_llrOut = llrOut_dev + CudaStreamIdx * NR_LDPC_MAX_NUM_LLR;
    // printf("4: It works here\n");
    //  LLR preprocessing
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2llrProcBuf));
    nrLDPC_llr2llrProcBuf(p_lut, pp_llr, pp_llrProcBuf, Z, BG);
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2llrProcBuf));
    // NR_LDPC_PROFILER_DETAIL(start_meas(&p_profiler->llr2CnProcBuf));
    if (BG == 1)
      nrLDPC_llr2CnProcBuf_BG1(p_lut, pp_llr, pp_cnProcBuf, Z);
    else
      nrLDPC_llr2CnProcBuf_BG2(p_lut, pp_llr, pp_cnProcBuf, Z);
    // NR_LDPC_PROFILER_DETAIL(stop_meas(&p_profiler->llr2CnProcBuf));
    //  Call scheduler for this segment and stream
    int8_t* pp_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? pp_out : pp_llrOut;
    // printf("5: It works here\n");
    //  Launch decoder on stream s

    // cudaEventCreate(&decoderDoneEvents[CudaStreamIdx]);
    // printf("Launching segment %d \n",CudaStreamIdx);
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
                                           &iter_ptr_array_dev[CudaStreamIdx],
                                           &PC_Flag_array_dev[CudaStreamIdx]); // stream index passed in
  }
  for (int s = 0; s < n_segments; ++s) {
    // printf("Synchronizing segment %d \n",s);
    cudaEventSynchronize(decoderDoneEvents[s]); // stop until segment decode
  }
  cudaDeviceSynchronize();
  /*
  if(LastTrial == 1){
    //printf("Now is the last trial\n");
    for (int s = 0; s < n_segments; s++) {
      cudaEventDestroy(decoderDoneEvents[s]);
      cudaStreamSynchronize(decoderStreams[s]);
      cudaStreamDestroy(decoderStreams[s]);
      streamsCreated = false;
    }
  }
    */
  // cudaDeviceSynchronize();
  //printf("p_out %p, temp_out %p\n",p_out,temp_out);
  memcpy(p_out, temp_out, n_segments /*MAX_NUM_DLSCH_SEGMENTS_DL*/ * 8448);
  //dumpASS(p_out, "Dump_Output_Stream_GH.txt");
  // printf("6: It works here\n");

  return numMaxIter;
}

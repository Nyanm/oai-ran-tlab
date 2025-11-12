#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

#include "nrLDPC_CUDA_lut.h"
#include "nrLDPC_CUDA_CnProcKernel_BG1.h"
#include "nrLDPC_CUDA_BnProcKernel_BG1.h"
#include "nrLDPC_CUDA_mPassKernel_BG1.h"
#include "nrLDPC_CUDA_shared_param.h"

#define ZC 384 // for BG1 test only
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#define RECORD_GRAPH 0 // set 1 to enable graph recording, 0 to unable.

#ifndef JETSON_TARGET
#define CUDA_THREADS 1024
#define CUDA_BLOCKS_R13 30
#define CUDA_BLOCKS_R23 14
#else
#define CUDA_THREADS 128
#define CUDA_BLOCKS_R13 237 // ceil(30336/128)
#define CUDA_BLOCKS_R23 108 // ceil(13824/128)
#endif

/*
#define CUDA_THREADS 960
#define CUDA_BLOCKS_R13 32 // ceil(30336/960)
#define CUDA_BLOCKS_R23 15 // ceil(13824/960)
*/
/*
#define CUDA_THREADS 768
#define CUDA_BLOCKS_R13 40 // ceil(30336/768)
#define CUDA_BLOCKS_R23 18 // ceil(13824/768)
*/
/*
#define CUDA_THREADS 512
#define CUDA_BLOCKS_R13 60 // ceil(30336/512)
#define CUDA_BLOCKS_R23 27 // ceil(13824/512)
*/
/*
#define CUDA_THREADS 384
#define CUDA_BLOCKS_R13 79// ceil(30336/384)
#define CUDA_BLOCKS_R23 36// ceil(13824/384)
*/

/*
#define CUDA_THREADS 96
#define CUDA_BLOCKS_R13 316 // ceil(30336/128)
#define CUDA_BLOCKS_R23 144 // ceil(13824/128)
*/
/*
#define CUDA_THREADS 64
#define CUDA_BLOCKS_R13 474 // ceil(30336/64)
#define CUDA_BLOCKS_R23 216 // ceil(13824/64)
*/
cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL] = {false};

SegmentPack segmentPacks[MAX_NUM_DLSCH_SEGMENTS_DL];

KernelLaunchConfig Kdim_R13[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
KernelLaunchConfig Kdim_R23[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
ThreadSize BG1_R13_threadSize, BG1_R23_threadSize;

// debug function
void dumpAssCUDA(const int8_t *cnProcBufRes, const char *filename)
{
  FILE *fp = fopen(filename, "w");
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

// === CUDA Error Checking ===
// Wrap any CUDA API call with CHECK(...) to automatically print error info with file and line number
// Example usage: CHECK(cudaMalloc(&ptr, size));
#define CHECK(call) ErrorCheck((call), __FILE__, __LINE__)
/**
 * @brief Checks CUDA error status and prints detailed diagnostic info if an error occurred.
 *
 * @param error_code The CUDA error code returned from a CUDA runtime API call.
 * @param filename   The name of the source file where the error occurred.
 * @param lineNumber The line number in the source file where the error occurred.
 * @return cudaError_t Returns the same error code passed in, for optional further handling.
 */
inline cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber)
{
  if (error_code != cudaSuccess) {
    printf("[CUDA ERROR] %s (%d): %s\nOccurred in file: %s at line %d\n",
           cudaGetErrorName(error_code),
           error_code,
           cudaGetErrorString(error_code),
           filename,
           lineNumber);
  }
  return error_code;
}

#define COPY_ARR_MEMBER(member, type, groups)                                                               \
  do {                                                                                                      \
    for (int i = 0; i < (groups); i++) {                                                                    \
      type *tmp_dev;                                                                                        \
      if (h_lut.member[i].d != NULL && h_lut.member[i].dim1 > 0 && h_lut.member[i].dim2 > 0) {              \
        size_t sz = h_lut.member[i].dim1 * h_lut.member[i].dim2 * sizeof(type);                             \
        err = cudaMalloc((void **)&tmp_dev, sz);                                                            \
        if (err != cudaSuccess) {                                                                           \
          fprintf(stderr, "cudaMalloc failed for " #member "[%d]: %s\n", i, cudaGetErrorString(err));       \
          exit(EXIT_FAILURE);                                                                               \
        }                                                                                                   \
        cudaMemcpy(tmp_dev, h_lut.member[i].d, sz, cudaMemcpyHostToDevice);                                 \
        /* updtae d_lut->member[i].d pointer */                                                             \
        cudaMemcpy(&(d_lut->member[i].d), &tmp_dev, sizeof(type *), cudaMemcpyHostToDevice);                \
        /* copy dim1 and dim2 */                                                                            \
        cudaMemcpy(&(d_lut->member[i].dim1), &(h_lut.member[i].dim1), sizeof(int), cudaMemcpyHostToDevice); \
        cudaMemcpy(&(d_lut->member[i].dim2), &(h_lut.member[i].dim2), sizeof(int), cudaMemcpyHostToDevice); \
      }                                                                                                     \
    }                                                                                                       \
  } while (0)

#define COPY_POINTER_MEMBER(member, type, count)                                           \
  do {                                                                                     \
    type *tmp_dev;                                                                         \
    printf("tmp_dev = %p\n", (void *)tmp_dev);                                             \
    err = cudaMalloc((void **)&tmp_dev, (count) * sizeof(type));                           \
    printf("malloc tmp_dev = %p\n", (void *)tmp_dev);                                      \
    if (err != cudaSuccess) {                                                              \
      fprintf(stderr, "cudaMalloc failed for " #member ": %s\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                                  \
    }                                                                                      \
    printf("h_lut.member = %p\n", (void *)h_lut.member);                                   \
    cudaMemcpy(tmp_dev, h_lut.member, (count) * sizeof(type), cudaMemcpyHostToDevice);     \
    printf("d_lut->member");                                                               \
    printf(" = %p\n", (void *)d_lut->member);                                              \
    cudaMemcpy(&(d_lut->member), &tmp_dev, sizeof(type *), cudaMemcpyHostToDevice);        \
  } while (0)

__device__ __constant__ t_nrLDPC_lut lut384_R13;
__device__ __constant__ t_nrLDPC_lut lut384_R23;

void copy_luts_to_constant()
{
  cudaError_t err;
  t_nrLDPC_lut h_lut;
  // ---------------------------
  // copy all the member pointers
  // ---------------------------
  t_nrLDPC_lut *d_lut = &lut384_R13;
  COPY_POINTER_MEMBER(startAddrCnGroups, uint32_t, 9);
  printf("Inside copy 3\n");
  COPY_POINTER_MEMBER(numCnInCnGroups, uint8_t, 9);
  printf("Inside copy 4\n");
  printf("host ptr = %p\n", (void *)d_lut->numBnInBnGroups);
  COPY_POINTER_MEMBER(numBnInBnGroups, uint8_t, 30);
  printf("Inside copy 5\n");
  printf("host ptr = %p\n", (void *)d_lut->startAddrBnGroups);
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

  COPY_ARR_MEMBER(circShift, uint16_t, 9);
  COPY_ARR_MEMBER(startAddrBnProcBuf, uint32_t, 9);
  COPY_ARR_MEMBER(bnPosBnProcBuf, uint8_t, 9);
  COPY_ARR_MEMBER(posBnInCnProcBuf, uint8_t, 9);
}
//-----------------------------------------↓↓↓ R13 ↓↓↓----------------------------------------
__global__ void cnProcKernel_BG1_R13_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                     const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_cnOutAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     uint32_t Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int8_t *PC_Flag)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R13) // 30336 is the total processed 316 msg * 96
    return;

  uint32_t segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_PC_Flag = PC_Flag + segIdx;

  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }
  // printf("I'm inside cnProc_kernel\n");

  // const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...315
  uint32_t lane = tid % RowLength; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  uint32_t InnerOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + 384 * CnIdx;
  uint32_t idxBn = cn_bn_map_BG1_R13[row][0];
  uint32_t circShift = cn_bn_map_BG1_R13[row][1];


  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + InnerOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + InnerOffset);
  int8_t *p_bnProcBuf = (int8_t *)(d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 1:
      cnProcKernel_BG1_int8_G4(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 2:
      cnProcKernel_BG1_int8_G5(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 3:
      cnProcKernel_BG1_int8_G6(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 4:
      cnProcKernel_BG1_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
  }
}

void check_lut_pointers_cu(const t_nrLDPC_lut *lut)
{
  if (!lut) {
    printf("check_lut_pointers: lut is NULL\n");
    return;
  }

  printf("Checking LUT pointers:\n");
  printf("startAddrCnGroups       = %p\n", (void *)lut->startAddrCnGroups);
  printf("numCnInCnGroups         = %p\n", (void *)lut->numCnInCnGroups);
  printf("numBnInBnGroups         = %p\n", (void *)lut->numBnInBnGroups);
  printf("startAddrBnGroups       = %p\n", (void *)lut->startAddrBnGroups);
  printf("startAddrBnGroupsLlr    = %p\n", (void *)lut->startAddrBnGroupsLlr);
  printf("llr2llrProcBufAddr      = %p\n", (void *)lut->llr2llrProcBufAddr);
  printf("llr2llrProcBufBnPos     = %p\n", (void *)lut->llr2llrProcBufBnPos);

  printf("circShift               = %p\n", (void *)lut->circShift);
  printf("startAddrBnProcBuf       = %p\n", (void *)lut->startAddrBnProcBuf);
  printf("bnPosBnProcBuf           = %p\n", (void *)lut->bnPosBnProcBuf);
  printf("posBnInCnProcBuf         = %p\n", (void *)lut->posBnInCnProcBuf);
}

void nrLDPC_cnProc_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            uint32_t Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int8_t *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  cnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      p_lut,
      cnProcBuf,
      cnProcBufRes,
      bnProcBuf,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
  // printf("Check point 1001: ");
  CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R13_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_cnProcBuf,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     uint32_t Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int8_t *PC_Flag)
{
  uint32_t segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_PC_Flag = PC_Flag + segIdx;

  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  /*if (tid == 0) {
    printf("3: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= num_TotalThreads_BG1_R13) {
    return;
  }

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint8_t GrpIdx = lut_BnGrpIdx_BG1_R13[row];
  uint8_t MsgIdx = lut_BnMsgIdx_BG1_R13[row];
  uint8_t BnIdx = lut_BnIdx_BG1_R13[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R13[GrpIdx - 1];
  uint8_t GrpNum = d_lut_numBnInBnGroups_BG1_R13[GrpIdx - 1];
  uint16_t cirShift = bn_cn_map_BG1_R13[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R13[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);

#if 0
    if (lane == 0) {
      int prevIdxWords = ((MsgIdx - 1) * GrpNum * Zc);
      int GlobalId = row;
      if (GlobalId < 316) {
        dumpBn[GlobalId].idxBn = (int)(p_bnProcBuf_Grp - d_bnProcBuf + prevIdxWords);
        dumpBn[GlobalId].dd = 1;
      }
    }
#endif
  bnProcKernelMerge_BG1_int8_Gn(p_bnProcBuf_Grp,
                                (int8_t *)p_cnProcBuf_Grp,
                                p_llrProcBuf_Grp,
                                (int8_t *)p_llrRes_Grp,
                                lane,
                                GrpIdx,
                                MsgIdx,
                                BnIdx,
                                GrpNum,
                                cirShift,
                                Zc);
  if (tid == 0 && (*p_iter_ptr) < numMaxIter) {
    (*p_iter_ptr)++;
  }
}

void nrLDPC_bnProc_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBuf,
                                            int8_t *cnProcBuf,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            uint32_t Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int8_t *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      bnProcBuf,
      cnProcBuf,
      llrProcBuf,
      llrRes,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
}

//-----------------------------------------↑↑↑ R13 ↑↑↑----------------------------------------

//-----------------------------------------↓↓↓ R23 ↓↓↓----------------------------------------

__global__ void cnProcKernel_BG1_R23_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                     const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_cnOutAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     uint32_t Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int8_t *PC_Flag)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_TotalThreads_BG1_R23) // 13824 is the total processed 144 msg * 96
    return;
  uint32_t segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_PC_Flag = PC_Flag + segIdx;
  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }
  // printf("I'm inside cnProc_kernel\n");

  // const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...143
  uint32_t lane = tid % RowLength; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R23[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R23[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R23[row];
  // uint16_t blockSize = h_block_thread_counts_cnProc[blk];
  uint32_t inOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + 384 * CnIdx;
  uint32_t idxBn = cn_bn_map_BG1_R23[row][0];
  uint32_t circShift = cn_bn_map_BG1_R23[row][1];
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + inOffset);
  int8_t *p_bnProcBuf = (int8_t *)(d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 1:
      printf("Shouldn't see case 1 in R23");
      break;
    case 2:
      printf("Shouldn't see case 2 in R23");
      break;
    case 3:
      printf("Shouldn't see case 3 in R23");
      break;
    case 4:
      cnProcKernel_BG1_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            uint32_t Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int8_t *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  cnProcKernel_BG1_R23_int8_BIG_stream<<<Kdim_R23[CudaStreamIdx].grid, Kdim_R23[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      p_lut,
      cnProcBuf,
      cnProcBufRes,
      bnProcBuf,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
  /*
 cudaError_t err=cudaPeekAtLastError();
 if (err!=cudaSuccess) {
    printf("cuda error: %s %s)\n",cudaGetErrorString(err),__FUNCTION__);
    exit(-1);
 }
 cudaDeviceSynchronize();*/
}

__global__ void bnProcKernel_BG1_R23_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_cnProcBuf,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     uint32_t Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int8_t *PC_Flag)
{
  uint32_t segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_PC_Flag = PC_Flag + segIdx;

  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  /*if (tid == 0) {
    printf("3: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= num_TotalThreads_BG1_R23) {
    return;
  }

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint8_t GrpIdx = lut_BnGrpIdx_BG1_R23[row];
  uint8_t MsgIdx = lut_BnMsgIdx_BG1_R23[row];
  uint8_t BnIdx = lut_BnIdx_BG1_R23[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R23[GrpIdx - 1];
  uint8_t GrpNum = d_lut_numBnInBnGroups_BG1_R23[GrpIdx - 1];
  uint16_t cirShift = bn_cn_map_BG1_R23[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R23[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);

#if 0
    if (lane == 0) {
      int prevIdxWords = ((MsgIdx - 1) * GrpNum * Zc);
      int GlobalId = row;
      if (GlobalId < 316) {
        dumpBn[GlobalId].idxBn = (int)(p_bnProcBuf_Grp - d_bnProcBuf + prevIdxWords);
        dumpBn[GlobalId].dd = 1;
      }
    }
#endif
  bnProcKernelMerge_BG1_int8_Gn(p_bnProcBuf_Grp,
                                (int8_t *)p_cnProcBuf_Grp,
                                p_llrProcBuf_Grp,
                                (int8_t *)p_llrRes_Grp,
                                lane,
                                GrpIdx,
                                MsgIdx,
                                BnIdx,
                                GrpNum,
                                cirShift,
                                Zc);
  if (tid == 0 && (*p_iter_ptr) < numMaxIter) {
    (*p_iter_ptr)++;
  }
}

void nrLDPC_bnProc_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBuf,
                                            int8_t *cnProcBuf,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            uint32_t Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int8_t *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R23_int8_BIG_stream<<<Kdim_R23[CudaStreamIdx].grid, Kdim_R23[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      bnProcBuf,
      cnProcBuf,
      llrProcBuf,
      llrRes,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
  // cudaDeviceSynchronize();
}

//-----------------------------------------↑↑↑ R23 ↑↑↑----------------------------------------
//-------------------------------------↓↓↓ general R ↓↓↓----------------------------------------
__global__ void llrPreProc_Kernel_BG1_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                          int8_t *__restrict__ d_llr,
                                                          int8_t *__restrict__ d_llrProcBuf,
                                                          int8_t *__restrict__ d_cnProcBuf,
                                                          uint32_t Zc,
                                                          uint8_t BG)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t segIdx = blockIdx.y;

  if (tid >= num_TotalThreads_BG1_R13) // 30336 is the total processed 316 msg * 96
    return;

  // const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...315
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  uint32_t InnerOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + Zc * CnIdx;

  int8_t *p_cnProcBuf = (int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + InnerOffset);
  int8_t *p_llr = (int8_t *)(d_llr + segIdx * 68 * Zc);
  int8_t *p_llrProcBuf = (int8_t *)(d_llrProcBuf + segIdx * NR_LDPC_MAX_NUM_LLR);

  switch (groupIdx) {
    case 0:
      llrPreProc_Kernel_BG1_int8_G3_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 1:
      llrPreProc_Kernel_BG1_int8_G4_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 2:
      llrPreProc_Kernel_BG1_int8_G5_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 3:
      llrPreProc_Kernel_BG1_int8_G6_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 4:
      llrPreProc_Kernel_BG1_int8_G7_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 5:
      llrPreProc_Kernel_BG1_int8_G8_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 6:
      llrPreProc_Kernel_BG1_int8_G9_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 7:
      llrPreProc_Kernel_BG1_int8_G10_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
    case 8:
      llrPreProc_Kernel_BG1_int8_G19_stream(p_lut, p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc, BG);
      break;
  }
}

void nrLDPC_llrPreProc_BG1_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                                int8_t *llr,
                                                int8_t *llrProcBuf,
                                                int8_t *cnProcBuf,
                                                uint32_t Z,
                                                uint8_t BG,
                                                cudaStream_t *streams,
                                                int8_t CudaStreamIdx)
{
  llrPreProc_Kernel_BG1_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid,
                                              Kdim_R13[CudaStreamIdx].block,
                                              0,
                                              streams[CudaStreamIdx]>>>(p_lut, llr, llrProcBuf, cnProcBuf, Z, BG);
  // printf("Check point 1001: ");
  CHECK(cudaGetLastError());
}

__global__ void llrRes2llrOut_Kernel_BG1_int8_BIG_stream(uint8_t R,
                                                         int8_t *d_llrRes,
                                                         uint32_t Zc,
                                                         int8_t *iter_ptr,
                                                         int8_t numMaxIter,
                                                         int8_t *PC_Flag,
                                                         e_nrLDPC_outMode outMode,
                                                         int8_t *p_llrOut,
                                                         uint32_t numLLR)
{
  uint32_t segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_PC_Flag = PC_Flag + segIdx;

  int8_t *p_llrRes = (int8_t *)(d_llrRes + segIdx * NR_LDPC_MAX_NUM_LLR);
  int8_t *p_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? p_llrOut + segIdx * 8448 : p_llrOut + segIdx * NR_LDPC_MAX_NUM_LLR;

  if (*p_iter_ptr == numMaxIter) { // output
    llrRes2llrOut_Kernel_BG1_int8(R, p_p_llrOut, p_llrRes, Zc);
  }
}

__global__ void OutPut_Kernel_BG1_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                  uint32_t Zc,
                                                  int8_t *iter_ptr,
                                                  int8_t numMaxIter,
                                                  int8_t *PC_Flag,
                                                  e_nrLDPC_outMode outMode,
                                                  int8_t *p_out,
                                                  int8_t *llrOut,
                                                  int8_t *p_llrOut,
                                                  uint32_t numLLR)
{
  // only activate in the last iteration
  uint32_t segIdx = blockIdx.y;

  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int8_t *p_p_out = p_out + segIdx * 8448;
  int8_t *p_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? p_llrOut + segIdx * 8448 : p_llrOut + segIdx * NR_LDPC_MAX_NUM_LLR;

  if (*iter_ptr == numMaxIter) {
    if (outMode == nrLDPC_outMode_BIT)
      llr2bitPacked_Kernel_BG1_int8((uint8_t *)p_p_out, p_p_llrOut, numLLR);

    else // if (outMode == nrLDPC_outMode_BITINT8)
      llr2bit_Kernel_BG1_int8((uint8_t *)p_p_out, p_p_llrOut, numLLR);
  } else
    return;
}

void nrLDPC_OutPut_BG1_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                        int8_t *llrRes,
                                        uint32_t Z,
                                        uint8_t R,
                                        int8_t *iter_ptr,
                                        int8_t numMaxIter,
                                        int8_t *PC_Flag,
                                        e_nrLDPC_outMode outMode,
                                        int8_t *p_out,
                                        int8_t *llrOut,
                                        int8_t *p_llrOut,
                                        uint32_t numLLR,
                                        cudaStream_t *streams,
                                        int8_t CudaStreamIdx)
{
  llrRes2llrOut_Kernel_BG1_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid,
                                             Kdim_R13[CudaStreamIdx].block,
                                             0,
                                             streams[CudaStreamIdx]>>>(R,
                                                                       llrRes,
                                                                       Z,
                                                                       iter_ptr,
                                                                       numMaxIter,
                                                                       PC_Flag,
                                                                       outMode,
                                                                       p_llrOut,
                                                                       numLLR);

  OutPut_Kernel_BG1_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      p_lut,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag,
      outMode,
      p_out,
      llrOut,
      p_llrOut,
      numLLR);
  /*
 cudaError_t err=cudaPeekAtLastError();
 if (err!=cudaSuccess) {
    printf("cuda error: %s %s)\n",cudaGetErrorString(err),__FUNCTION__);
    exit(-1);
 }
 cudaDeviceSynchronize();*/
}
//---------------------------------↑↑↑ general R ↑↑↑----------------------------------------

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//-----------------------CUDA Scheduler Area------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------

extern "C" void nrLDPC_decoder_scheduler_BG1_cuda_core(const t_nrLDPC_lut *p_lut,
                                                       int8_t *p_out,
                                                       uint32_t numLLR,
                                                       int8_t *llr,
                                                       int8_t *cnProcBuf,
                                                       int8_t *cnProcBufRes,
                                                       int8_t *bnProcBuf,
                                                       int8_t *bnProcBufRes,
                                                       int8_t *llrRes,
                                                       int8_t *llrProcBuf,
                                                       int8_t *llrOut,
                                                       int8_t *p_llrOut,
                                                       uint32_t Z,
                                                       uint8_t BG,
                                                       uint8_t R,
                                                       uint8_t numMaxIter,
                                                       e_nrLDPC_outMode outMode,
                                                       cudaStream_t *streams,
                                                       uint8_t CudaStreamIdx,
                                                       cudaEvent_t *doneEvent,
                                                       int8_t *iter_ptr,
                                                       int8_t *PC_Flag)
{
  cudaStream_t stream = streams[CudaStreamIdx];

  if (!graphCreated[CudaStreamIdx]) {
#if RECORD_GRAPH
    printf("Creating the graph for stream %d, format R%d\n", CudaStreamIdx, R);

    if (CudaStreamIdx != 0) {
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }
#endif
    // Start graph recording
#if RECORD_GRAPH
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
#endif
    Kdim_R13[CudaStreamIdx].block = dim3(BG1_R13_threadSize.NumThreads, 1, 1);
    Kdim_R13[CudaStreamIdx].grid = dim3(BG1_R13_threadSize.NumBlocks, segmentPacks[CudaStreamIdx].nSeg, 1);
    Kdim_R23[CudaStreamIdx].block = dim3(BG1_R23_threadSize.NumThreads, 1, 1);
    Kdim_R23[CudaStreamIdx].grid = dim3(BG1_R23_threadSize.NumBlocks, segmentPacks[CudaStreamIdx].nSeg, 1);

    nrLDPC_llrPreProc_BG1_cuda_stream_core(p_lut, llr, llrProcBuf, cnProcBuf, Z, BG, streams, CudaStreamIdx);

    switch (R) {
      case 13: {
        for (int i = 0; i <= numMaxIter; i++) {
          // printf("I'm inside the loop i = %d\n", i);
          nrLDPC_cnProc_BG1_R13_cuda_stream_core(p_lut,
                                                 cnProcBuf,
                                                 cnProcBufRes,
                                                 bnProcBuf,
                                                 Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          //          CHECK(cudaGetLastError());
          // cd cudaDeviceSynchronize();
          /*cudaDeviceSynchronize();
          if(i == 0){
            dumpAssCUDA(cnProcBuf, "Dump_cnProcBuf_cuda.txt");
            dumpAssCUDA(cnProcBufRes, "Dump_cnProcBufRes_cuda.txt");
            dumpAssCUDA(bnProcBuf, "Dump_bnProcBuf_cuda.txt");
          }
*/
          // printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          nrLDPC_bnProc_BG1_R13_cuda_stream_core(p_lut,
                                                 bnProcBuf,
                                                 cnProcBuf,
                                                 llrProcBuf,
                                                 llrRes,
                                                 Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          // cudaDeviceSynchronize();
          /*
                  if(i == 0){
                    cudaDeviceSynchronize();
                    dumpAssCUDA(cnProcBuf, "Dump_cnProcBuf_de_Bn_cuda.txt");
                    dumpAssCUDA(llrRes, "Dump_llrRes_cuda.txt");
                  }*/
          // printf("In stream %d 2: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          //        CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();

          //      CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();
          // printf("In stream %d 3: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
        }

      } break;
      case 23: {
        for (int i = 0; i <= numMaxIter; i++) {
          // printf("I'm inside the loop i = %d\n", i);
          nrLDPC_cnProc_BG1_R23_cuda_stream_core(p_lut,
                                                 cnProcBuf,
                                                 cnProcBufRes,
                                                 bnProcBuf,
                                                 Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          CHECK(cudaGetLastError());

          /*
          if(i == 0){
            cudaDeviceSynchronize();
            dumpAssCUDA(cnProcBuf, "Dump_cnProcBuf_cuda_R23.txt");
            dumpAssCUDA(cnProcBufRes, "Dump_cnProcBufRes_cuda_R23.txt");
            dumpAssCUDA(bnProcBuf, "Dump_bnProcBuf_cuda_R23.txt");
          }*/

          // printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          nrLDPC_bnProc_BG1_R23_cuda_stream_core(p_lut,
                                                 bnProcBuf,
                                                 cnProcBuf,
                                                 llrProcBuf,
                                                 llrRes,
                                                 Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);

          /*
                   if(i == 0){
                     cudaDeviceSynchronize();
                     dumpAssCUDA(bnProcBufRes, "Dump_bnProcBufRes_cuda_R23.txt");
                     dumpAssCUDA(llrRes, "Dump_llrRes_cuda_R23.txt");
                   }
                    */
          CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();

          // cudaDeviceSynchronize();
        }

      } break;

      default:
        printf("Format not support yet\n");
        break;
    }
    nrLDPC_OutPut_BG1_cuda_stream_core(p_lut,
                                       llrRes,
                                       Z,
                                       R,
                                       iter_ptr,
                                       numMaxIter,
                                       PC_Flag,
                                       outMode,
                                       p_out,
                                       llrOut,
                                       p_llrOut,
                                       numLLR,
                                       streams,
                                       CudaStreamIdx);
    /*
    {
    DumpEntry h_dumpBuf[316];
    cudaMemcpyFromSymbol(h_dumpBuf, dumpBuf, sizeof(h_dumpBuf), 0, cudaMemcpyDeviceToHost);
    FILE *fp = fopen("Dump_CnToBn_index.txt", "w");
    for (int i = 0; i < 316; ++i) {
      fprintf(fp,
              "GlobalId=%d, idxBn=%d, idxCn=%d, circShift = %d, dd = %d\n",
              i,
              h_dumpBuf[i].idxBn,
              h_dumpBuf[i].idxCn,
              h_dumpBuf[i].circShift,
              h_dumpBuf[i].dd);
    }
}
{
DumpEntry h_dumpBn[316];
    cudaMemcpyFromSymbol(h_dumpBn, dumpBn, sizeof(h_dumpBn), 0, cudaMemcpyDeviceToHost);
    FILE *fp = fopen("Dump_Bn_index.txt", "w");
    for (int i = 0; i < 316; ++i) {
      fprintf(fp,
              "GlobalId=%d, idxBn=%d, dd = %d\n",
              i,
              h_dumpBn[i].idxBn,
              h_dumpBn[i].dd);
    }
}*/
#if RECORD_GRAPH
    // stop recording
    cudaStreamEndCapture(stream, &decoderGraphs[CudaStreamIdx]);
    cudaGraphInstantiate(&decoderGraphExec[CudaStreamIdx], decoderGraphs[CudaStreamIdx], NULL, NULL, 0);
    graphCreated[CudaStreamIdx] = true;

    // Execute （make sure the first trial finish）
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
#endif
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);

  } else {
    //  reuse the graph after
    if (CudaStreamIdx != 0) {
      // uncomment below if you want streams works in sequence
      cudaStreamWaitEvent(streams[CudaStreamIdx], doneEvent[CudaStreamIdx - 1], 0);
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    //
  }
}

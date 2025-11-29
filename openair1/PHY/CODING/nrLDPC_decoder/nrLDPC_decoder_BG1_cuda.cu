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
#define STREAM_SEQUENCE 1 // default 1, set 0 different streams will work in parellel(not recommended)

#ifndef JETSON_TARGET
#define CUDA_THREADS 1024
#define CUDA_BLOCKS_R13 30
#define CUDA_BLOCKS_R23 14
#else
#define CUDA_THREADS 128
#define CUDA_BLOCKS_R13 237 // ceil(30336/128)
#define CUDA_BLOCKS_R23 108 // ceil(13824/128)
#endif

cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL] = {false};

SegmentPack segmentPacks[MAX_NUM_DLSCH_SEGMENTS_DL];

KernelLaunchConfig Kdim_R13[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
KernelLaunchConfig Kdim_R23[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
KernelLaunchConfig Kdim_llr[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
KernelLaunchConfig Kdim_output[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
ThreadSize BG1_R13_threadSize, BG1_R23_threadSize, R_general_threadSize;

// debug function
void dumpAssCUDA(const int8_t *cnProcBufRes, const char *filename)
{
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }

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

  COPY_POINTER_MEMBER(numCnInCnGroups, uint8_t, 9);

  printf("host ptr = %p\n", (void *)d_lut->numBnInBnGroups);
  COPY_POINTER_MEMBER(numBnInBnGroups, uint8_t, 30);

  printf("host ptr = %p\n", (void *)d_lut->startAddrBnGroups);

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

//-----------------------------------------↓↓↓ R13 ↓↓↓----------------------------------------
__global__ void cnProcKernel_BG1_R13_int8_BIG_stream(const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     uint32_t Zc)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R13) // 30336 is the total processed 316 msg * 96
    return;
  uint32_t segIdx = blockIdx.y;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...315
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  uint32_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint32_t MsgIdx = lut_CnMsgIdx_BG1_R13[row] - 1;
  uint32_t InnerOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + 384 * CnIdx;
  uint32_t idxBn = cn_bn_map_BG1_R13[row][0];
  uint32_t circShift = cn_bn_map_BG1_R13[row][1];

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + InnerOffset);
  int8_t *p_bnProcBuf = (int8_t *)(d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 1:
      cnProcKernel_BG1_int8_G4(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 2:
      cnProcKernel_BG1_int8_G5(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 3:
      cnProcKernel_BG1_int8_G6(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 4:
      cnProcKernel_BG1_int8_G7(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_R13_cuda_stream_core(int8_t *cnProcBuf,
                                            int8_t *bnProcBuf,
                                            uint32_t Z,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  cnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      cnProcBuf,
      bnProcBuf,
      Z);

  CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R13_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_cnProcBuf,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     uint32_t Zc)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R13) {
    return;
  }

  uint32_t segIdx = blockIdx.y;

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t GrpIdx = lut_BnGrpIdx_BG1_R13[row];
  uint32_t MsgIdx = lut_BnMsgIdx_BG1_R13[row] - 1;
  uint32_t BnIdx = lut_BnIdx_BG1_R13[row];
  uint32_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R13[GrpIdx - 1];
  uint32_t GrpNum = d_lut_numBnInBnGroups_BG1_R13[GrpIdx - 1];
  uint32_t cirShift = bn_cn_map_BG1_R13[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R13[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);

  bnProcKernel_BG1_int8_Gn(p_bnProcBuf_Grp,
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
}

void nrLDPC_bnProc_BG1_R13_cuda_stream_core(int8_t *bnProcBuf,
                                            int8_t *cnProcBuf,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            uint32_t Z,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      bnProcBuf,
      cnProcBuf,
      llrProcBuf,
      llrRes,
      Z);
  CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R13_int8_BIG_stream_last(const int8_t *__restrict__ d_bnProcBuf,
                                                          int8_t *__restrict__ d_cnProcBuf,
                                                          int8_t *__restrict__ d_llrProcBuf,
                                                          int8_t *__restrict__ d_llrRes,
                                                          uint32_t Zc)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R13) {
    return;
  }

  uint32_t segIdx = blockIdx.y;

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t GrpIdx = lut_BnGrpIdx_BG1_R13[row];
  uint32_t MsgIdx = lut_BnMsgIdx_BG1_R13[row] - 1;
  uint32_t BnIdx = lut_BnIdx_BG1_R13[row];
  uint32_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R13[GrpIdx - 1];
  uint32_t GrpNum = d_lut_numBnInBnGroups_BG1_R13[GrpIdx - 1];
  uint32_t cirShift = bn_cn_map_BG1_R13[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R13[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R13[BnToAddrIdx - 1]);

  bnProcKernel_BG1_int8_Gn_last(p_bnProcBuf_Grp,
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
}

void nrLDPC_bnProc_BG1_R13_cuda_stream_core_last(int8_t *bnProcBuf,
                                                 int8_t *cnProcBuf,
                                                 int8_t *llrProcBuf,
                                                 int8_t *llrRes,
                                                 uint32_t Z,
                                                 cudaStream_t *streams,
                                                 int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R13_int8_BIG_stream_last<<<Kdim_R13[CudaStreamIdx].grid,
                                              Kdim_R13[CudaStreamIdx].block,
                                              0,
                                              streams[CudaStreamIdx]>>>(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z);
  CHECK(cudaGetLastError());
}

//-----------------------------------------↑↑↑ R13 ↑↑↑----------------------------------------

//-----------------------------------------↓↓↓ R23 ↓↓↓----------------------------------------

__global__ void cnProcKernel_BG1_R23_int8_BIG_stream(const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     uint32_t Zc)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R23) // 13824 is the total processed 144 msg * 96
    return;

  uint32_t segIdx = blockIdx.y;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...143
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t groupIdx = lut_CnGrpIdx_BG1_R23[row] - 1;
  uint32_t CnIdx = lut_CnIdx_BG1_R23[row] - 1;
  uint32_t MsgIdx = lut_CnMsgIdx_BG1_R23[row] - 1;
  uint32_t inOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + 384 * CnIdx;
  uint32_t idxBn = cn_bn_map_BG1_R23[row][0];
  uint32_t circShift = cn_bn_map_BG1_R23[row][1];

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + inOffset);
  int8_t *p_bnProcBuf = (int8_t *)(d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
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
      cnProcKernel_BG1_int8_G7(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_cnProcBuf, p_bnProcBuf, MsgIdx, lane, idxBn, circShift, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_R23_cuda_stream_core(int8_t *cnProcBuf,
                                            int8_t *bnProcBuf,
                                            uint32_t Z,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  cnProcKernel_BG1_R23_int8_BIG_stream<<<Kdim_R23[CudaStreamIdx].grid, Kdim_R23[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      cnProcBuf,
      bnProcBuf,
      Z);
  CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R23_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_cnProcBuf,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     uint32_t Zc)
{
  uint32_t segIdx = blockIdx.y;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R23) {
    return;
  }

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t GrpIdx = lut_BnGrpIdx_BG1_R23[row];
  uint32_t MsgIdx = lut_BnMsgIdx_BG1_R23[row] - 1;
  uint32_t BnIdx = lut_BnIdx_BG1_R23[row];
  uint32_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R23[GrpIdx - 1];
  uint32_t GrpNum = d_lut_numBnInBnGroups_BG1_R23[GrpIdx - 1];
  uint32_t cirShift = bn_cn_map_BG1_R23[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R23[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);

  bnProcKernel_BG1_int8_Gn(p_bnProcBuf_Grp,
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
}

void nrLDPC_bnProc_BG1_R23_cuda_stream_core(int8_t *bnProcBuf,
                                            int8_t *cnProcBuf,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            uint32_t Z,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R23_int8_BIG_stream<<<Kdim_R23[CudaStreamIdx].grid, Kdim_R23[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      bnProcBuf,
      cnProcBuf,
      llrProcBuf,
      llrRes,
      Z);
  CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R23_int8_BIG_stream_last(const int8_t *__restrict__ d_bnProcBuf,
                                                          int8_t *__restrict__ d_cnProcBuf,
                                                          int8_t *__restrict__ d_llrProcBuf,
                                                          int8_t *__restrict__ d_llrRes,
                                                          uint32_t Zc)
{
  uint32_t segIdx = blockIdx.y;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R23) {
    return;
  }

  uint32_t row = tid / RowLength; // to decide the inner block
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t GrpIdx = lut_BnGrpIdx_BG1_R23[row];
  uint32_t MsgIdx = lut_BnMsgIdx_BG1_R23[row] - 1;
  uint32_t BnIdx = lut_BnIdx_BG1_R23[row];
  uint32_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R23[GrpIdx - 1];
  uint32_t GrpNum = d_lut_numBnInBnGroups_BG1_R23[GrpIdx - 1];
  uint32_t cirShift = bn_cn_map_BG1_R23[row][1];
  const uint32_t baseBn = (BnIdx - 1) * Zc;

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + baseBn + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + d_lut_startAddrBnGroups_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_cnProcBuf_Grp = (const int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + bn_cn_map_BG1_R23[row][0]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp =
      (const int8_t *)(d_llrRes + baseBn + segIdx * NR_LDPC_MAX_NUM_LLR + d_lut_startAddrBnGroupsLlr_BG1_R23[BnToAddrIdx - 1]);

  bnProcKernel_BG1_int8_Gn_last(p_bnProcBuf_Grp,
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
}

void nrLDPC_bnProc_BG1_R23_cuda_stream_core_last(int8_t *bnProcBuf,
                                                 int8_t *cnProcBuf,
                                                 int8_t *llrProcBuf,
                                                 int8_t *llrRes,
                                                 uint32_t Z,
                                                 cudaStream_t *streams,
                                                 int8_t CudaStreamIdx)
{
  bnProcKernel_BG1_R23_int8_BIG_stream_last<<<Kdim_R23[CudaStreamIdx].grid,
                                              Kdim_R23[CudaStreamIdx].block,
                                              0,
                                              streams[CudaStreamIdx]>>>(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z);
  CHECK(cudaGetLastError());
}
//-----------------------------------------↑↑↑ R23 ↑↑↑----------------------------------------
//-------------------------------------↓↓↓ general R ↓↓↓----------------------------------------
__global__ void llrPreProc_Kernel_BG1_int8_BIG_stream(int8_t *__restrict__ d_llr,
                                                      int8_t *__restrict__ d_llrProcBuf,
                                                      int8_t *__restrict__ d_cnProcBuf,
                                                      uint32_t Zc,
                                                      uint32_t R)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num_TotalThreads_BG1_R13) // 30336 is the total processed 316 msg * 96
    return;

  uint32_t segIdx = blockIdx.y;

  uint32_t row = tid / RowLength; // to decide the global MsgIdx; row = 0,1,2...315
  uint32_t lane = tid % RowLength; // to decide the inner lane

  uint32_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  uint32_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint32_t MsgIdx = lut_CnMsgIdx_BG1_R13[row] - 1;
  uint32_t InnerOffset = d_lut_startAddrCnGroups_BG1[groupIdx] + Zc * CnIdx;
  uint32_t idxBn = llr_cn_preProc_map_BG1_R13[row][0];
  uint32_t circShift = llr_cn_preProc_map_BG1_R13[row][1];

  int8_t *p_cnProcBuf = (int8_t *)(d_cnProcBuf + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + InnerOffset);
  int8_t *p_llr = (int8_t *)(d_llr + segIdx * 68 * Zc);
  int8_t *p_llrProcBuf = (int8_t *)(d_llrProcBuf + segIdx * NR_LDPC_MAX_NUM_LLR);

  llrPreProc_Kernel_BG1_int8_Gn_stream(p_llr, p_llrProcBuf, p_cnProcBuf, MsgIdx, lane, idxBn, groupIdx, circShift, Zc, R);
}

void nrLDPC_llrPreProc_BG1_cuda_stream_core(int8_t *llr,
                                            int8_t *llrProcBuf,
                                            int8_t *cnProcBuf,
                                            uint32_t Z,
                                            uint32_t R,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  llrPreProc_Kernel_BG1_int8_BIG_stream<<<Kdim_R13[CudaStreamIdx].grid, Kdim_R13[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      llr,
      llrProcBuf,
      cnProcBuf,
      Z,
      R);

  CHECK(cudaGetLastError());
}

__global__ void llrOutPut_Kernel_BG1_int8_BIG_stream(uint32_t R,
                                                     int8_t *d_llrRes,
                                                     uint32_t Zc,
                                                     e_nrLDPC_outMode outMode,
                                                     int8_t *d_out,
                                                     uint32_t numLLR)
{
  uint32_t segIdx = blockIdx.y;

  int8_t *p_out = d_out + segIdx * 8448;
  int8_t *p_llrRes = (int8_t *)(d_llrRes + segIdx * NR_LDPC_MAX_NUM_LLR);
  // output
  if (outMode == nrLDPC_outMode_BIT)
    llr2bitPacked_Kernel_BG1_int8(R, (uint8_t *)p_out, p_llrRes, numLLR, Zc);

  else if (outMode == nrLDPC_outMode_BITINT8)
    llr2bit_Kernel_BG1_int8(R, (uint8_t *)p_out, p_llrRes, numLLR, Zc);
}

void nrLDPC_OutPut_BG1_cuda_stream_core(int8_t *llrRes,
                                        uint32_t Z,
                                        uint8_t R,
                                        e_nrLDPC_outMode outMode,
                                        int8_t *p_out,
                                        uint32_t numLLR,
                                        cudaStream_t *streams,
                                        int8_t CudaStreamIdx)
{
  llrOutPut_Kernel_BG1_int8_BIG_stream<<<Kdim_llr[CudaStreamIdx].grid, Kdim_llr[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      R,
      llrRes,
      Z,
      outMode,
      p_out,
      numLLR);

  // OutPut_Kernel_BG1_int8_BIG_stream<<<Kdim_output[CudaStreamIdx].grid,
  //                                    Kdim_output[CudaStreamIdx].block,
  //                                    0,
  //                                    streams[CudaStreamIdx]>>>(Z, outMode, p_out, llrOut, p_llrOut, numLLR);
  CHECK(cudaGetLastError());
}
//---------------------------------↑↑↑ general R ↑↑↑----------------------------------------

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//-----------------------CUDA Scheduler Area------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------

extern "C" void nrLDPC_decoder_scheduler_BG1_cuda_core(int8_t *p_out,
                                                       uint32_t numLLR,
                                                       int8_t *llr,
                                                       int8_t *cnProcBuf,
                                                       int8_t *bnProcBuf,
                                                       int8_t *llrRes,
                                                       int8_t *llrProcBuf,
                                                       uint32_t Z,
                                                       uint8_t BG,
                                                       uint8_t R,
                                                       uint8_t numMaxIter,
                                                       e_nrLDPC_outMode outMode,
                                                       cudaStream_t *streams,
                                                       uint8_t CudaStreamIdx,
                                                       cudaEvent_t *doneEvent)
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
    Kdim_llr[CudaStreamIdx].block = dim3(R_general_threadSize.NumThreads, 1, 1);
    Kdim_llr[CudaStreamIdx].grid = dim3(R_general_threadSize.NumBlocks_llr, segmentPacks[CudaStreamIdx].nSeg, 1);
    Kdim_output[CudaStreamIdx].block = dim3(R_general_threadSize.NumThreads, 1, 1);
    Kdim_output[CudaStreamIdx].grid = dim3(R_general_threadSize.NumBlocks_output, segmentPacks[CudaStreamIdx].nSeg, 1);
    // decoding starts here
    nrLDPC_llrPreProc_BG1_cuda_stream_core(llr, llrProcBuf, cnProcBuf, Z, R, streams, CudaStreamIdx);

    switch (R) {
      case 13: {
        for (int i = 0; i <= numMaxIter; i++) {
          nrLDPC_cnProc_BG1_R13_cuda_stream_core(cnProcBuf, bnProcBuf, Z, streams, CudaStreamIdx);
          if(i == 0){
            cudaDeviceSynchronize();
            dumpAssCUDA(bnProcBuf,"First_iter_bnProc_Dump_cuda.txt");
          }

          if (i == numMaxIter)
            nrLDPC_bnProc_BG1_R13_cuda_stream_core_last(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z, streams, CudaStreamIdx);
          else
            nrLDPC_bnProc_BG1_R13_cuda_stream_core(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z, streams, CudaStreamIdx);
        }

      } break;
      case 23: {
        for (int i = 0; i <= numMaxIter; i++) {
          nrLDPC_cnProc_BG1_R23_cuda_stream_core(cnProcBuf, bnProcBuf, Z, streams, CudaStreamIdx);
          if (i == numMaxIter)
            nrLDPC_bnProc_BG1_R23_cuda_stream_core_last(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z, streams, CudaStreamIdx);
          else
            nrLDPC_bnProc_BG1_R23_cuda_stream_core(bnProcBuf, cnProcBuf, llrProcBuf, llrRes, Z, streams, CudaStreamIdx);
        }

      } break;

      default:
        printf("Format not support yet\n");
        break;
    }

    nrLDPC_OutPut_BG1_cuda_stream_core(llrRes, Z, R, outMode, p_out, numLLR, streams, CudaStreamIdx);
    {
            cudaDeviceSynchronize();
            dumpAssCUDA(p_out,"Dump_p_out_cuda.txt");
          }
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
#if STREAM_SEQUENCE
    if (CudaStreamIdx != 0) {
      cudaStreamWaitEvent(streams[CudaStreamIdx], doneEvent[CudaStreamIdx - 1], 0);
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }
#endif
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    //
  }
}

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

#include "nrLDPC_CUDA_lut.h"
#include "nrLDPC_CUDA_CnProcKernel_BG1.h"
#include "nrLDPC_CUDA_BnProcKernel_BG1.h"
#include "nrLDPC_CUDA_BnToCnPC_Kernel_BG1.h"
#include "decoder_graphs.h"

#define ZC 384 // for BG1 test only
#define RECORD_GRAPH 1 // set 1 to enable graph recording, 0 to unable

cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL] = {false};

SegmentPack segmentPacks[MAX_NUM_DLSCH_SEGMENTS_DL];

KernelLaunchConfig Kdim[MAX_NUM_DLSCH_SEGMENTS_DL / 8];
/* debug function
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
*/

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

//-----------------------------------------↓↓↓ R13 ↓↓↓----------------------------------------

__global__ void cnProcKernel_BG1_R13_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                     const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_cnOutAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     int Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int *PC_Flag)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int *p_PC_Flag = PC_Flag + segIdx;

  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }
  // printf("I'm inside cnProc_kernel\n");

  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...315
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  // uint16_t blockSize = h_block_thread_counts_cnProc[blk];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 30336) // 30336 is the total processed 316 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF;

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 1:
      cnProcKernel_BG1_int8_G4(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 2:
      cnProcKernel_BG1_int8_G5(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 3:
      cnProcKernel_BG1_int8_G6(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 4:
      cnProcKernel_BG1_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  cnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim[CudaStreamIdx].grid, Kdim[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      p_lut,
      cnProcBuf,
      cnProcBufRes,
      bnProcBuf,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
  // printf("Check point 1001: ");
  // CHECK(cudaGetLastError());
}

__global__ void bnProcKernel_BG1_R13_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_bnProcBufRes,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     const uint8_t *lut_numBnInBnGroups,
                                                     const uint32_t *lut_startAddrBnBuf,
                                                     const uint16_t *lut_startAddrBnLlr,
                                                     int Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int *PC_Flag)
{
  int segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int *p_PC_Flag = PC_Flag + segIdx;

  // Early stopping
  if (*p_iter_ptr > numMaxIter || *p_PC_Flag == 0) {
    return;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  /*if (tid == 0) {
    printf("3: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= 30336) {
    return;
  }

  int row = tid / 96; // to decide the inner block
  int lane = tid % 96; // to decide the inner lane

  uint8_t GrpIdx = lut_BnGrpIdx_BG1_R13[row];
  uint8_t MsgIdx = lut_BnMsgIdx_BG1_R13[row];
  uint8_t BnIdx = lut_BnIdx_BG1_R13[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R13[GrpIdx - 1];
  uint8_t GrpNum = lut_numBnInBnGroups[GrpIdx - 1];

  const int8_t *p_bnProcBuf_Grp =
      (const int8_t *)(d_bnProcBuf + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_bnProcBufRes_Grp =
      (const int8_t *)(d_bnProcBufRes + segIdx * NR_LDPC_SIZE_BN_PROC_BUF + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_llrProcBuf_Grp =
      (const int8_t *)(d_llrProcBuf + segIdx * NR_LDPC_MAX_NUM_LLR + lut_startAddrBnLlr[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp = (const int8_t *)(d_llrRes + segIdx * NR_LDPC_MAX_NUM_LLR + lut_startAddrBnLlr[BnToAddrIdx - 1]);

  bnProcKernelMerge_BG1_int8_Gn(p_bnProcBuf_Grp,
                                (int8_t *)p_bnProcBufRes_Grp,
                                p_llrProcBuf_Grp,
                                (int8_t *)p_llrRes_Grp,
                                lane,
                                GrpIdx,
                                MsgIdx,
                                BnIdx,
                                GrpNum,
                                Zc);
}

void nrLDPC_bnProc_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBuf,
                                            int8_t *bnProcBufRes,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  const uint8_t *lut_numBnInBnGroups;
  const uint32_t *lut_startAddrBnGroups;
  const uint16_t *lut_startAddrBnGroupsLlr;

  lut_numBnInBnGroups = p_lut->numBnInBnGroups;
  lut_startAddrBnGroups = p_lut->startAddrBnGroups;
  lut_startAddrBnGroupsLlr = p_lut->startAddrBnGroupsLlr;

  bnProcKernel_BG1_R13_int8_BIG_stream<<<Kdim[CudaStreamIdx].grid, Kdim[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      bnProcBuf,
      bnProcBufRes,
      llrProcBuf,
      llrRes,
      lut_numBnInBnGroups,
      lut_startAddrBnGroups,
      lut_startAddrBnGroupsLlr,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag);
}

__global__ void BnToCnPC_Kernel_BG1_R13_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                        int8_t *__restrict__ d_bnOutAll,
                                                        const int8_t *__restrict__ d_cnBufAll,
                                                        int8_t *__restrict__ d_cnOutAll,
                                                        int8_t *__restrict__ d_bnBufAll,
                                                        int8_t *d_llrRes,
                                                        int Zc,
                                                        int8_t *iter_ptr,
                                                        int8_t numMaxIter,
                                                        int *PC_Flag,
                                                        e_nrLDPC_outMode outMode,
                                                        int8_t *p_out,
                                                        int8_t *llrOut,
                                                        int8_t *p_llrOut,
                                                        uint32_t numLLR)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int segIdx = blockIdx.y;
  int8_t *p_iter_ptr = iter_ptr + segIdx;
  int *p_PC_Flag = PC_Flag + segIdx;

  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...315
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 30336) // 30336 is the total processed 316 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + segIdx * NR_LDPC_SIZE_CN_PROC_BUF + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)(d_bnBufAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);
  int8_t *p_bnProcBufRes = (int8_t *)(d_bnOutAll + segIdx * NR_LDPC_SIZE_BN_PROC_BUF);
  int8_t *p_llrRes = (int8_t *)(d_llrRes + segIdx * NR_LDPC_MAX_NUM_LLR);
  int8_t *p_p_llrOut = (outMode == nrLDPC_outMode_LLRINT8) ? p_llrOut + segIdx * 8448 : p_llrOut + segIdx * NR_LDPC_MAX_NUM_LLR;

  // Early stopping
  if (!(*p_iter_ptr > numMaxIter - 1 || *p_PC_Flag == 0)) {
    if (tid == 0) {
      *p_PC_Flag = 0;
      // printf("4: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
    }
    //__syncthreads();

    // uint32_t pcRes = 0; // setting flag for Parity Check

    switch (groupIdx) {
      case 0:
        CnToBnPC_Kernel_BG1_int8_G3_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 1:
        CnToBnPC_Kernel_BG1_int8_G4_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 2:
        CnToBnPC_Kernel_BG1_int8_G5_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 3:
        CnToBnPC_Kernel_BG1_int8_G6_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 4:
        CnToBnPC_Kernel_BG1_int8_G7_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 5:
        CnToBnPC_Kernel_BG1_int8_G8_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 6:
        CnToBnPC_Kernel_BG1_int8_G9_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           p_PC_Flag);
        break;
      case 7:
        CnToBnPC_Kernel_BG1_int8_G10_Stream(p_lut,
                                            p_bnProcBufRes,
                                            p_cnProcBuf,
                                            p_cnProcBufRes,
                                            p_bnProcBuf,
                                            MsgIdx,
                                            lane,
                                            groupIdx,
                                            CnIdx,
                                            Zc,
                                            p_PC_Flag);
        break;
      case 8:
        CnToBnPC_Kernel_BG1_int8_G19_Stream(p_lut,
                                            p_bnProcBufRes,
                                            p_cnProcBuf,
                                            p_cnProcBufRes,
                                            p_bnProcBuf,
                                            MsgIdx,
                                            lane,
                                            groupIdx,
                                            CnIdx,
                                            Zc,
                                            p_PC_Flag);
        break;
    }
  }

  if (*p_iter_ptr == numMaxIter) { // output
    llrRes2llrOut_Kernel_BG1_int8(p_lut, p_p_llrOut, p_llrRes, Zc);
  } else {
    if (tid == 0) {
      // printf("Why you guys not here when iter_ptr = %d???\n",*iter_ptr);
      (*p_iter_ptr)++;
    }
  }
}

__global__ void OutPut_Kernel_BG1_R13_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                      int Zc,
                                                      int8_t *iter_ptr,
                                                      int8_t numMaxIter,
                                                      int *PC_Flag,
                                                      e_nrLDPC_outMode outMode,
                                                      int8_t *p_out,
                                                      int8_t *llrOut,
                                                      int8_t *p_llrOut,
                                                      uint32_t numLLR)
{
  // only activate in the last iteration
  int segIdx = blockIdx.y;
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

void nrLDPC_BnToCnPC_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                              int8_t *bnProcBufRes,
                                              int8_t *cnProcBuf,
                                              int8_t *cnProcBufRes,
                                              int8_t *bnProcBuf,
                                              int8_t *llrRes,
                                              int Z,
                                              int8_t *iter_ptr,
                                              int8_t numMaxIter,
                                              int *PC_Flag,
                                              e_nrLDPC_outMode outMode,
                                              int8_t *p_out,
                                              int8_t *llrOut,
                                              int8_t *p_llrOut,
                                              uint32_t numLLR,
                                              cudaStream_t *streams,
                                              int8_t CudaStreamIdx)
{
  BnToCnPC_Kernel_BG1_R13_int8_BIG_stream<<<Kdim[CudaStreamIdx].grid, Kdim[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
      p_lut,
      bnProcBufRes,
      cnProcBuf,
      cnProcBufRes,
      bnProcBuf,
      llrRes,
      Z,
      iter_ptr,
      numMaxIter,
      PC_Flag,
      outMode,
      p_out,
      llrOut,
      p_llrOut,
      numLLR);
}

void nrLDPC_OutPut_BG1_R13_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBufRes,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            int8_t *llrRes,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            e_nrLDPC_outMode outMode,
                                            int8_t *p_out,
                                            int8_t *llrOut,
                                            int8_t *p_llrOut,
                                            uint32_t numLLR,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{

  OutPut_Kernel_BG1_R13_int8_BIG_stream<<<Kdim[CudaStreamIdx].grid, Kdim[CudaStreamIdx].block, 0, streams[CudaStreamIdx]>>>(
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
}
//-----------------------------------------↑↑↑ R13 ↑↑↑----------------------------------------

//-----------------------------------------↓↓↓ R23 ↓↓↓----------------------------------------
__global__ void cnProcKernel_BG1_R23_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                     const int8_t *__restrict__ d_cnBufAll,
                                                     int8_t *__restrict__ d_cnOutAll,
                                                     int8_t *__restrict__ d_bnBufAll,
                                                     int Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int *PC_Flag)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }
  // printf("I'm inside cnProc_kernel\n");

  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...143
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R23[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R23[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R23[row];
  // uint16_t blockSize = h_block_thread_counts_cnProc[blk];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 13824) // 13824 is the total processed 144 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
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
      cnProcKernel_BG1_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 5:
      cnProcKernel_BG1_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 6:
      cnProcKernel_BG1_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 7:
      cnProcKernel_BG1_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 8:
      cnProcKernel_BG1_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  int maxBlockSize = 1024; // Maximun threads are 1024
  dim3 gridDim(14); // 50
  dim3 blockDim(maxBlockSize);

  cnProcKernel_BG1_R23_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                         cnProcBuf,
                                                                                         cnProcBufRes,
                                                                                         bnProcBuf,
                                                                                         Z,
                                                                                         iter_ptr,
                                                                                         numMaxIter,
                                                                                         PC_Flag);
}

__global__ void bnProcKernel_BG1_R23_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                                     int8_t *__restrict__ d_bnProcBufRes,
                                                     int8_t *__restrict__ d_llrProcBuf,
                                                     int8_t *__restrict__ d_llrRes,
                                                     const uint8_t *lut_numBnInBnGroups,
                                                     const uint32_t *lut_startAddrBnBuf,
                                                     const uint16_t *lut_startAddrBnLlr,
                                                     int Zc,
                                                     int8_t *iter_ptr,
                                                     int8_t numMaxIter,
                                                     int *PC_Flag)
{
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  /*if (tid == 0) {
    printf("3: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= 13824) {
    return;
  }

  int row = tid / 96; // to decide the inner block
  int lane = tid % 96; // to decide the inner lane

  uint8_t GrpIdx = lut_BnGrpIdx_BG1_R23[row];
  uint8_t MsgIdx = lut_BnMsgIdx_BG1_R23[row];
  uint8_t BnIdx = lut_BnIdx_BG1_R23[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx_BG1_R23[GrpIdx - 1];
  uint8_t GrpNum = lut_numBnInBnGroups[GrpIdx - 1];

  const int8_t *p_bnProcBuf_Grp = (const int8_t *)(d_bnProcBuf + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_bnProcBufRes_Grp = (const int8_t *)(d_bnProcBufRes + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_llrProcBuf_Grp = (const int8_t *)(d_llrProcBuf + lut_startAddrBnLlr[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp = (const int8_t *)(d_llrRes + lut_startAddrBnLlr[BnToAddrIdx - 1]);

  bnProcKernelMerge_BG1_int8_Gn(p_bnProcBuf_Grp,
                                (int8_t *)p_bnProcBufRes_Grp,
                                p_llrProcBuf_Grp,
                                (int8_t *)p_llrRes_Grp,
                                lane,
                                GrpIdx,
                                MsgIdx,
                                BnIdx,
                                GrpNum,
                                Zc);
}

void nrLDPC_bnProc_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBuf,
                                            int8_t *bnProcBufRes,
                                            int8_t *llrProcBuf,
                                            int8_t *llrRes,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  const uint8_t *lut_numBnInBnGroups;
  const uint32_t *lut_startAddrBnGroups;
  const uint16_t *lut_startAddrBnGroupsLlr;

  lut_numBnInBnGroups = p_lut->numBnInBnGroups;
  lut_startAddrBnGroups = p_lut->startAddrBnGroups;
  lut_startAddrBnGroupsLlr = p_lut->startAddrBnGroupsLlr;

  int8_t *p_bnProcBuf = (int8_t *)bnProcBuf;
  int8_t *p_bnProcBufRes = (int8_t *)bnProcBufRes;
  int8_t *p_llrProcBuf = (int8_t *)llrProcBuf;
  int8_t *p_llrRes = (int8_t *)llrRes;

  int maxBlockSize = 1024; // Z;
  int totalBlocks = 14;

  dim3 gridDim(totalBlocks);
  dim3 blockDim(maxBlockSize);

  bnProcKernel_BG1_R23_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_bnProcBuf,
                                                                                         p_bnProcBufRes,
                                                                                         p_llrProcBuf,
                                                                                         p_llrRes,
                                                                                         lut_numBnInBnGroups,
                                                                                         lut_startAddrBnGroups,
                                                                                         lut_startAddrBnGroupsLlr,
                                                                                         Z,
                                                                                         iter_ptr,
                                                                                         numMaxIter,
                                                                                         PC_Flag);
}

__global__ void BnToCnPC_Kernel_BG1_R23_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                        int8_t *__restrict__ d_bnOutAll,
                                                        const int8_t *__restrict__ d_cnBufAll,
                                                        int8_t *__restrict__ d_cnOutAll,
                                                        int8_t *__restrict__ d_bnBufAll,
                                                        int8_t *d_llrRes,
                                                        int Zc,
                                                        int8_t *iter_ptr,
                                                        int8_t numMaxIter,
                                                        int *PC_Flag,
                                                        e_nrLDPC_outMode outMode,
                                                        int8_t *p_out,
                                                        int8_t *llrOut,
                                                        int8_t *p_llrOut,
                                                        uint32_t numLLR)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...315
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R23[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R23[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R23[row];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 13824) // 30336 is the total processed 144 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;
  int8_t *p_bnProcBufRes = (int8_t *)d_bnOutAll;
  int8_t *p_llrRes = (int8_t *)d_llrRes;
  // int8_t *p_p_llrout =

  // Early stopping
  if (!(*iter_ptr > numMaxIter - 1 || *PC_Flag == 0)) {
    if (tid == 0) {
      *PC_Flag = 0;
      // printf("4: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
    }
    //__syncthreads();

    // uint32_t pcRes = 0; // setting flag for Parity Check

    switch (groupIdx) {
      case 0:
        CnToBnPC_Kernel_BG1_int8_G3_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           PC_Flag);
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
        CnToBnPC_Kernel_BG1_int8_G7_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           PC_Flag);
        break;
      case 5:
        CnToBnPC_Kernel_BG1_int8_G8_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           PC_Flag);
        break;
      case 6:
        CnToBnPC_Kernel_BG1_int8_G9_Stream(p_lut,
                                           p_bnProcBufRes,
                                           p_cnProcBuf,
                                           p_cnProcBufRes,
                                           p_bnProcBuf,
                                           MsgIdx,
                                           lane,
                                           groupIdx,
                                           CnIdx,
                                           Zc,
                                           PC_Flag);
        break;
      case 7:
        CnToBnPC_Kernel_BG1_int8_G10_Stream(p_lut,
                                            p_bnProcBufRes,
                                            p_cnProcBuf,
                                            p_cnProcBufRes,
                                            p_bnProcBuf,
                                            MsgIdx,
                                            lane,
                                            groupIdx,
                                            CnIdx,
                                            Zc,
                                            PC_Flag);
        break;
      case 8:
        CnToBnPC_Kernel_BG1_int8_G19_Stream(p_lut,
                                            p_bnProcBufRes,
                                            p_cnProcBuf,
                                            p_cnProcBufRes,
                                            p_bnProcBuf,
                                            MsgIdx,
                                            lane,
                                            groupIdx,
                                            CnIdx,
                                            Zc,
                                            PC_Flag);
        break;
    }
  }

  if (*iter_ptr == numMaxIter) { // output
    llrRes2llrOut_Kernel_BG1_int8(p_lut, p_llrOut, p_llrRes, Zc);
  } else {
    if (tid == 0) {
      // printf("Why you guys not here when iter_ptr = %d???\n",*iter_ptr);
      (*iter_ptr)++;
    }
  }
}

__global__ void OutPut_Kernel_BG1_R23_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                      int Zc,
                                                      int8_t *iter_ptr,
                                                      int8_t numMaxIter,
                                                      int *PC_Flag,
                                                      e_nrLDPC_outMode outMode,
                                                      int8_t *p_out,
                                                      int8_t *llrOut,
                                                      int8_t *p_llrOut,
                                                      uint32_t numLLR)
{
  // only activate in the last iteration

  if (*iter_ptr == numMaxIter) {
    if (outMode == nrLDPC_outMode_BIT)
      llr2bitPacked_Kernel_BG1_int8((uint8_t *)p_out, p_llrOut, numLLR);

    else // if (outMode == nrLDPC_outMode_BITINT8)
      llr2bit_Kernel_BG1_int8((uint8_t *)p_out, p_llrOut, numLLR);
  } else
    return;
}

void nrLDPC_BnToCnPC_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                              int8_t *bnProcBufRes,
                                              int8_t *cnProcBuf,
                                              int8_t *cnProcBufRes,
                                              int8_t *bnProcBuf,
                                              int8_t *llrRes,
                                              int Z,
                                              int8_t *iter_ptr,
                                              int8_t numMaxIter,
                                              int *PC_Flag,
                                              e_nrLDPC_outMode outMode,
                                              int8_t *p_out,
                                              int8_t *llrOut,
                                              int8_t *p_llrOut,
                                              uint32_t numLLR,
                                              cudaStream_t *streams,
                                              int8_t CudaStreamIdx)
{
  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  int maxBlockSize = 1024; // Maximun threads are 1024
  dim3 gridDim(14); // only need 14 for R23
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  // printf("In stream %d BC: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
  BnToCnPC_Kernel_BG1_R23_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                            bnProcBufRes,
                                                                                            cnProcBuf,
                                                                                            cnProcBufRes,
                                                                                            bnProcBuf,
                                                                                            llrRes,
                                                                                            Z,
                                                                                            iter_ptr,
                                                                                            numMaxIter,
                                                                                            PC_Flag,
                                                                                            outMode,
                                                                                            p_out,
                                                                                            llrOut,
                                                                                            p_llrOut,
                                                                                            numLLR);
}

void nrLDPC_OutPut_BG1_R23_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                            int8_t *bnProcBufRes,
                                            int8_t *cnProcBuf,
                                            int8_t *cnProcBufRes,
                                            int8_t *bnProcBuf,
                                            int8_t *llrRes,
                                            int Z,
                                            int8_t *iter_ptr,
                                            int8_t numMaxIter,
                                            int *PC_Flag,
                                            e_nrLDPC_outMode outMode,
                                            int8_t *p_out,
                                            int8_t *llrOut,
                                            int8_t *p_llrOut,
                                            uint32_t numLLR,
                                            cudaStream_t *streams,
                                            int8_t CudaStreamIdx)
{
  int maxBlockSize = 1024; // Maximun threads are 1024
  dim3 gridDim(14); // only need 14 for R23
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  // printf("In stream %d BC: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
  OutPut_Kernel_BG1_R23_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                          Z,
                                                                                          iter_ptr,
                                                                                          numMaxIter,
                                                                                          PC_Flag,
                                                                                          outMode,
                                                                                          p_out,
                                                                                          llrOut,
                                                                                          p_llrOut,
                                                                                          numLLR);
}

//-----------------------------------------↑↑↑ R23 ↑↑↑----------------------------------------

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//-----------------------CUDA Scheduler Area------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------

extern "C" void nrLDPC_decoder_scheduler_BG1_cuda_core(const t_nrLDPC_lut *p_lut,
                                                       int8_t *p_out,
                                                       uint32_t numLLR,
                                                       int8_t *cnProcBuf,
                                                       int8_t *cnProcBufRes,
                                                       int8_t *bnProcBuf,
                                                       int8_t *bnProcBufRes,
                                                       int8_t *llrRes,
                                                       int8_t *llrProcBuf,
                                                       int8_t *llrOut,
                                                       int8_t *p_llrOut,
                                                       int Z,
                                                       uint8_t BG,
                                                       uint8_t R,
                                                       uint8_t numMaxIter,
                                                       e_nrLDPC_outMode outMode,
                                                       cudaStream_t *streams,
                                                       uint8_t CudaStreamIdx,
                                                       cudaEvent_t *doneEvent,
                                                       int8_t *iter_ptr,
                                                       int *PC_Flag)
{
  cudaStream_t stream = streams[CudaStreamIdx];

  if (!graphCreated[CudaStreamIdx]) {
#if RECORD_GRAPH
    printf("Creating the graph for stream %d, format R%d\n", CudaStreamIdx, R);
#endif
    if (CudaStreamIdx != 0) {
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }

    // Start graph recording
#if RECORD_GRAPH
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
#endif
    // check_ptr_kernel_easy<<<1,10>>>(2);
    // cudaDeviceSynchronize();
    // CHECK(cudaGetLastError());
    switch (R) {
      case 13: {
        Kdim[CudaStreamIdx].block = dim3(1024, 1, 1);
        Kdim[CudaStreamIdx].grid = dim3(30, segmentPacks[CudaStreamIdx].nSeg, 1);
        for (int i = 0; i <= numMaxIter; i++) {
          // printf("I'm inside the loop i = %d\n", i);
          nrLDPC_cnProc_BG1_R13_cuda_stream_core(p_lut,
                                                 cnProcBuf,
                                                 cnProcBufRes,
                                                 bnProcBuf,
                                                 (int)Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          //          CHECK(cudaGetLastError());
          // cd cudaDeviceSynchronize();

          // printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          nrLDPC_bnProc_BG1_R13_cuda_stream_core(p_lut,
                                                 bnProcBuf,
                                                 bnProcBufRes,
                                                 llrProcBuf,
                                                 llrRes,
                                                 (int)Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          // cudaDeviceSynchronize();

          // printf("In stream %d 2: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          //        CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();
          nrLDPC_BnToCnPC_BG1_R13_cuda_stream_core(p_lut,
                                                   bnProcBufRes,
                                                   cnProcBuf,
                                                   cnProcBufRes,
                                                   bnProcBuf,
                                                   llrRes,
                                                   (int)Z,
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
          //      CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();
          // printf("In stream %d 3: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
        }
        nrLDPC_OutPut_BG1_R13_cuda_stream_core(p_lut,
                                               bnProcBufRes,
                                               cnProcBuf,
                                               cnProcBufRes,
                                               bnProcBuf,
                                               llrRes,
                                               (int)Z,
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

      } break;
      case 23: {
        Kdim[CudaStreamIdx].block = dim3(1024, 1, 1);
        Kdim[CudaStreamIdx].grid = dim3(14, segmentPacks[CudaStreamIdx].nSeg, 1);
        for (int i = 0; i <= numMaxIter; i++) {
          // printf("I'm inside the loop i = %d\n", i);
          nrLDPC_cnProc_BG1_R23_cuda_stream_core(p_lut,
                                                 cnProcBuf,
                                                 cnProcBufRes,
                                                 bnProcBuf,
                                                 (int)Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);
          CHECK(cudaGetLastError());

          // cudaDeviceSynchronize();
          /*if(i == 0){
            dumpAssCUDA(cnProcBuf, "Dump_cnProcBuf_cuda.txt");
            dumpAssCUDA(cnProcBufRes, "Dump_cnProcBufRes_cuda.txt");
            dumpAssCUDA(bnProcBuf, "Dump_bnProcBuf_cuda.txt");
          }
            */
          // printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
          nrLDPC_bnProc_BG1_R23_cuda_stream_core(p_lut,
                                                 bnProcBuf,
                                                 bnProcBufRes,
                                                 llrProcBuf,
                                                 llrRes,
                                                 (int)Z,
                                                 iter_ptr,
                                                 numMaxIter,
                                                 PC_Flag,
                                                 streams,
                                                 CudaStreamIdx);

          // cudaDeviceSynchronize();
          /*          if(i == 0){
                      dumpAssCUDA(bnProcBufRes, "Dump_bnProcBufRes_cuda.txt");
                      dumpAssCUDA(llrRes, "Dump_llrRes_cuda.txt");
                    }
                      */
          CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();
          nrLDPC_BnToCnPC_BG1_R23_cuda_stream_core(p_lut,
                                                   bnProcBufRes,
                                                   cnProcBuf,
                                                   cnProcBufRes,
                                                   bnProcBuf,
                                                   llrRes,
                                                   (int)Z,
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
          CHECK(cudaGetLastError());
          // cudaDeviceSynchronize();
        }
        nrLDPC_OutPut_BG1_R23_cuda_stream_core(p_lut,
                                               bnProcBufRes,
                                               cnProcBuf,
                                               cnProcBufRes,
                                               bnProcBuf,
                                               llrRes,
                                               (int)Z,
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

      } break;

      default:
        printf("Format not support yet\n");
        break;
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
    if (CudaStreamIdx != 0) {
      // uncomment below if you want streams works in sequence
      // cudaStreamWaitEvent(streams[CudaStreamIdx], doneEvent[CudaStreamIdx-1], 0);//cudaEventSynchronize(doneEvent[CudaStreamIdx -
      // 1]);
    }
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    //
  }
}

extern "C" bool is_device_pointer(const void *p)
{
  if (p == NULL)
    return false;
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, p);
  if (err != cudaSuccess) {
    // cudaPointerGetAttributes return value might vary in different runtime version
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attrs.type == cudaMemoryTypeDevice);
#else
  return (attrs.memoryType == cudaMemoryTypeDevice);
#endif
}

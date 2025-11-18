#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"
#include "nrLDPC_CUDA_shared_param.h"

//------------------------------Stream Version------------------------

__device__ void llrPreProc_Kernel_BG1_int8_Gn_stream(const t_nrLDPC_lut *p_lut,
                                                     const int8_t *p_llr,
                                                     int8_t *p_llrProcBuf,
                                                     int8_t *p_cnProcBuf,
                                                     uint32_t row,
                                                     uint32_t lane,
                                                     uint32_t idxBn,
                                                     uint32_t GrpIdx,
                                                     uint32_t circShift,
                                                     uint32_t Zc,
                                                     uint32_t R)
{   
  {
    uint32_t *p_cnProcBufBit;

    uint8_t bricksLocal[4];
    uint8_t *BricksToBeMoved = bricksLocal;

    p_cnProcBufBit = (uint32_t *)(p_cnProcBuf + d_lut_numCnInCnGroups_BG1_R13[GrpIdx] * Zc * row + lane * 4);

    moveBricks_invget_circ((int8_t *)&p_llr[idxBn], lane * 4, BricksToBeMoved, Zc, circShift);

    *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;
  }
  // Sencond part is llr to llrProcBuf
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t colIdx = tid / RowLength;
  if (colIdx >= 42) // need to modify later
    return;

  const uint8_t numBn2CnG1 = (R == 13) ? d_lut_numBnInBnGroups_BG1_R13[0]:d_lut_numBnInBnGroups_BG1_R23[0]; // for R13 is 42
  const uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1; // 26 for BG1
  const uint32_t colG1 = startColParity * Zc;

  const uint32_t *lut_llr2llrProcBufAddr = (R == 13) ? d_llr2llrProcBufAddr_BG1_R13:d_llr2llrProcBufAddr_BG1_R23;
  const uint32_t *lut_llr2llrProcBufBnPos = (R == 13) ? d_llr2llrProcBufBnPos_BG1_R13:d_llr2llrProcBufBnPos_BG1_R23;

  // -----------------------------
  // Part 1: Copy parity section
  // -----------------------------
  if (numBn2CnG1 > 0 && colIdx < numBn2CnG1) {
    int32_t *dst = (int32_t *)(&p_llrProcBuf[colIdx * Zc] + lane * 4);
    int32_t *src = (int32_t *)(&p_llr[colG1 + colIdx * Zc] + lane * 4);
    *dst = *src;
  }

  // -----------------------------
  // Part 2: Copy systematic section (0..startColParity)
  // -----------------------------
  if (colIdx < startColParity) {
    const uint32_t idxBn = lut_llr2llrProcBufAddr[colIdx] + lut_llr2llrProcBufBnPos[colIdx] * Zc;
    int32_t *dst = (int32_t *)(&p_llrProcBuf[idxBn] + lane * 4);
    int32_t *src = (int32_t *)(&p_llr[colIdx * Zc] + lane * 4);
    *dst = *src;
  }
}


__device__ void llrRes2llrOut_Kernel_BG1_int8(uint8_t R, int8_t *llrOut, int8_t *llrRes, uint32_t Zc)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t colIdx = tid / RowLength;
  int lane = tid % RowLength;
  if (colIdx >= 42)
    return;

  const uint8_t numBn2CnG1 =
      (R == 13) ? d_lut_numBnInBnGroups_BG1_R13[0] : d_lut_numBnInBnGroups_BG1_R23[0]; // numBnInBnGroups[0] = 42
  uint32_t startColParity = // BG1=26
      NR_LDPC_START_COL_PARITY_BG1; //(BG == 1) ? (NR_LDPC_START_COL_PARITY_BG1) : (NR_LDPC_START_COL_PARITY_BG2);

  uint32_t colG1 = startColParity * Zc;

  const uint32_t *lut_llr2llrProcBufAddr = (R == 13) ? d_llr2llrProcBufAddr_BG1_R13 : d_llr2llrProcBufAddr_BG1_R23;
  const uint32_t *lut_llr2llrProcBufBnPos = (R == 13) ? d_llr2llrProcBufBnPos_BG1_R13 : d_llr2llrProcBufBnPos_BG1_R23;

  int8_t *p_llrOut = &llrOut[0];
  if (colIdx < startColParity) {
    const uint32_t idxBn = lut_llr2llrProcBufAddr[colIdx] + lut_llr2llrProcBufBnPos[colIdx] * Zc;
    int32_t *dst_ptr2 = (int32_t *)(p_llrOut + colIdx * Zc + lane * 4);
    int32_t *src_ptr2 = (int32_t *)(&llrRes[idxBn] + lane * 4);
    *dst_ptr2 = *src_ptr2; // 0x01010101*colIdx;//
  }

  //  __syncthreads();
  if (numBn2CnG1 > 0) {
    if (colIdx < numBn2CnG1) {
      int32_t *dst_ptr1 = (int32_t *)(&llrOut[colG1] + colIdx * Zc + lane * 4);
      int32_t *src_ptr1 = (int32_t *)(llrRes + colIdx * Zc + lane * 4);
      *dst_ptr1 = *src_ptr1; // 0x10101010*colIdx;//
    }
  }
}

__device__ void llr2bitPacked_Kernel_BG1_int8(uint8_t *out, int8_t *llrOut, uint32_t numLLR)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalGroups = numLLR >> 3; // every 8  LLR as a group

  if (tid >= totalGroups)
    return;

  int8_t *p_llr = llrOut + tid * 8;
  uint8_t result = 0;

  //  shuffling
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result |= (p_llr[7 - i] < 0) << i;
  }

  out[tid] = result;
}

__device__ void llr2bit_Kernel_BG1_int8(uint8_t *out, int8_t *llrOut, uint32_t numLLR)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalGroups = numLLR >> 3; // every 8 LLR as a group

  if (tid >= totalGroups)
    return;

  int8_t *p_llr = llrOut + tid * 8;
  uint8_t result = 0;

#pragma unroll
  for (int i = 0; i < 8; i++)
    out[tid * 8 + i] = (p_llr[i] < 0);
}

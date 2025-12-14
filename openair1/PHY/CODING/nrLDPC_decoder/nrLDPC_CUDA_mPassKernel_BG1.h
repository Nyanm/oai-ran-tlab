#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"
#include "nrLDPC_CUDA_shared_param.h"

//------------------------------Stream Version------------------------

__device__ __forceinline__ void llrPreProc_Kernel_BG1_int8_Gn_stream(const int8_t *p_llr,
                                                                     int8_t *p_llrProcBuf,
                                                                     int8_t *p_cnProcBuf,
                                                                     uint32_t MsgIdx,
                                                                     uint32_t lane,
                                                                     uint32_t colIdx,
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

    p_cnProcBufBit = (uint32_t *)(p_cnProcBuf + d_lut_numCnInCnGroups_BG1_R13[GrpIdx] * NR_LDPC_ZMAX * MsgIdx + lane * 4);

    moveBricks_invget_circ((int8_t *)&p_llr[idxBn*Zc], lane * 4, BricksToBeMoved, Zc, circShift);

    *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;
  }
  // Sencond part is llr to llrProcBuf

  if (colIdx >= 68) // need to modify later
    return;

  const uint8_t numBn2CnG1 = (R == 13) ? d_lut_numBnInBnGroups_BG1_R13[0] : d_lut_numBnInBnGroups_BG1_R23[0]; // for R13 is 42
  const uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1; // 26 for BG1
  const uint32_t colG1 = startColParity * Zc;

  const uint32_t *lut_llr2llrProcBufAddr = (R == 13) ? d_llr2llrProcBufAddr_BG1_R13 : d_llr2llrProcBufAddr_BG1_R23;
  const uint32_t *lut_llr2llrProcBufBnPos = (R == 13) ? d_llr2llrProcBufBnPos_BG1_R13 : d_llr2llrProcBufBnPos_BG1_R23;

  // -----------------------------
  // Part 1: Copy systematic section (0..startColParity)
  // -----------------------------
  if (colIdx < startColParity) {
    const uint32_t idxBn = lut_llr2llrProcBufAddr[colIdx] + lut_llr2llrProcBufBnPos[colIdx] * NR_LDPC_ZMAX;
    int32_t *dst = (int32_t *)(&p_llrProcBuf[idxBn] + lane * 4);
    int32_t *src = (int32_t *)(&p_llr[colIdx * Zc] + lane * 4);
    *dst = *src;
  } else {
    // -----------------------------
    // Part 2: Copy parity section
    // -----------------------------
    colIdx = colIdx - startColParity;
    if (numBn2CnG1 > 0 && colIdx < numBn2CnG1) {
      int32_t *dst = (int32_t *)(&p_llrProcBuf[colIdx * NR_LDPC_ZMAX] + lane * 4);
      int32_t *src = (int32_t *)(&p_llr[colG1 + colIdx * Zc] + lane * 4);
      *dst = *src;
    }
  }
}

__device__ void llr2bit_Kernel_BG1_int8(uint32_t R,
                                 uint8_t *__restrict__ out,
                                 const int8_t *__restrict__ llrRes,
                                 uint32_t numLLR,
                                 uint32_t Zc)
{
  uint32_t lane = threadIdx.x;
  uint32_t outColIdx = (blockIdx.x << 2) + threadIdx.y;

  if (outColIdx >= 68)
    return;

  // --- LUT and Constants Setup ---
  const uint8_t numBn2CnG1 = (R == 13) ? d_lut_numBnInBnGroups_BG1_R13[0] : d_lut_numBnInBnGroups_BG1_R23[0];
  const uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1; // Typically 26

  const uint32_t *lut_Addr = (R == 13) ? d_llr2llrProcBufAddr_BG1_R13 : d_llr2llrProcBufAddr_BG1_R23;
  const uint32_t *lut_Pos = (R == 13) ? d_llr2llrProcBufBnPos_BG1_R13 : d_llr2llrProcBufBnPos_BG1_R23;

  int32_t raw_llrs; // Register to hold 4 input LLRs (4 bytes)

  // ==========================================
  // PHASE 1: Load Data (Gather or Shift)
  // ==========================================
  if (outColIdx < startColParity) {
    // --- Mode A: Systematic Bits (Gather) ---
    uint32_t idxBn = lut_Addr[outColIdx] + lut_Pos[outColIdx] * NR_LDPC_ZMAX;
    raw_llrs = *(const int32_t *)(&llrRes[idxBn] + lane * 4);
  } else {
    // --- Mode B: Parity Bits (Linear Shift) ---
    uint32_t srcParityIdx = outColIdx - startColParity;
    if (numBn2CnG1 > 0 && outColIdx < numBn2CnG1) {
      raw_llrs = *(const int32_t *)(llrRes + srcParityIdx * NR_LDPC_ZMAX + lane * 4);
    } else {
      raw_llrs = 0;
    }
  }

  // ==========================================
  // PHASE 2: Hard Decision & Packing
  // ==========================================
  // Convert 4 int8 LLRs into 4 Bytes (0x00 or 0x01) inside a uint32 register
  int8_t *p_val = (int8_t *)&raw_llrs;
  uint32_t my_word = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    // If LLR < 0, byte is 1. Otherwise 0.
    uint32_t byte_val = (p_val[i] < 0) ? 1 : 0;
    my_word |= (byte_val << (i * 8));
  }

  // ==========================================
  // PHASE 3: Store Data
  // ==========================================
  // Write 4 bytes linearly.
  uint32_t outAddr = outColIdx * Zc + lane * 4;

  *(uint32_t *)(&out[outAddr]) = my_word;
}


__device__ void llr2bitPacked_Kernel_BG1_int8(uint32_t R,
                                       uint8_t *__restrict__ out,
                                       const int8_t *__restrict__ llrRes,
                                       uint32_t numLLR,
                                       uint32_t Zc)
{
  uint32_t lane = threadIdx.x;
  uint32_t outColIdx = (blockIdx.x << 2) + threadIdx.y;

  if (outColIdx >= 68)
    return;

  const uint8_t numBn2CnG1 = (R == 13) ? d_lut_numBnInBnGroups_BG1_R13[0] : d_lut_numBnInBnGroups_BG1_R23[0];
  uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1;

  const uint32_t *lut_Addr = (R == 13) ? d_llr2llrProcBufAddr_BG1_R13 : d_llr2llrProcBufAddr_BG1_R23;
  const uint32_t *lut_Pos = (R == 13) ? d_llr2llrProcBufBnPos_BG1_R13 : d_llr2llrProcBufBnPos_BG1_R23;

  int32_t raw_llrs;

  // ==========================================
  // PHASE 1: Load Data (Gather or Shift)
  // ==========================================
  if (outColIdx < startColParity) {
    // Mode A: Systematic Bits
    uint32_t idxBn = lut_Addr[outColIdx] + lut_Pos[outColIdx] * NR_LDPC_ZMAX;
    raw_llrs = *(const int32_t *)(&llrRes[idxBn] + lane * 4);
  } else {
    // Mode B: Parity Bits
    uint32_t srcParityIdx = outColIdx - startColParity;
    if (numBn2CnG1 > 0 && outColIdx < numBn2CnG1) {
      raw_llrs = *(const int32_t *)(llrRes + srcParityIdx * NR_LDPC_ZMAX + lane * 4);
    } else {
      raw_llrs = 0;
    }
  }

  // ==========================================
  // PHASE 2: Extract Sign Bits (Local Packing)
  // ==========================================
  // Each thread holds 4 LLRs. We need to extract 4 bits.

  uint32_t my_4_bits = 0;
  int8_t *p_val = (int8_t *)&raw_llrs;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    if (p_val[i] < 0) {
      my_4_bits |= (1 << (3 - i));
    }
  }

  // ==========================================
  // PHASE 3: Thread Cooperation & Store
  // ==========================================

  // Exchange data between Even (0,2..) and Odd (1,3..) threads.
  // 'neighbor_bits' will contain 'my_4_bits' from the other thread.
  uint32_t neighbor_bits = __shfl_xor_sync(0xffffffff, my_4_bits, 1);

  // Only Even threads perform the write (reducing stores by 50%)
if ((lane & 1) == 0) {
    // Combine 4 bits from self (Low nibble) and 4 bits from neighbor (High nibble)
    // Output: [Thread N+1 bits][Thread N bits]
    uint8_t packed_byte = (neighbor_bits & 0xF) | ((my_4_bits & 0xF) << 4);

    // Calculate Output Address
    // Each column size compresses from Zc bytes to (Zc / 8) bytes.
    // 'lane' steps by 4 LLRs. 'lane >> 1' steps by 8 LLRs (1 Byte).
    uint32_t outAddr = outColIdx * (Zc >> 3) + (lane >> 1);

    out[outAddr] = packed_byte;
}
}
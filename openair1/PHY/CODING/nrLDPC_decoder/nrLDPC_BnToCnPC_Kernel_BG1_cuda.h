#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CnProcKernel_BG1_cuda.h"

// #define arrPos(a, b) a.d + b *a.dim2

// enum CircShiftDirection { FORWARD = 0, INVERSE = 1 };
// enum CircShiftOp { PUT_BRICKS = 0, GET_BRICKS = 1 };
/*
__device__ void moveBricks_circ(int8_t *__restrict__ dstBuf,
                                uint16_t dstBuf_Offset,
                                uint8_t *__restrict__ Four_Bricks,
                                uint16_t Z,
                                uint16_t cshift,
                                CircShiftDirection dir,
                                CircShiftOp op)
{
  int8_t *DstBuf = (int8_t *)dstBuf;
  uint16_t shift;

  if (dir == FORWARD) {
    shift = (Z - ((cshift + dstBuf_Offset) % Z)) % Z;
  } else {
    shift = (cshift + dstBuf_Offset) % Z;
  }

  uint16_t pos = shift;
  uintptr_t ptr = (uintptr_t)(DstBuf + pos);

  if (op == PUT_BRICKS) {
    // put bricks
    if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
      *(uint32_t *)(DstBuf + pos) = *(const uint32_t *)(Four_Bricks);
    } else {
      for (uint16_t j = 0; j < 4; j++) {
        DstBuf[(pos + j) % Z] = Four_Bricks[j];
      }
    }
  } else if (op == GET_BRICKS) {
    // get bricks
    if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
      *(uint32_t *)(Four_Bricks) = *(const uint32_t *)(DstBuf + pos);
    } else {
      for (uint16_t j = 0; j < 4; j++) {
        Four_Bricks[j] = DstBuf[(pos + j) % Z];
      }
    }
  }
}
*/
__device__ void CnToBnPC_Kernel_int8_G3(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 3; // Gn = 3
  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  const uint baseShift = Zc * row;
  const uint destByte = baseShift + lane * 4;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const int idxBn = lut_startAddrBnProcBuf_CNG[0];
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------
  moveBricks_circ((int8_t *)&d_bnOutAll[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;
  //------------------------------------Done----------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 3; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 1 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 1 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G3, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_int8_G4(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 4; // Gn = 4
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 4; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G4, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_int8_G5(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 5; // Gn = 5
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 18 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 5; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 18 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 18 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G5, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G6(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 6; // Gn = 6
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 8 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 6; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 8 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 8 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G6, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G7(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 7; // Gn = 7
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 7; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G7, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G8(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 8; // Gn = 8
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 8; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G8, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G9(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 9; // Gn = 9
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 9; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G9, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G10(const t_nrLDPC_lut *p_lut,
                                         int8_t *__restrict__ d_bnOutAll,
                                         const int8_t *__restrict__ d_cnBufAll,
                                         int8_t *__restrict__ d_cnOutAll,
                                         int8_t *__restrict__ d_bnBufAll,
                                         int tid,
                                         uint8_t groupId,
                                         uint8_t CnIdx,
                                         int Zc,
                                         int *PC_Flag)
{
  const uint8_t NUM = 10; // Gn = 10
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 1 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 10; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 1 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 1 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G10, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G19(const t_nrLDPC_lut *p_lut,
                                         int8_t *__restrict__ d_bnOutAll,
                                         const int8_t *__restrict__ d_cnBufAll,
                                         int8_t *__restrict__ d_cnOutAll,
                                         int8_t *__restrict__ d_bnBufAll,
                                         int Tid,
                                         uint8_t groupId,
                                         uint8_t CnIdx,
                                         int Zc,
                                         int *PC_Flag)
{
  const uint8_t NUM = 19; // Gn = 19
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  //------------first half----------------
  int tid = Tid; //+ 912 * (blockIdx.x % 2); // same reason, now the following no need to change

  if (tid >= NUM * Zc / 4)
    return;

  uint row = tid / 96;
  uint lane = tid % 96;

  uint baseShift = 4 * Zc * row; // offset pointed at different BN
  uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);
  // p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + destByte);

  uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------First half DONE----------------------------------------

  // Second half start
  tid = Tid + 912; // same reason, now the following no need to change

  if (tid >= NUM * Zc / 4)
    return;

  row = tid / 96;
  lane = tid % 96;

  baseShift = 4 * Zc * row; // offset pointed at different BN
  destByte = baseShift + lane * 4; // offset to different part inside different BN

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------Second half DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (Tid < 96) {
    for (int i = 0; i < 19; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 4 * Zc * i + Tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 4 * Zc * i + Tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (Tid % warpSize == 0) {
        printf("It's wrong here G19, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
//------------------------------Stream Version------------------------
__device__ void CnToBnPC_Kernel_int8_G3_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 3; // Gn = 3
  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  const uint baseShift = Zc * row;
  const uint destByte = baseShift + lane * 4;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const int idxBn = lut_startAddrBnProcBuf_CNG[0];
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------
  moveBricks_circ((int8_t *)&d_bnOutAll[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;
  //------------------------------------Done----------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 3; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 1 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 1 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G3, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_int8_G4_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 4; // Gn = 4
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 4; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G4, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_int8_G5_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 5; // Gn = 5
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 18 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 5; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 18 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 18 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G5, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G6_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 6; // Gn = 6
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 8 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 6; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 8 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 8 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G6, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G7_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 7; // Gn = 7
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 7; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G7, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G8_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 8; // Gn = 8
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 8; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G8, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G9_Stream(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc,
                                        int *PC_Flag)
{
  const uint8_t NUM = 9; // Gn = 9
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 9; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G9, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G10_Stream(const t_nrLDPC_lut *p_lut,
                                         int8_t *__restrict__ d_bnOutAll,
                                         const int8_t *__restrict__ d_cnBufAll,
                                         int8_t *__restrict__ d_cnOutAll,
                                         int8_t *__restrict__ d_bnBufAll,
                                         int tid,
                                         uint8_t groupId,
                                         uint8_t CnIdx,
                                         int Zc,
                                         int *PC_Flag)
{
  const uint8_t NUM = 10; // Gn = 10
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 1 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (tid < 96) {
    for (int i = 0; i < 10; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 1 * Zc * i + tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 1 * Zc * i + tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        printf("It's wrong here G10, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_int8_G19_Stream(const t_nrLDPC_lut *p_lut,
                                         int8_t *__restrict__ d_bnOutAll,
                                         const int8_t *__restrict__ d_cnBufAll,
                                         int8_t *__restrict__ d_cnOutAll,
                                         int8_t *__restrict__ d_bnBufAll,
                                         int Tid,
                                         uint8_t groupId,
                                         uint8_t CnIdx,
                                         int Zc,
                                         int *PC_Flag)
{
  const uint8_t NUM = 19; // Gn = 19
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  //------------first half----------------
  int tid = Tid; //+ 912 * (blockIdx.x % 2); // same reason, now the following no need to change

  if (tid >= NUM * Zc / 4)
    return;

  uint row = tid / 96;
  uint lane = tid % 96;

  uint baseShift = 4 * Zc * row; // offset pointed at different BN
  uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit, *p_cnProcBufResBit;

  uint8_t bricksLocal[4];
  uint8_t *BricksToBeMoved = bricksLocal;

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);
  // p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + destByte);

  uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------First half DONE----------------------------------------

  // Second half start
  tid = Tid + 912; // same reason, now the following no need to change

  if (tid >= NUM * Zc / 4)
    return;

  row = tid / 96;
  lane = tid % 96;

  baseShift = 4 * Zc * row; // offset pointed at different BN
  destByte = baseShift + lane * 4; // offset to different part inside different BN

  p_cnProcBufBit = (uint32_t *)(d_cnBufAll + destByte);

  lut_circShift_CNG = arrPos(p_lut->circShift[groupId], row);
  lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupId], row);
  lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupId], row);

  idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;
  //-----------------------Copy BnProcBufRes to CnProcBuf---------------------

  moveBricks_circ((int8_t *)&p_bnProcBufRes[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, GET_BRICKS);

  *p_cnProcBufBit = *(uint32_t *)BricksToBeMoved;

  //-------------------------------------Second half DONE----------------------------------------
  __syncthreads();
 uint32_t pcRes = 0;
  uint32_t ymm0, ymm1;
  if (Tid < 96) {
    for (int i = 0; i < 19; i++) {
      p_cnProcBufBit = (uint32_t *)(d_cnBufAll + 4 * Zc * i + Tid * 4);
      p_cnProcBufResBit = (uint32_t *)(d_cnOutAll + 4 * Zc * i + Tid * 4);
      ymm0 = *p_cnProcBufBit;
      ymm1 = *p_cnProcBufResBit;

      pcRes ^= __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (Tid % warpSize == 0) {
        printf("It's wrong here G19, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void llrRes2llrOut_Kernel_int8_BG1(const t_nrLDPC_lut *p_lut, int8_t *llrOut, int8_t *llrRes, int Zc)
{
  int colIdx = blockIdx.x; // 每个 block 对应一个逻辑列
  int tid = threadIdx.x; // 每个线程负责 Z 内的一部分 (0 ~ 23)
    // 限制线程数量
  if (tid >= (Zc / 4))
    return;
    
  const uint8_t numBn2CnG1 = p_lut->numBnInBnGroups[0];
  uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1;//(BG == 1) ? (NR_LDPC_START_COL_PARITY_BG1) : (NR_LDPC_START_COL_PARITY_BG2);

  uint32_t colG1 = startColParity * Zc;

  const uint16_t* lut_llr2llrProcBufAddr = p_lut->llr2llrProcBufAddr;
  const uint8_t* lut_llr2llrProcBufBnPos = p_lut->llr2llrProcBufBnPos;

  int8_t* p_llrOut = &llrOut[0];

  if (numBn2CnG1 > 0) {
    if(colIdx<numBn2CnG1){
      int32_t* dst_ptr = (int32_t*)(&llrOut[colG1] + colIdx * Zc + tid * 4);
      int32_t* src_ptr = (int32_t*)(llrRes + colIdx * Zc + tid * 4);
      *dst_ptr = *src_ptr;//0x10101010*colIdx;//
    }
  }

  if(colIdx < startColParity){
    const int idxBn = lut_llr2llrProcBufAddr[colIdx] + lut_llr2llrProcBufBnPos[colIdx] * Zc;
    int32_t* dst_ptr = (int32_t*)(p_llrOut + colIdx * Zc + tid * 4);
    int32_t* src_ptr = (int32_t*)(&llrRes[idxBn] + tid * 4);
    *dst_ptr = *src_ptr;//0x01010101*colIdx;//
  }

}

__device__ void llr2bitPacked_Kernel_int8_BG1(uint8_t* out, int8_t* llrOut, uint32_t numLLR) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalGroups = numLLR >> 3;  // 每 8 个 LLR 为一组

    if (tid >= totalGroups)
        return;

    int8_t* p_llr = llrOut + tid * 8;
    uint8_t result = 0;

    // 按照原始 shuffle 逆序：从 index 7 到 0 判断符号位
    for (int i = 0; i < 8; i++) {
        result |= (p_llr[7 - i] < 0) << i;
    }

    out[tid] = result;
}

__device__ void llr2bit_Kernel_int8_BG1(uint8_t* out, int8_t* llrOut, uint32_t numLLR) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalGroups = numLLR >> 3;  // 每 8 个 LLR 为一组

    if (tid >= totalGroups)
        return;

    int8_t* p_llr = llrOut + tid * 8;
    uint8_t result = 0;

    // don't need shuffle 
    for (int i = 0; i < 8; i++) {
        result |= (p_llr[i] < 0) << i;
    }

    out[tid] = result;
}
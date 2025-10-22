#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"

//------------------------------Stream Version------------------------
__device__ void CnToBnPC_Kernel_BG1_int8_G3_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
                                                   uint8_t groupId,
                                                   uint8_t CnIdx,
                                                   int Zc,
                                                   int *PC_Flag)
{
  const uint8_t NUM = 3; // Gn = 3
  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[3];
  if (tid < 96) {
    uint32_t tmp[8];
#pragma unroll
    for (int i = 0; i < 3; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 3; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        ////printf("It's wrong here G3, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_BG1_int8_G4_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[4];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      pcRes ^= tmp[i];
    }


    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        ////printf("It's wrong here G4, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__device__ void CnToBnPC_Kernel_BG1_int8_G5_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[5];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 5; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 18 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 18 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 5; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G5, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G6_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[6];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 8 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 8 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 6; i++) {
      pcRes ^= tmp[i];
    }


    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G6, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G7_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[7];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 7; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 5 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 5 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 7; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G7, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G8_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[8];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G8, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G9_Stream(const t_nrLDPC_lut *p_lut,
                                                   int8_t *__restrict__ d_bnOutAll,
                                                   const int8_t *__restrict__ d_cnBufAll,
                                                   int8_t *__restrict__ d_cnOutAll,
                                                   int8_t *__restrict__ d_bnBufAll,
                                                   int8_t MsgIdx,
                                                   int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[9];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 9; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 2 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 2 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 9; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G9, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G10_Stream(const t_nrLDPC_lut *p_lut,
                                                    int8_t *__restrict__ d_bnOutAll,
                                                    const int8_t *__restrict__ d_cnBufAll,
                                                    int8_t *__restrict__ d_cnOutAll,
                                                    int8_t *__restrict__ d_bnBufAll,
                                                    int8_t MsgIdx,
                                                    int lane,
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

  const uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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
  uint32_t tmp[10];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 10; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 10; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G10, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}
__device__ void CnToBnPC_Kernel_BG1_int8_G19_Stream(const t_nrLDPC_lut *p_lut,
                                                    int8_t *__restrict__ d_bnOutAll,
                                                    const int8_t *__restrict__ d_cnBufAll,
                                                    int8_t *__restrict__ d_cnOutAll,
                                                    int8_t *__restrict__ d_bnBufAll,
                                                    int8_t MsgIdx,
                                                    int lane,
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

  uint row = MsgIdx - 1;
  int tid = row * 96 + lane;

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

  //-------------------------------------DONE----------------------------------------
  __syncthreads();
  uint32_t pcRes = 0;
  uint32_t tmp[19];
  if (tid < 96) {
#pragma unroll
    for (int i = 0; i < 19; i++) {
      uint32_t ymm0 = *(uint32_t *)(d_cnBufAll + 4 * Zc * i + tid * 4);
      uint32_t ymm1 = *(uint32_t *)(d_cnOutAll + 4 * Zc * i + tid * 4);
      tmp[i] = __vcmples4(__vaddss4(ymm0, ymm1), 0);
    }
#pragma unroll
    for (int i = 0; i < 19; i++) {
      pcRes ^= tmp[i];
    }

    if (__any_sync(0xffffffff, pcRes != 0)) {
      if (tid % warpSize == 0) {
        // printf("It's wrong here G19, pcRes = %d\n", pcRes);
        *PC_Flag = 1; // atomicOr(PC_Flag, 1);
      }
    }
  }
}

__global__ void llrPreProc_Kernel_BG1_int8(const t_nrLDPC_lut *p_lut,
                                               const int8_t * llr,
                                               int8_t * llrProcBuf,
                                               int8_t* cnProcBuf,
                                               int Zc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int colIdx = tid / 96;   
    int lane   = tid % 96;   

    if (colIdx >= 42)//need to modify later
        return;


    

    const uint8_t numBn2CnG1 = p_lut->numBnInBnGroups[0]; // for R13 is 42
    const uint32_t startColParity = NR_LDPC_START_COL_PARITY_BG1; // 26 for BG1
    const uint32_t colG1 = startColParity * Zc;

    const uint16_t *lut_addr = p_lut->llr2llrProcBufAddr;
    const uint8_t  *lut_pos  = p_lut->llr2llrProcBufBnPos;

    // -----------------------------
    // Part 1: Copy parity section
    // -----------------------------
    if (numBn2CnG1 > 0 && colIdx < numBn2CnG1) {
        int32_t *dst = (int32_t *)(&llrProcBuf[colIdx * Zc] + lane * 4);
        int32_t *src = (int32_t *)(&llr[colG1 + colIdx * Zc] + lane * 4);
        *dst = *src;
    }

    // -----------------------------
    // Part 2: Copy systematic section (0..startColParity)
    // -----------------------------
    if (colIdx < startColParity) {
        const int idxBn = lut_addr[colIdx] + lut_pos[colIdx] * Zc;
        int32_t *dst = (int32_t *)(&llrProcBuf[idxBn] + lane * 4);
        int32_t *src = (int32_t *)(&llr[colIdx * Zc] + lane * 4);
        *dst = *src;
    }
}

__device__ void llrRes2llrOut_Kernel_BG1_int8(const t_nrLDPC_lut *p_lut, int8_t *llrOut, int8_t *llrRes, int Zc)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colIdx = tid / 96;
  int lane = tid % 96;
  if (colIdx >= 42)
    return;

  const uint8_t numBn2CnG1 = p_lut->numBnInBnGroups[0]; // numBnInBnGroups[0] = 42
  uint32_t startColParity = // BG1=26
      NR_LDPC_START_COL_PARITY_BG1; //(BG == 1) ? (NR_LDPC_START_COL_PARITY_BG1) : (NR_LDPC_START_COL_PARITY_BG2);

  uint32_t colG1 = startColParity * Zc;

  const uint16_t *lut_llr2llrProcBufAddr = p_lut->llr2llrProcBufAddr;
  const uint8_t *lut_llr2llrProcBufBnPos = p_lut->llr2llrProcBufBnPos;

  int8_t *p_llrOut = &llrOut[0];
  if (colIdx < startColParity) {
    const int idxBn = lut_llr2llrProcBufAddr[colIdx] + lut_llr2llrProcBufBnPos[colIdx] * Zc;
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
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalGroups = numLLR >> 3; // every 8  LLR as a group

  if (tid >= totalGroups)
    return;

  int8_t *p_llr = llrOut + tid * 8;
  uint8_t result = 0;

  //  shuffling
#pragma unroll
  for(int i = 0; i < 8; i++) {
    result |= (p_llr[7 - i] < 0) << i;
  }

  out[tid] = result;
}

__device__ void llr2bit_Kernel_BG1_int8(uint8_t *out, int8_t *llrOut, uint32_t numLLR)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalGroups = numLLR >> 3; // every 8 LLR as a group

  if (tid >= totalGroups)
    return;

  int8_t *p_llr = llrOut + tid * 8;
  uint8_t result = 0;

#pragma unroll
  for (int i = 0; i < 8; i++)
    out[tid * 8 + i] = (p_llr[i] < 0);
}

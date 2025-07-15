#pragma once


#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CnProcKernel_BG1_cuda.h"

//#define arrPos(a, b) a.d + b *a.dim2

//enum CircShiftDirection { FORWARD = 0, INVERSE = 1 };
//enum CircShiftOp { PUT_BRICKS = 0, GET_BRICKS = 1 };
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
                                        int Zc)
{
  const uint8_t NUM = 3; // Gn = 3
  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  uint32_t *p_cnProcBufBit;

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
}

__device__ void CnToBnPC_Kernel_int8_G4(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}

__device__ void CnToBnPC_Kernel_int8_G5(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G6(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G7(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G8(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G9(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G10(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
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

  uint32_t *p_cnProcBufBit;

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
}
__device__ void CnToBnPC_Kernel_int8_G19(const t_nrLDPC_lut *p_lut,
                                        int8_t *__restrict__ d_bnOutAll,
                                        const int8_t *__restrict__ d_cnBufAll,
                                        int8_t *__restrict__ d_cnOutAll,
                                        int8_t *__restrict__ d_bnBufAll,
                                        int Tid,
                                        uint8_t groupId,
                                        uint8_t CnIdx,
                                        int Zc)
{
  const uint8_t NUM = 19; // Gn = 19
  const int8_t *p_bnProcBufRes = (const int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  int tid = Tid + 912 * (blockIdx.x % 2); // same reason, now the following no need to change


  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;

  const uint baseShift = 4 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN

  uint32_t *p_cnProcBufBit;

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
}
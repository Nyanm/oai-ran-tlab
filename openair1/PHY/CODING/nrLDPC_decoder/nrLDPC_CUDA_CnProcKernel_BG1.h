#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"
#include "nrLDPC_CUDA_lut.h"

__device__ __forceinline__ void cnProcKernel_BG1_int8_G3(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 3; // Gn = 3
  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint row = MsgIdx - 1; // row = 0,1,2  -> 3 BNs

  // 1*384/4 = 96

  const uint baseShift = Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;

  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  // loop starts here
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);
  /*if(row == 0 && blockIdx.x == 0){
      printf("In thread %d, in address offset: %d, ymm0 = %02x\n", tid, lane * 4 + c_lut_idxG3[row][0], ymm0);
  }*/
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  min = __vminu4(min, 0x7F7F7F7F);

  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x13131313;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  // const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[0];

  // printf("tid = %d,row = %d\n", tid, row);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G4(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 4; // Gn = 4
  // if(threadIdx.x == 0 && blockIdx.x == 1)printf("1.3\n");
  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;


  const uint8_t row = MsgIdx - 1;

  // 5*384/4 = 480

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;

  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  // if(threadIdx.x == 0 && blockIdx.x == 1)printf("1.5\n");
  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][0] * 4);

  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);
  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------

  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x14141414;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;


  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);

}


__device__ __forceinline__ void cnProcKernel_BG1_int8_G5(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 5; // Gn = 5

  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 18 * 384 / 4 = 1728
  const uint baseShift = 18 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x15151515;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G6(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 6; // Gn = 6

  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 8 * 384 / 4 = 768

  const uint32_t baseShift = 8 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x16161616;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G7(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 7; // Gn = 7

  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;


  const uint8_t row = MsgIdx - 1;
 
  // 5 * 384 / 4 = 480

  const uint32_t baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][5] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x17171717;
  //uint8_t *BricksToBeMoved = (uint8_t *)&p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G8(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 8; // Gn = 8

  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 2 * 384 / 4 = 192
  const uint32_t baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = lane * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][5] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][6] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x18181818;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}


__device__ __forceinline__ void cnProcKernel_BG1_int8_G9(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ p_cnProcBuf,
                                             int8_t *__restrict__ p_cnProcBufRes,
                                             int8_t *__restrict__ p_bnProcBuf,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  //const uint8_t NUM = 9; // Gn = 9
  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 2 * 384 / 4 = 192

  const uint32_t baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][5] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][6] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][7] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x19191919;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}


__device__ __forceinline__ void cnProcKernel_BG1_int8_G10(const t_nrLDPC_lut *p_lut,
                                              const int8_t *__restrict__ p_cnProcBuf,
                                              int8_t *__restrict__ p_cnProcBufRes,
                                              int8_t *__restrict__ p_bnProcBuf,
                                              int8_t MsgIdx,
                                              int lane,
                                              uint8_t groupIdx,
                                              uint8_t CnIdx,
                                              int Zc)
{
  //const uint8_t NUM = 10; // Gn = 10
  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 1 * 384 / 4 = 96

  const uint32_t baseShift = 1 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][0] * 4);
  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][5] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][6] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][7] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][8] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x1a1a1a1a;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G19(const t_nrLDPC_lut *p_lut,
                                              const int8_t *__restrict__ p_cnProcBuf,
                                              int8_t *__restrict__ p_cnProcBufRes,
                                              int8_t *__restrict__ p_bnProcBuf,
                                              int8_t MsgIdx,
                                              int lane,
                                              uint8_t groupIdx,
                                              uint8_t CnIdx,
                                              int Zc)
{
  //const uint8_t NUM = 19; // Gn = 19
  // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this

  //const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  //const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  //const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;


  const uint8_t row = MsgIdx - 1; // row = 0,1,...,18
 
  // 4 * 384 / 4 = 384

  const uint32_t baseShift = 4 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  //// const uint srcByte = tid * 4;
  //const uint32_t p_ones = 0x01010101;
  //const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  //uint32_t *p_cnProcBufResBit;
  //p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4);

  sgn = __vxor4_first(0x01010101, &ymm0);
  min = __vabs4(ymm0);

  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][3] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][4] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][5] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][6] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][7] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][8] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][9] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][10] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][11] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][12] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][13] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][14] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][15] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][16] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][17] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------
  min = __vminu4(min, 0x7F7F7F7F);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0xbcbcbcbc;
  //uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx]);
}

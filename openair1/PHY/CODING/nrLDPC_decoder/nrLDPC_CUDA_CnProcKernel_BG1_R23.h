#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"

__device__ void cnProcKernel_BG1_R23_int8_G3(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ d_cnBufAll,
                                             int8_t *__restrict__ d_cnOutAll,
                                             int8_t *__restrict__ d_bnBufAll,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  const uint8_t NUM = 3; // Gn = 3
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint row = MsgIdx - 1; // row = 0,1,2  -> 3 BNs

  // 1*384/4 = 96
  const uint16_t c_lut_idxG3[3][2] = {{96, 192}, {0, 192}, {0, 96}};

  const uint baseShift = Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;

  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
  min = __vabs4(ymm0);

  // loop starts here
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);
  /*if(row == 0 && blockIdx.x == 0){
      printf("In thread %d, in address offset: %d, ymm0 = %02x\n", tid, lane * 4 + c_lut_idxG3[row][0], ymm0);
  }*/
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  min = __vminu4(min, maxLLR);

  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0x13131313;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  // const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[0];

  // printf("tid = %d,row = %d\n", tid, row);

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}

// R23 doesn't have G4,G5,G6

__device__ void cnProcKernel_BG1_R23_int8_G7(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ d_cnBufAll,
                                             int8_t *__restrict__ d_cnOutAll,
                                             int8_t *__restrict__ d_bnBufAll,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  const uint8_t NUM = 7; // Gn = 7

  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;


  const uint8_t row = MsgIdx - 1;
 
 // 3 * 384 / 4 = 480      //it's 3 here in R23 compared to R13
  const uint16_t c_lut_idxG7[7][6] = {

      {288, 576, 864, 1152, 1440, 1728},
      {0, 576, 864, 1152, 1440, 1728},
      {0, 288, 864, 1152, 1440, 1728},
      {0, 288, 576, 1152, 1440, 1728},
      {0, 288, 576, 864, 1440, 1728},
      {0, 288, 576, 864, 1152, 1440},
  };
  const uint32_t baseShift = 3 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  min = __vminu4(min, maxLLR);
  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0x17171717;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}

__device__ void cnProcKernel_BG1_R23_int8_G8(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ d_cnBufAll,
                                             int8_t *__restrict__ d_cnOutAll,
                                             int8_t *__restrict__ d_bnBufAll,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  const uint8_t NUM = 8; // Gn = 8

  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 2 * 384 / 4 = 192
  const uint16_t c_lut_idxG8[8][7] = {

      {192, 384, 576, 768, 960, 1152, 1344},
      {0, 384, 576, 768, 960, 1152, 1344},
      {0, 192, 576, 768, 960, 1152, 1344},
      {0, 192, 384, 768, 960, 1152, 1344},
      {0, 192, 384, 576, 960, 1152, 1344},
      {0, 192, 384, 576, 768, 1152, 1344},
      {0, 192, 384, 576, 768, 960, 1344},
      {0, 192, 384, 576, 768, 960, 1152}};
  const uint32_t baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = lane * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  min = __vminu4(min, maxLLR);
  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0x18181818;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}

__device__ void cnProcKernel_BG1_R23_int8_G9(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ d_cnBufAll,
                                             int8_t *__restrict__ d_cnOutAll,
                                             int8_t *__restrict__ d_bnBufAll,
                                             int8_t MsgIdx,
                                             int lane,
                                             uint8_t groupIdx,
                                             uint8_t CnIdx,
                                             int Zc)
{
  const uint8_t NUM = 9; // Gn = 9
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  /*
          if(tid == 0 && blockIdx.x == 0){
              printf("BG7 CN: p_cnProcBuf first all elements: ");
              for (int idx = 0; idx < 768; idx++)
              {
                  printf("%02x ", *(&p_cnProcBuf[idx]-384));
              }
              printf("\n");
              __syncthreads();
          }*/

  const uint8_t row = MsgIdx - 1;
 
  // 2 * 384 / 4 = 192
  const uint16_t c_lut_idxG9[9][8] = {

      {192, 384, 576, 768, 960, 1152, 1344, 1536},
      {0, 384, 576, 768, 960, 1152, 1344, 1536},
      {0, 192, 576, 768, 960, 1152, 1344, 1536},
      {0, 192, 384, 768, 960, 1152, 1344, 1536},
      {0, 192, 384, 576, 960, 1152, 1344, 1536},
      {0, 192, 384, 576, 768, 1152, 1344, 1536},
      {0, 192, 384, 576, 768, 960, 1344, 1536},
      {0, 192, 384, 576, 768, 960, 1152, 1536},
      {0, 192, 384, 576, 768, 960, 1152, 1344}};

  const uint32_t baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  min = __vminu4(min, maxLLR);
  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0x19191919;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}

__device__ void cnProcKernel_BG1_R23_int8_G10(const t_nrLDPC_lut *p_lut,
                                              const int8_t *__restrict__ d_cnBufAll,
                                              int8_t *__restrict__ d_cnOutAll,
                                              int8_t *__restrict__ d_bnBufAll,
                                              int8_t MsgIdx,
                                              int lane,
                                              uint8_t groupIdx,
                                              uint8_t CnIdx,
                                              int Zc)
{
  const uint8_t NUM = 10; // Gn = 10
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;

  const uint8_t row = MsgIdx - 1;
 
  // 1 * 384 / 4 = 96
  const uint16_t c_lut_idxG10[10][9] = {

      {96, 192, 288, 384, 480, 576, 672, 768, 864},
      {0, 192, 288, 384, 480, 576, 672, 768, 864},
      {0, 96, 288, 384, 480, 576, 672, 768, 864},
      {0, 96, 192, 384, 480, 576, 672, 768, 864},
      {0, 96, 192, 288, 480, 576, 672, 768, 864},
      {0, 96, 192, 288, 384, 576, 672, 768, 864},
      {0, 96, 192, 288, 384, 480, 672, 768, 864},
      {0, 96, 192, 288, 384, 480, 576, 768, 864},
      {0, 96, 192, 288, 384, 480, 576, 672, 864},
      {0, 96, 192, 288, 384, 480, 576, 672, 768}};

  const uint32_t baseShift = 1 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  // const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  min = __vminu4(min, maxLLR);
  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0x1a1a1a1a;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}

__device__ void cnProcKernel_BG1_R23_int8_G19(const t_nrLDPC_lut *p_lut,
                                              const int8_t *__restrict__ d_cnBufAll,
                                              int8_t *__restrict__ d_cnOutAll,
                                              int8_t *__restrict__ d_bnBufAll,
                                              int8_t MsgIdx,
                                              int lane,
                                              uint8_t groupIdx,
                                              uint8_t CnIdx,
                                              int Zc)
{
  const uint8_t NUM = 19; // Gn = 19
  // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this

  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  const int8_t *p_bnProcBuf = (const int8_t *)d_bnBufAll;


  const uint8_t row = MsgIdx - 1; // row = 0,1,...,18
 
  // 4 * 384 / 4 = 384
  const uint16_t c_lut_idxG19[19][18] = {

      {384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 4224, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4608, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4992, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 5376, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5760, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 6144, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6528, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6912},
      {0, 384, 768, 1152, 1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224, 4608, 4992, 5376, 5760, 6144, 6528}};

  const uint32_t baseShift = 4 * Zc * row; // offset pointed at different BN
  const uint32_t destByte = baseShift + lane * 4; // offset to different part inside different BN
  //// const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4);
  // if( blockIdx.x == 45 && threadIdx.x == 1){
  // printf("tid = %d, p_cnProcBuf = %p, p_cnProcBufRes = %p, p_cnProcBufResBit = %p, first ymm0 addr = %p\n", tid, p_cnProcBuf,
  // p_cnProcBufRes, p_cnProcBufResBit, (const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4));
  //}
  sgn = __vxor4(&p_ones, &ymm0);
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
  min = __vminu4(min, maxLLR);
  *p_cnProcBufResBit = __vsign4(&min, &sgn); // 0xbcbcbcbc;
  uint8_t *BricksToBeMoved = (uint8_t *)p_cnProcBufResBit;

  const uint16_t *lut_circShift_CNG = arrPos(p_lut->circShift[groupIdx], row);
  const uint32_t *lut_startAddrBnProcBuf_CNG = arrPos(p_lut->startAddrBnProcBuf[groupIdx], row);
  const uint8_t *lut_bnPosBnProcBuf_CNG = arrPos(p_lut->bnPosBnProcBuf[groupIdx], row);

  const int idxBn = lut_startAddrBnProcBuf_CNG[CnIdx] + lut_bnPosBnProcBuf_CNG[CnIdx] * Zc;

  moveBricks_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, BricksToBeMoved, Zc, lut_circShift_CNG[CnIdx], INVERSE, PUT_BRICKS);
}


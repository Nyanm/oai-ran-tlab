#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "nrLDPC_CUDA_public.h"
#include "nrLDPC_CUDA_lut.h"

__device__ __forceinline__ void cnProcKernel_BG1_int8_G3(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
  min = __vabs4(ymm0);

  // loop starts here
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);

  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //min = __vminu4(min, maxLLR);

  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G4(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][0] * 4);

  sgn = __vxor4(&ones, &ymm0);
  min = __vabs4(ymm0);
  //-------------------------loop starts here-------------------------------
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][1] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][2] * 4);
  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  //-------------------------------------------------------------------------

  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G5(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
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
  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G6(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

    ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][0] * 4);
    sgn = __vxor4(&ones, &ymm0);
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

    uint32_t BricksToBeMoved = __vsign4(&min, &sgn);
    moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);


}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G7(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

    ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][0] * 4);
    sgn = __vxor4(&ones, &ymm0);
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
    uint32_t BricksToBeMoved = __vsign4(&min, &sgn);
    moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
  
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G8(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG8[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
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
  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);
  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G9(const int8_t *__restrict__ p_cnProcBuf,
                                                         int8_t *__restrict__ p_bnProcBuf,
                                                         uint32_t row,
                                                         uint32_t lane,
                                                         uint32_t idxBn,
                                                         uint32_t circShift,
                                                         uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG9[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
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
  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn); // 0x19191919;

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G10(const int8_t *__restrict__ p_cnProcBuf,
                                                          int8_t *__restrict__ p_bnProcBuf,
                                                          uint32_t row,
                                                          uint32_t lane,
                                                          uint32_t idxBn,
                                                          uint32_t circShift,
                                                          uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG10[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
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
  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

__device__ __forceinline__ void cnProcKernel_BG1_int8_G19(const int8_t *__restrict__ p_cnProcBuf,
                                                          int8_t *__restrict__ p_bnProcBuf,
                                                          uint32_t row,
                                                          uint32_t lane,
                                                          uint32_t idxBn,
                                                          uint32_t circShift,
                                                          uint32_t Zc)
{
  uint32_t ymm0, sgn, min;
  const uint32_t ones = 0x01010101;
   
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4);

  sgn = __vxor4(&ones, &ymm0);
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
  //min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

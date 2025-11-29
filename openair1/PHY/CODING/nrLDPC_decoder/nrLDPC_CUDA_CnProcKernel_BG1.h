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
  const uint32_t maxLLR = 0x7F7F7F7F;

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
  sgn = __vxor4(&ones, &ymm0);
  min = __vabs4(ymm0);

  // loop starts here
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);

  min = __vminu4(min, __vabs4(ymm0));
  sgn = __vxor4(&sgn, &ymm0);
  min = __vminu4(min, maxLLR);

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
  const uint32_t maxLLR = 0x7F7F7F7F;

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

  min = __vminu4(min, maxLLR);
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);

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
    const uint32_t maxLLR = 0x7F7F7F7F;
    
    // 计算全局线程ID
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // =========================================================
    // DEBUG 分支：针对特定 TID 开启详细追踪
    // =========================================================
    if(tid == 16157) {
        const uint8_t* pY; // 用于打印 ymm0 的字节指针
        const uint8_t* pM; // 用于打印 min 的字节指针

        printf("\n=== DEBUG TID %d START (Row %d, Lane %d) ===\n", tid, row, lane);

        // --- Step 0 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][0] * 4);
        sgn = __vxor4(&ones, &ymm0);
        min = __vabs4(ymm0);
        
        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 0 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Step 1 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][1] * 4);
        min = __vminu4(min, __vabs4(ymm0));
        sgn = __vxor4(&sgn, &ymm0);

        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 1 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Step 2 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][2] * 4);
        min = __vminu4(min, __vabs4(ymm0));
        sgn = __vxor4(&sgn, &ymm0);

        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 2 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Step 3 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][3] * 4);
        min = __vminu4(min, __vabs4(ymm0));
        sgn = __vxor4(&sgn, &ymm0);

        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 3 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Step 4 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][4] * 4);
        min = __vminu4(min, __vabs4(ymm0));
        sgn = __vxor4(&sgn, &ymm0);

        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 4 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Step 5 ---
        ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG7[row][5] * 4);
        min = __vminu4(min, __vabs4(ymm0));
        sgn = __vxor4(&sgn, &ymm0);

        pY = (const uint8_t*)&ymm0; pM = (const uint8_t*)&min;
        printf("Step 5 (Load): Raw=[%02x %02x %02x %02x] | AbsMin=[%02x %02x %02x %02x]\n", 
               pY[0], pY[1], pY[2], pY[3], pM[0], pM[1], pM[2], pM[3]);

        // --- Final Processing ---
        min = __vminu4(min, maxLLR);
        uint32_t BricksToBeMoved = __vsign4(&min, &sgn);
        const uint8_t* bytes = (const uint8_t*)&BricksToBeMoved;

        printf("=== Final Result: [%02x %02x %02x %02x] ===\n\n", 
               bytes[0], bytes[1], bytes[2], bytes[3]);
        
        moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
    } 
    // =========================================================
    // 正常分支 (保持高性能)
    // =========================================================
    else {
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
        min = __vminu4(min, maxLLR);
        uint32_t BricksToBeMoved = __vsign4(&min, &sgn);
        moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
    }
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);
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
  const uint32_t maxLLR = 0x7F7F7F7F;

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
  min = __vminu4(min, maxLLR);
  uint32_t BricksToBeMoved = __vsign4(&min, &sgn);

  moveBricks_invput_circ((int8_t *)&p_bnProcBuf[idxBn], lane * 4, (uint8_t *)&BricksToBeMoved, Zc, circShift);
}

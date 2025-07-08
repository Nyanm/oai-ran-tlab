#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"


__device__ __forceinline__ uint32_t __vxor4(const uint32_t *a, uint32_t *b)
{
  return a[0] ^ b[0]; // increase accuracy
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{
  uint32_t mask = __vcmples4(b[0] | 0x01010101, 0); // 0xFF / 0x00 per‑byte
  uint32_t bneg = __vneg4(a[0]);
  return (mask & bneg) | (~mask & a[0]); // Compute ±magnitude in two steps
}

__device__ void cnProcKernel_int8_G3(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 3; // Gn = 3
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4) // NUM * Zc / 4 is the number of threads assigned to each block
    return;
  const uint row = tid / 96; // row = 0,1,2  -> 3 BNs
  const uint lane = tid % 96; // lane = 0,1,...,95 -> one thread in one of 96 process units
                              // and produce 1/96 of the Msg to one BN
  // 1*384/4 = 96
  const uint16_t c_lut_idxG3[3][2] = {{96, 192}, {0, 192}, {0, 96}};

  const uint baseShift = Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G4(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 4; // Gn = 3
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  // if(tid == 1){
  // printf("\nThis is block %d in G4", blockIdx.x);
  //}
  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
  // 5*384/4 = 480
  const uint16_t c_lut_idxG4[4][3] = {

      {480, 960, 1440},
      {0, 960, 1440},
      {0, 480, 1440},
      {0, 480, 960}};

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;

  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG4[row][0] * 4);

  sgn = __vxor4(&p_ones, &ymm0);
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
  // if(row == 0 && blockIdx.x == 0){
  // printf("In thread %d, result = %02x\n", tid, result);
  //}
}

__device__ void cnProcKernel_int8_G5(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 5; // Gn = 5
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
  // 18 * 384 / 4 = 1728
  const uint16_t c_lut_idxG5[5][4] = {

      {1728, 3456, 5184, 6912},
      {0, 3456, 5184, 6912},
      {0, 1728, 5184, 6912},
      {0, 1728, 3456, 6912},
      {0, 1728, 3456, 5184}};

  const uint baseShift = 18 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG5[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G6(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 6; // Gn = 6
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
  // 8 * 384 / 4 = 768
  const uint16_t c_lut_idxG6[6][5] = {

      {768, 1536, 2304, 3072, 3840},
      {0, 1536, 2304, 3072, 3840},
      {0, 768, 2304, 3072, 3840},
      {0, 768, 1536, 3072, 3840},
      {0, 768, 1536, 2304, 3840},
      {0, 768, 1536, 2304, 3072}};

  const uint baseShift = 8 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
  const uint32_t p_ones = 0x01010101;
  const uint32_t maxLLR = 0x7F7F7F7F;
  uint32_t ymm0, sgn, min;
  uint32_t *p_cnProcBufResBit;
  p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);
  ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG6[row][0] * 4);
  sgn = __vxor4(&p_ones, &ymm0);
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G7(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 7; // Gn = 7
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
  // 5 * 384 / 4 = 480
  const uint16_t c_lut_idxG7[7][6] = {

      {480, 960, 1440, 1920, 2400, 2880},
      {0, 960, 1440, 1920, 2400, 2880},
      {0, 480, 1440, 1920, 2400, 2880},
      {0, 480, 960, 1920, 2400, 2880},
      {0, 480, 960, 1440, 2400, 2880},
      {0, 480, 960, 1440, 1920, 2880},
      {0, 480, 960, 1440, 1920, 2400}};

  const uint baseShift = 5 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G8(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 8; // Gn = 8
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
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
  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G9(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 9; // Gn = 9
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;
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

  const uint row = tid / 96;
  const uint lane = tid % 96;
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

  const uint baseShift = 2 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G10(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid, int Zc)
{
  const uint8_t NUM = 10; // Gn = 10
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96;
  const uint lane = tid % 96;
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

  const uint baseShift = 1 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

__device__ void cnProcKernel_int8_G19(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int Tid, int Zc)
{
  const uint8_t NUM = 19; // Gn = 19
  // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll; // output pointer each block tackle with

  int tid = Tid + 912 * (blockIdx.x % 2); // same reason, now the following no need to change

  if (tid >= NUM * Zc / 4)
    return;

  const uint row = tid / 96; // row = 0,1,...,18
  const uint lane = tid % 96;
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

  const uint baseShift = 4 * Zc * row; // offset pointed at different BN
  const uint destByte = baseShift + lane * 4; // offset to different part inside different BN
  const uint srcByte = tid * 4;
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
  *p_cnProcBufResBit = __vsign4(&min, &sgn);
}

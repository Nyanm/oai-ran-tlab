#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

#define Q_SCALE 8.0
#define BG1_GRP0_CN 1
#define ZC 384 // for BG1 test only
#define CPU_ADDRESSING 1 // 0 means copy data into gpu memory, for common gpu; 1 for grace hopper which can read cpu memory directly
#define CUDA_STREAM 0 // 1 means use cudastream to run kernels in parallel; for grace hopper, GPU automatically run kernels in parallel
    //  so 0 is enough.
#define BIG_KERNEL 1

__constant__ static uint8_t d_lut_numBnInCnGroups_BG1_R13[9];
__constant__ static int d_lut_numThreadsEachCnGroupsNeed_BG1_R13[9];
//__constant__ static uint8_t d_lut_numCnInCnGroups_BG1_R13[9];

// === CUDA Error Checking ===
// Wrap any CUDA API call with CHECK(...) to automatically print error info with file and line number
// Example usage: CHECK(cudaMalloc(&ptr, size));
#define CHECK(call) ErrorCheck((call), __FILE__, __LINE__)

/**
 * @brief Checks CUDA error status and prints detailed diagnostic info if an error occurred.
 *
 * @param error_code The CUDA error code returned from a CUDA runtime API call.
 * @param filename   The name of the source file where the error occurred.
 * @param lineNumber The line number in the source file where the error occurred.
 * @return cudaError_t Returns the same error code passed in, for optional further handling.
 */
inline cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber)
{
  if (error_code != cudaSuccess) {
    printf("[CUDA ERROR] %s (%d): %s\nOccurred in file: %s at line %d\n",
           cudaGetErrorName(error_code),
           error_code,
           cudaGetErrorString(error_code),
           filename,
           lineNumber);
  }
  return error_code;
}

__device__ __forceinline__ uint32_t __vxor4(const uint32_t *a, uint32_t *b)
{
  return a[0] ^ b[0] ; // increase accuracy
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{

  uint32_t mask = __vcmples4(b[0] | 0x01010101, 0); // 0xFF / 0x00 per‑byte
  uint32_t bneg = __vneg4(a[0]);
  return (mask & bneg) | (~mask & a[0]); // Compute ±magnitude in two steps
}





__device__ void cnProcKernel_int8_G3(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G4(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G6(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G7(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G8(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G9(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int tid,int Zc)
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

__device__ void cnProcKernel_int8_G19(const int8_t *__restrict__ d_cnBufAll, int8_t *__restrict__ d_cnOutAll, int Tid,int Zc)
{
  const uint8_t NUM = 19; // Gn = 19
  // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this
  const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll ; // input pointer each block tackle with
  const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll ; // output pointer each block tackle with

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
    //if( blockIdx.x == 45 && threadIdx.x == 1){
    //printf("tid = %d, p_cnProcBuf = %p, p_cnProcBufRes = %p, p_cnProcBufResBit = %p, first ymm0 addr = %p\n", tid, p_cnProcBuf, p_cnProcBufRes, p_cnProcBufResBit, (const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4));
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

__global__ void cnProcKernel_int8_BIG(
    const int8_t* __restrict__ d_cnBufAll,
    int8_t* __restrict__ d_cnOutAll,
    const uint8_t* __restrict__ block_group_ids,
    const uint16_t* __restrict__ block_thread_counts,
    const uint32_t* __restrict__ block_input_offsets,
    const uint32_t* __restrict__ block_output_offsets,
    int Zc
) {
    int blk = blockIdx.x;
    int tid = threadIdx.x;

    uint8_t groupId = block_group_ids[blk];
    uint16_t blockSize = block_thread_counts[blk];
    uint32_t inOffset = block_input_offsets[blk];
    uint32_t outOffset = block_output_offsets[blk];

    if (tid >= blockSize) return;

    const int8_t* p_cnProcBuf = (const int8_t*)(d_cnBufAll + inOffset);
     int8_t* p_cnProcBufRes = (int8_t*)(d_cnOutAll + outOffset);
    //if(blk == 45 && tid == 64){
      //printf("d_cnBufAll = %p, d_cnOutAll = %p, p_cnProcBuf = %p, p_cnProcBufRes = %p, inOffset = %d, outOffset = %d \n", d_cnBufAll, d_cnOutAll, p_cnProcBuf, p_cnProcBufRes, inOffset, outOffset);
    //}

    switch(groupId) {
        case 0:
            cnProcKernel_int8_G3(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 1:
            cnProcKernel_int8_G4(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 2:
            cnProcKernel_int8_G5(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 3:
            cnProcKernel_int8_G6(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 4:
            cnProcKernel_int8_G7(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 5:
            cnProcKernel_int8_G8(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 6:
            cnProcKernel_int8_G9(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 7:
            cnProcKernel_int8_G10(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
        case 8:
            cnProcKernel_int8_G19(p_cnProcBuf, p_cnProcBufRes, tid, Zc);
            break;
    }
}



void nrLDPC_cnProc_BG1_cuda_core(const t_nrLDPC_lut *p_lut, int8_t *cnProcBuf, int8_t *cnProcBufRes, int Z)
{
  const uint8_t h_lut_numBnInCnGroups_BG1_R13[]        = {3  , 4  , 5  , 6  , 7  , 8  , 9  , 10 , 19};
  const int h_lut_numThreadsEachCnGroupsNeed_BG1_R13[] = {288, 384, 480, 576, 672, 768, 864, 960, 1824};
  const uint8_t h_lut_numCnInCnGroups_BG1_R13[]        = {1  , 5  , 18 , 8  , 5  , 2  , 2  , 1  , 4   };

  // const uint8_t *lut_numCnInCnGroups = (const uint8_t *)p_lut->numCnInCnGroups;
  const uint32_t *lut_startAddrCnGroups = lut_startAddrCnGroups_BG1;



  const int numGroups = 9;

#if BIG_KERNEL

static const uint8_t  h_block_group_ids[50] = {
    0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    3, 3, 4, 4, 4, 4, 4, 5, 5, 6,
    6, 7, 8, 8, 8, 8, 8, 8, 8, 8
};

static const uint16_t h_block_thread_counts[50] = {
    288, 384, 384, 384, 384, 384, 480, 480, 480, 480,
    480, 480, 480, 480, 480, 480, 480, 480, 480, 480,
    480, 480, 480, 480, 576, 576, 576, 576, 576, 576,
    576, 576, 672, 672, 672, 672, 672, 768, 768, 864,
    864, 960, 912, 912, 912, 912, 912, 912, 912, 912
};

static const uint32_t h_block_input_offsets[50] = {
      0,   1152,   1536,   1920,   2304,   2688,   8832,   9216,   9600,   9984,
   10368,  10752,  11136,  11520,  11904,  12288,  12672,  13056,  13440,  13824,
   14208,  14592,  14976,  15360,  43392,  43776,  44160,  44544,  44928,  45312,
   45696,  46080,  61824,  62208,  62592,  62976,  63360,  75264,  75648,  81408,
   81792,  88320,  92160,  92160,  92544,  92544,  92928,  92928,  93312,  93312
};

static const uint32_t h_block_output_offsets[50] = {
      0,   1152,   1536,   1920,   2304,   2688,   8832,   9216,   9600,   9984,
   10368,  10752,  11136,  11520,  11904,  12288,  12672,  13056,  13440,  13824,
   14208,  14592,  14976,  15360,  43392,  43776,  44160,  44544,  44928,  45312,
   45696,  46080,  61824,  62208,  62592,  62976,  63360,  75264,  75648,  81408,
   81792,  88320,  92160,  92160,  92544,  92544,  92928,  92928,  93312,  93312
};

//printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);


int maxBlockSize = 960; // G10最大是960
dim3 gridDim(50);
dim3 blockDim(maxBlockSize);

cnProcKernel_int8_BIG<<<gridDim, blockDim>>>(
    cnProcBuf,
    cnProcBufRes,
    h_block_group_ids,
    h_block_thread_counts,
    h_block_input_offsets,
    h_block_output_offsets,
    Z
);
//CHECK(cudaGetLastError());

#else
#if !CUDA_STREAM
  // No cuda stream using
  for (int i = 0; i < numGroups; ++i) {
    p_cnProcBuf = cnProcBuf + lut_startAddrCnGroups[i];
    p_cnProcBufRes = cnProcBufRes + lut_startAddrCnGroups[i];
    // printf("\nlut_startAddrCnGroups[%d]: %d\n", i, (int)lut_startAddrCnGroups[i]);

    // printf("In i = %d, p_cnProcBuf = %p, p_cnProcBufRes = %p", i, (void *)p_cnProcBuf, (void *)p_cnProcBufRes);

    switch (i) {
      case 0:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);

        // print p_cnProcBuf和p_cnProcBufRes firt int8_t element
        /*printf("BG3: p_cnProcBuf first all elements: ");
        for (int idx = 0; idx < 1152; idx++)
        {
            printf("%x ", p_cnProcBuf[idx]);
        }*/
        // printf("\n");

        // cudaPointerAttributes attr;
        // cudaPointerGetAttributes(&attr, p_cnProcBuf);
        // printf("p_cnProcBuf is on %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");
        cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);

        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        /*
                    printf("p_cnProcBufRes first 10 elements: ");
                    for (int idx = 0; idx < 1152; idx++)
                    {
                        printf("%d ", p_cnProcBufRes[idx]);
                    }
                    printf("\n");*/

        break;
      case 1:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        // h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 2:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 3:
        //("launching kernel[%d]: grid=%d, block=%d\n", i,
        //      h_lut_numCnInCnGroups_BG1_R13[i],
        //      h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 4:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 5:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 6:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 7:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                 p_cnProcBufRes,
                                                                                                                 Z);
        /*printf("BG3: p_cnProcBuf first all elements: ");
for (int idx = 0; idx < 1152; idx++)
{
printf("%x ", p_cnProcBuf[idx]);
}
printf("\n");*/
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 8:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        //  Group 19: split into 2x blocks, half threads
        cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2, h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
    }
  }
#else
  // Create CUDA streams for concurrent kernel execution
  cudaStream_t streams[numGroups];
  for (int i = 0; i < numGroups; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  // Launch each group kernel on a separate stream
  for (int i = 0; i < numGroups; ++i) {
    p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[i]];
    p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[i]];

    switch (i) {
      case 0:
        cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 1:
        cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 2:
        cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 3:
        cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 4:
        cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 5:
        cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 6:
        cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 7:
        cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 8:
        // Group 19 requires more than 1024 threads, so split into 2x blocks, half threads
        cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2,
                                h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2,
                                0,
                                streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    }
  }

  // Wait for all streams to finish
  for (int i = 0; i < numGroups; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]); // Release stream resources
  }

#endif
#endif
  // CHECK(cudaGetLastError());
  //CHECK(cudaDeviceSynchronize());
}

// CUDA wrapper function: external interface identical to the original C version
extern "C" void nrLDPC_cnProc_BG1_cuda(const t_nrLDPC_lut *p_lut, int8_t *cnProcBuf, int8_t *cnProcBufRes, uint16_t Z)
{
  // printf("CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
#if CPU_ADDRESSING
  //printf("\nVery very first cnProcBuf = %p, cnProcBufRes = %p \n", cnProcBuf, cnProcBufRes);
  nrLDPC_cnProc_BG1_cuda_core(p_lut, cnProcBuf, cnProcBufRes, (int)Z);

#else
  // printf("Here CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
  size_t cnProcBuf_size = 200000 /* buffer size for cnProcBuf  */;
  size_t cnProcBufRes_size = 200000 /* buffer size for cnProcBufRes  */;

  // use Unified Memory
  int8_t *d_cnProcBuf = nullptr;
  int8_t *d_cnProcBufRes = nullptr;

  cudaError_t err;

  // ask for unified memory
  err = cudaMallocManaged(&d_cnProcBuf, cnProcBuf_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBuf failed: %s\n", cudaGetErrorString(err));
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBuf success, d_cnProcBuf = %p\n", (void *)d_cnProcBuf);
  }

  err = cudaMallocManaged(&d_cnProcBufRes, cnProcBufRes_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBufRes failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_cnProcBuf);
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBufRes success, d_cnProcBufRes = %p\n", (void *)d_cnProcBufRes);
  }

  //
  memcpy(d_cnProcBuf, cnProcBuf, cnProcBuf_size);
  memset(d_cnProcBufRes, 0, cnProcBufRes_size);

  // kernel function
  nrLDPC_cnProc_BG1_cuda_core(p_lut, d_cnProcBuf, d_cnProcBufRes, (int)Z);

  memcpy(cnProcBufRes, d_cnProcBufRes, cnProcBufRes_size);

  // free memory
  cudaFree(d_cnProcBuf);
  cudaFree(d_cnProcBufRes);
#endif
  // cudaPointerAttributes attr;
  // cudaPointerGetAttributes(&attr, d_cnProcBuf);
  // printf("d_cnProcBuf is on %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");
  cudaDeviceSynchronize();
}

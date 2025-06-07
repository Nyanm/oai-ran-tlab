#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
//#include "nrLDPC_types.h"  // 包含 t_nrLDPC_lut 类型
//#include "nrLDPC_cnProc.h" // 如果已有函数声明

#define Q_SCALE 8.0
#define BG1_GRP0_CN 1
#define ZC 384 // for BG1 test only

__constant__ static uint8_t d_lut_numBnInCnGroups_BG1_R13[9];
__constant__ static int d_lut_numThreadsEachCnGroupsNeed_BG1_R13[9];
__constant__ static uint8_t d_lut_numCnInCnGroups_BG1_R13[9];
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
inline cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
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
    return *a ^ *b; //| 0x01010101;
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{
    uint32_t mask = __vcmples4(b[0], 0); // 0xFF / 0x00 per‑byte
    uint32_t bneg = __vneg4(a[0]);
    return (mask & bneg ) | (~mask & a[0] ); // Compute ±magnitude in two steps
}

__global__ void cnProcKernel_int8_G3(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    printf("G3");
    const uint8_t NUM = 3;                                     // Gn = 3
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)    // NUM * Zc / 4 is the number of threads assigned to each block
        return;

    const uint row = tid / 96;  // row = 0,1,2  -> 3 BNs
    const uint lane = tid % 96; // lane = 0,1,...,95 -> one thread in one of 96 process units
                                // and produce 1/96 of the Msg to one BN
    // 288/3 = 96
    const uint8_t c_lut_idxG3[3][2] = {
        {96, 192},
        {0, 192},
        {0, 96}};
    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;
    // uint32_t p_cnProcBufBit;
    
    // uint32_t input_BN1 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
    // uint32_t input_BN2 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);

    p_cnProcBufResBit = (uint32_t *)(p_cnProcBufRes + destByte);

    //if(tid ==1){
    //printf("\n**************We are using cuda in decoder now*******************\n");}
    ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][0] * 4);
    printf("thread%d:  ymm = %x  ", tid, ymm0);
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    // loop starts here
    ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG3[row][1] * 4);
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);

    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
    printf("thread%d&result:  %x  ", tid, result);
}

__global__ void cnProcKernel_int8_G4(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 4;                                     // Gn = 3
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 384/4 = 96
    const uint16_t c_lut_idxG4[4][3] = {
        {96, 192, 288},
        {0, 192, 288},
        {0, 96, 192},
        {0, 96, 288}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G5(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 5;                                     // Gn = 5
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 480/5 = 96
    const uint16_t c_lut_idxG5[5][4] = {
        {96, 192, 288, 384},
        {0, 192, 288, 384},
        {0, 96, 288, 384},
        {0, 96, 192, 384},
        {0, 96, 192, 288}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;


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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G6(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 6;                                     // Gn = 6
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 576/6 = 96
    const uint16_t c_lut_idxG6[6][5] = {
        {96, 192, 288, 384, 480},
        {0, 192, 288, 384, 480},
        {0, 96, 288, 384, 480},
        {0, 96, 192, 384, 480},
        {0, 96, 192, 288, 480},
        {0, 96, 192, 288, 384}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G7(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 7;                                     // Gn = 7
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 672/7 = 96
    const uint16_t c_lut_idxG7[7][6] = {
        {96, 192, 288, 384, 480, 576},
        {0, 192, 288, 384, 480, 576},
        {0, 96, 288, 384, 480, 576},
        {0, 96, 192, 384, 480, 576},
        {0, 96, 192, 288, 480, 576},
        {0, 96, 192, 288, 384, 576},
        {0, 96, 192, 288, 384, 480}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G8(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 8;                                     // Gn = 8
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 768/8 = 96
    const uint16_t c_lut_idxG8[8][7] = {
        {96, 192, 288, 384, 480, 576, 672},
        {0, 192, 288, 384, 480, 576, 672},
        {0, 96, 288, 384, 480, 576, 672},
        {0, 96, 192, 384, 480, 576, 672},
        {0, 96, 192, 288, 480, 576, 672},
        {0, 96, 192, 288, 384, 576, 672},
        {0, 96, 192, 288, 384, 480, 672},
        {0, 96, 192, 288, 384, 480, 576}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G9(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 9;                                     // Gn = 9
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 864/9 = 96
    const uint16_t c_lut_idxG9[9][8] = {
        {96, 192, 288, 384, 480, 576, 672, 768},
        {0, 192, 288, 384, 480, 576, 672, 768},
        {0, 96, 288, 384, 480, 576, 672, 768},
        {0, 96, 192, 384, 480, 576, 672, 768},
        {0, 96, 192, 288, 480, 576, 672, 768},
        {0, 96, 192, 288, 384, 576, 672, 768},
        {0, 96, 192, 288, 384, 480, 672, 768},
        {0, 96, 192, 288, 384, 480, 576, 768},
        {0, 96, 192, 288, 384, 480, 576, 672}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G10(const int8_t *__restrict__ d_cnBufAll,
                                      int8_t *__restrict__ d_cnOutAll,
                                      int Zc)
{
    const uint8_t NUM = 10;                                    // Gn = 10
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + blockIdx.x * Zc;   // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + blockIdx.x * Zc;   // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 960/10 = 96
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

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

__global__ void cnProcKernel_int8_G19(const int8_t *__restrict__ d_cnBufAll,
                                      int8_t *__restrict__ d_cnOutAll,
                                      int Zc)
{
    const uint8_t NUM = 19; // Gn = 19
    // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this
    const int8_t *p_cnProcBuf = (const int8_t *)d_cnBufAll + (int)(blockIdx.x / 2) * Zc;         // input pointer each block tackle with
    const int8_t *p_cnProcBufRes = (const int8_t *)d_cnOutAll + (int)(blockIdx.x / 2) * Zc;      // output pointer each block tackle with

    int tid = threadIdx.x + 48 * blockIdx.x % 2; // same reason, now the following no need to change
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 1824/19 = 96
    const uint16_t c_lut_idxG19[19][18] = {
        {96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 1056, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1152, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1248, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1344, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1440, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1536, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1632, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1728},
        {0, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440, 1536, 1632}};

    const uint baseShift = Zc * row;   // offset pointed at different BN
    const uint destByte = baseShift + lane  * 4;  // offset to different part inside different BN
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    uint32_t* p_cnProcBufResBit;

    ymm0 = *(const uint32_t *)(p_cnProcBuf + lane * 4 + c_lut_idxG19[row][0] * 4);
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
    uint32_t result = __vsign4(&min, &sgn);
    *p_cnProcBufResBit = result;
}

// CUDA wrapper function: external interface identical to the original C version
extern "C"
void nrLDPC_cnProc_BG1_cuda(const t_nrLDPC_lut *p_lut,
                       int8_t *cnProcBuf,
                       int8_t *cnProcBufRes,
                       uint16_t Z)
{
    const uint8_t h_lut_numBnInCnGroups_BG1_R13[] = {3, 4, 5, 6, 7, 8, 9, 10, 19};
    const int h_lut_numThreadsEachCnGroupsNeed_BG1_R13[] = {288, 384, 480, 576, 672, 768, 864, 960, 1824};
    const uint8_t h_lut_numCnInCnGroups_BG1_R13[] = {1, 5, 18, 8, 5, 2, 2, 1, 4};

    const uint8_t *lut_numCnInCnGroups = (const uint8_t *)p_lut->numCnInCnGroups;
    const uint8_t *lut_startAddrCnGroups = (const uint8_t *)p_lut->startAddrCnGroups;

    const int numGroups = 9;
    cudaStream_t streams[numGroups];

    int8_t *p_cnProcBuf;
    int8_t *p_cnProcBufRes;

    // Create CUDA streams for concurrent kernel execution
    for (int i = 0; i < numGroups; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Launch each group kernel on a separate stream
    for (int i = 0; i < numGroups; ++i)
    {
        p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[i]];
        p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[i]];

        switch (i)
        {
        case 0:
            cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 1:
            cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 2:
            cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 3:
            cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 4:
            cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 5:
            cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 6:
            cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 7:
            cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        case 8:
            // Group 19 requires more than 1024 threads, so split into 2x blocks, half threads
            cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2, h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2, 0, streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
            break;
        }
    }

    // Wait for all streams to finish
    for (int i = 0; i < numGroups; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]); // Release stream resources
	printf("CUDA kernel synchronized");
    }

/*


//No cuda stream using
for (int i = 0; i < numGroups; ++i)
{
    p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[i]];
    p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[i]];

    switch (i)
    {
    case 0:
        cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 1:
        cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 2:
        cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 3:
        cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 4:
        cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 5:
        cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 6:
        cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 7:
        cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    case 8:
        // Group 19: split into 2x blocks, half threads
        cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2, h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    }
}
*/
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
}

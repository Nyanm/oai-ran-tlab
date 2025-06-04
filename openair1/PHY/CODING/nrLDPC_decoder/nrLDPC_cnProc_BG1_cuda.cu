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

extern "C" void LDPCinit(void) {
// empty
}


__device__ __forceinline__ uint32_t __vxor4(const uint32_t *b, uint32_t *a)
{
    return *b ^ *a; //| 0x01010101;
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t *b, uint32_t *a)
{
    uint32_t mask = __vcmples4(a[0], 0); // 0xFF / 0x00 per‑byte
    uint32_t bneg = __vneg4(b[0]);
    return (mask & bneg) | (~mask & b[0]); // Compute ±magnitude in two steps
}

__global__ void cnProcKernel_int8_G3(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 3;                                     // Gn = 3
    const int8_t *d_in = (const int8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    int8_t *d_out = (int8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

    int tid = threadIdx.x;
    if (tid >= NUM * Zc / 4)
        return;

    const uint row = tid / 96;
    const uint lane = tid % 96;
    // 288/3 = 96
    const uint8_t c_lut_idxG3[3][2] = {
        {96, 192},
        {0, 192},
        {0, 96}};
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    if(tid ==1){
    printf("/n**************We are using cuda in decoder now*******************/n")

}
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG3[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    // loop strats here
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG3[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);

    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G4(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 4;                                     // Gn = 3
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG4[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG4[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG4[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G5(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 5;                                     // Gn = 5
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG5[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG5[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG5[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG5[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G6(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 6;                                     // Gn = 6
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;
    if(tid == 1){
    printf("decoder is using cuda now!");
}
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG6[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG6[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG6[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG6[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG6[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G7(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 7;                                     // Gn = 7
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG7[row][5] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G8(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 8;                                     // Gn = 8
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][5] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG8[row][6] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G9(const int8_t *__restrict__ d_cnBufAll,
                                     int8_t *__restrict__ d_cnOutAll,
                                     int Zc)
{
    const uint8_t NUM = 9;                                     // Gn = 9
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][5] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][6] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG9[row][7] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G10(const int8_t *__restrict__ d_cnBufAll,
                                      int8_t *__restrict__ d_cnOutAll,
                                      int Zc)
{
    const uint8_t NUM = 10;                                    // Gn = 10
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][5] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][6] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][7] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG10[row][8] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

__global__ void cnProcKernel_int8_G19(const int8_t *__restrict__ d_cnBufAll,
                                      int8_t *__restrict__ d_cnOutAll,
                                      int Zc)
{
    const uint8_t NUM = 19; // Gn = 19
    // Here the block 0 and block 1, block 2 and block 3, ... are doing the same thing, so we use blockIdx.x/2 to tackle this
    const uint8_t *d_in = (const uint8_t *)d_cnBufAll + blockIdx.x / 2 * NUM * Zc; // input pointer each block tackle with
    uint8_t *d_out = (uint8_t *)d_cnOutAll + blockIdx.x / 2 * NUM * Zc;      // output pointer each block tackle with

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
    const uint baseShift = Zc * blockIdx.x;
    const uint destByte = (baseShift + (lane & 0x2F)) * 4;
    const uint srcByte = tid * 4;
    const uint32_t p_ones = 0x01010101;
    const uint32_t maxLLR = 0x7F7F7F7F;
    uint32_t ymm0, sgn, min;

    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][0] * 4));
    sgn = __vxor4(&p_ones, &ymm0);
    min = __vabs4(ymm0);

    //-------------------------loop starts here-------------------------------
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][1] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][2] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][3] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][4] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][5] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][6] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][7] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][8] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][9] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][10] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][11] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][12] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][13] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][14] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][15] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][16] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    ymm0 = *((const uint32_t *)(d_in + lane * 4 + c_lut_idxG19[row][17] * 4));
    min = __vminu4(min, __vabs4(ymm0));
    sgn = __vxor4(&sgn, &ymm0);
    //-------------------------------------------------------------------------
    min = __vminu4(min, maxLLR);
    uint32_t result = __vsign4(&min, &sgn);
    *((uint32_t *)(d_out + destByte)) = result;
}

// CUDA wrapper function: external interface identical to the original C version
void nrLDPC_cnProc_BG1(const t_nrLDPC_lut *p_lut,
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
    }
}

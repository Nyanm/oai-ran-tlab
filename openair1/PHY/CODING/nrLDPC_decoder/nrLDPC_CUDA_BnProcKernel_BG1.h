#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

__device__ void gpu_sleep(unsigned int cycles)
{
  clock_t start = clock();
  while ((clock() - start) < cycles) {
    // Busy wait
  }
}
// ======================= //
//  BN_PROC PC Kernel BG1  //
// ======================= //

template<int NUM>
__device__ __forceinline__ void bnProcPcKernel_BG1_int8_NUM(
    const int8_t *__restrict__ d_bnProcBuf,
    const int8_t *__restrict__ d_bnProcBufRes, 
    const int8_t *__restrict__ d_llrProcBuf,
    const int8_t *__restrict__ d_llrRes,
    int8_t lane,
    int8_t BnIdx,
    int8_t GrpNum,
    int Zc)
{
    // --- Compute buffer base pointers ---
    int8_t *d_bnProcBuf_BnIdx  = (int8_t *)(d_bnProcBuf  + (BnIdx - 1) * Zc);
    int8_t *d_llrProcBuf_BnIdx = (int8_t *)(d_llrProcBuf + (BnIdx - 1) * Zc);
    int8_t *d_llrRes_BnIdx     = (int8_t *)(d_llrRes     + (BnIdx - 1) * Zc);

    int32_t *bnProcBufPtr = (int32_t *)(d_bnProcBuf_BnIdx + lane * 4);
    int32_t MsgSum = bnProcBufPtr[0];

    // --- Unrolled summation over NUM-1 groups ---
#pragma unroll
    for (int i = 1; i < NUM; ++i) {
        int32_t val = bnProcBufPtr[(GrpNum * i * Zc) / 4];
        MsgSum = __vaddss4(MsgSum, val);
    }

    // --- Add LLR and write result ---
    int32_t llrData = *(const int32_t *)(d_llrProcBuf_BnIdx + lane * 4);
    int32_t result  = __vaddss4(MsgSum, llrData);
    *(int32_t *)(d_llrRes_BnIdx + lane * 4) = result;
}



__device__ __forceinline__ void bnProcPcKernel_BG1_int8_Gn(
    const int8_t *__restrict__ d_bnProcBuf,
    const int8_t *__restrict__ d_bnProcBufRes,
    const int8_t *__restrict__ d_llrProcBuf,
    const int8_t *__restrict__ d_llrRes,
    int8_t lane,
    int8_t GrpIdx,
    int8_t BnIdx,
    int8_t GrpNum,
    int Zc)
{
    switch (GrpIdx)
    {
    case 1:  bnProcPcKernel_BG1_int8_NUM<1 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 2:  bnProcPcKernel_BG1_int8_NUM<2 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 3:  bnProcPcKernel_BG1_int8_NUM<3 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 4:  bnProcPcKernel_BG1_int8_NUM<4 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 5:  bnProcPcKernel_BG1_int8_NUM<5 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 6:  bnProcPcKernel_BG1_int8_NUM<6 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 7:  bnProcPcKernel_BG1_int8_NUM<7 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 8:  bnProcPcKernel_BG1_int8_NUM<8 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 9:  bnProcPcKernel_BG1_int8_NUM<9 >(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 10: bnProcPcKernel_BG1_int8_NUM<10>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 11: bnProcPcKernel_BG1_int8_NUM<11>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 12: bnProcPcKernel_BG1_int8_NUM<12>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 13: bnProcPcKernel_BG1_int8_NUM<13>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 14: bnProcPcKernel_BG1_int8_NUM<14>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 15: bnProcPcKernel_BG1_int8_NUM<15>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 16: bnProcPcKernel_BG1_int8_NUM<16>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 17: bnProcPcKernel_BG1_int8_NUM<17>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 18: bnProcPcKernel_BG1_int8_NUM<18>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 19: bnProcPcKernel_BG1_int8_NUM<19>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 20: bnProcPcKernel_BG1_int8_NUM<20>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 21: bnProcPcKernel_BG1_int8_NUM<21>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 22: bnProcPcKernel_BG1_int8_NUM<22>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 23: bnProcPcKernel_BG1_int8_NUM<23>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 24: bnProcPcKernel_BG1_int8_NUM<24>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 25: bnProcPcKernel_BG1_int8_NUM<25>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 26: bnProcPcKernel_BG1_int8_NUM<26>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 27: bnProcPcKernel_BG1_int8_NUM<27>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 28: bnProcPcKernel_BG1_int8_NUM<28>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 29: bnProcPcKernel_BG1_int8_NUM<29>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    case 30: bnProcPcKernel_BG1_int8_NUM<30>(d_bnProcBuf, d_bnProcBufRes, d_llrProcBuf, d_llrRes, lane, BnIdx, GrpNum, Zc); break;
    default: break;
    }
}


__device__ void bnProcKernel_BG1_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
                                         const int8_t *__restrict__ d_bnProcBufRes,
                                         const int8_t *__restrict__ d_llrProcBuf,
                                         const int8_t *__restrict__ d_llrRes,
                                         int8_t lane,
                                         int8_t GrpIdx,
                                         int8_t MsgIdx,
                                         int8_t BnIdx,
                                         int8_t GrpNum,
                                         int Zc)
{
  const uint8_t NUM = (const uint8_t)GrpIdx;

  int8_t *d_bnProcBuf_BnIdx = (int8_t *)(d_bnProcBuf + (BnIdx - 1) * Zc);
  int8_t *d_bnProcBufRes_BnIdx = (int8_t *)(d_bnProcBufRes + (BnIdx - 1) * Zc);
  int8_t *d_llrProcBuf_BnIdx = (int8_t *)(d_llrProcBuf + (BnIdx - 1) * Zc);
  int8_t *d_llrRes_BnIdx = (int8_t *)(d_llrRes + (BnIdx - 1) * Zc);

  int32_t ymm0Res = *(const int32_t *)(d_llrRes_BnIdx + lane * 4);

  int32_t prevMsg = *(const int32_t *)(d_bnProcBuf_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4);

  int32_t MsgRes = __vsubss4(ymm0Res, prevMsg);

  *(int32_t *)(d_bnProcBufRes_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4) = MsgRes;
}

__device__ void bnProcKernel_BG1_int8_Gn_United(const int8_t *__restrict__ d_bnProcBuf,
                                                const int8_t *__restrict__ d_bnProcBufRes,
                                                const int8_t *__restrict__ d_llrProcBuf,
                                                const int8_t *__restrict__ d_llrRes,
                                                int8_t lane,
                                                int8_t GrpIdx,
                                                int8_t MsgIdx,
                                                int8_t BnIdx,
                                                int8_t GrpNum,
                                                int Zc)
// cg::grid_group grid)
{
  const uint8_t NUM = (const uint8_t)GrpIdx;

  int8_t *d_bnProcBuf_BnIdx = (int8_t *)(d_bnProcBuf + (BnIdx - 1) * Zc);
  int8_t *d_bnProcBufRes_BnIdx = (int8_t *)(d_bnProcBufRes + (BnIdx - 1) * Zc);
  int8_t *d_llrProcBuf_BnIdx = (int8_t *)(d_llrProcBuf + (BnIdx - 1) * Zc);
  int8_t *d_llrRes_BnIdx = (int8_t *)(d_llrRes + (BnIdx - 1) * Zc);

  if (MsgIdx == 1) {
    int32_t *bnProcBufPtr = (int32_t *)(d_bnProcBuf_BnIdx + lane * 4);

    int32_t MsgSum = bnProcBufPtr[0];

    for (uint8_t i = 1; i < NUM; i++) {
      int32_t ymm0 = bnProcBufPtr[(GrpNum * i * Zc) / 4];
      MsgSum = __vaddss4(MsgSum, ymm0);
    }

    int32_t llrData = *(const int32_t *)(d_llrProcBuf_BnIdx + lane * 4);

    int32_t ymm0Res = __vaddss4(MsgSum, llrData);

    *(int32_t *)(d_llrRes_BnIdx + lane * 4) = ymm0Res;
  }

  __syncthreads();

  int32_t ymm0Res = *(const int32_t *)(d_llrRes_BnIdx + lane * 4);

  int32_t prevMsg = *(const int32_t *)(d_bnProcBuf_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4);

  int32_t MsgRes = __vsubss4(ymm0Res, prevMsg);

  *(int32_t *)(d_bnProcBufRes_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4) = MsgRes;

  // --------------------------
  // check MsgRes == 0 and print
  // --------------------------
  /*if (MsgRes == 0)
  {
      printf(
          "bnProcKernel_int8_Gn Debug | lane=%d | GrpIdx=%d | MsgIdx=%d | BnIdx=%d | GrpNum=%d | Zc=%d | ymm0Res=0x%08x |
  prevMsg=0x%08x | MsgRes=0x%08x\n", lane, GrpIdx, MsgIdx, BnIdx, GrpNum, Zc, ymm0Res, prevMsg, MsgRes
      );
  }*/
}
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

__device__ void bnProcPcKernel_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
                                       const int8_t *__restrict__ d_bnProcBufRes,
                                       const int8_t *__restrict__ d_llrProcBuf,
                                       const int8_t *__restrict__ d_llrRes,
                                       int8_t lane,
                                       int8_t GrpIdx,
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

  // First part: update llrRes if MsgIdx == 1
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

__device__ void bnProcKernel_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
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

__device__ void bnProcKernel_int8_Gn_United(const int8_t *__restrict__ d_bnProcBuf,
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
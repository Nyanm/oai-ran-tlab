#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

// ======================= //
//  BN_PROC PC Kernel BG1  //
// ======================= //

template <int NUM>
__device__ __forceinline__ void bnProcKernelMerge_BG1_int8_NUM(const int8_t *__restrict__ d_bnProcBuf,
                                                               int8_t *__restrict__ d_cnProcBuf,
                                                               const int8_t *__restrict__ d_llrProcBuf,
                                                               int8_t *__restrict__ d_llrRes,
                                                               uint32_t lane,
                                                               uint32_t MsgIdx,
                                                               uint32_t BnIdx,
                                                               uint32_t GrpNum,
                                                               uint32_t circShift,
                                                               uint32_t Zc)
{
  const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
  uint32_t prevIdxWords = (MsgIdx * GrpNum * Zc) >> 2;
  uint32_t prev = bnProcBufPtr[prevIdxWords];

  // ---- Unrolled accumulation ----
  uint32_t MsgSum = bnProcBufPtr[0];
  uint32_t off = (GrpNum * Zc) >> 2;
#pragma unroll
  for (int i = 1; i < NUM; ++i) {
    bnProcBufPtr += off;
    MsgSum = __vaddss4(MsgSum, *bnProcBufPtr);
  }

  // ---- Compute llrRes ----

  int32_t computed_llrRes = __vaddss4(MsgSum, ((const int32_t *)(d_llrProcBuf))[lane]);
  //  Only write to llrRes when MsgIdx == 1
  if (MsgIdx == 0) {
    ((int32_t *)(d_llrRes))[lane] = computed_llrRes;
  }

  uint32_t BricksToBeGet = __vsubss4(computed_llrRes, prev);
  // ---- Write result ----
  moveBricks_forput_circ(d_cnProcBuf, lane * 4, (uint8_t *)&BricksToBeGet, Zc, circShift);
}

__device__ __forceinline__ void bnProcKernelMerge_BG1_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
                                                              int8_t *__restrict__ d_cnProcBuf,
                                                              const int8_t *__restrict__ d_llrProcBuf,
                                                              int8_t *__restrict__ d_llrRes,
                                                              uint32_t lane,
                                                              uint32_t GrpIdx,
                                                              uint32_t MsgIdx,
                                                              uint32_t BnIdx,
                                                              uint32_t GrpNum,
                                                              uint32_t circShift,
                                                              uint32_t Zc)
{
  switch (GrpIdx) {
    case 1:
      bnProcKernelMerge_BG1_int8_NUM<
          1>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 2:
      bnProcKernelMerge_BG1_int8_NUM<
          2>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 3:
      bnProcKernelMerge_BG1_int8_NUM<
          3>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 4:
      bnProcKernelMerge_BG1_int8_NUM<
          4>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 5:
      bnProcKernelMerge_BG1_int8_NUM<
          5>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 6:
      bnProcKernelMerge_BG1_int8_NUM<
          6>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 7:
      bnProcKernelMerge_BG1_int8_NUM<
          7>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 8:
      bnProcKernelMerge_BG1_int8_NUM<
          8>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 9:
      bnProcKernelMerge_BG1_int8_NUM<
          9>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 10:
      bnProcKernelMerge_BG1_int8_NUM<
          10>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 11:
      bnProcKernelMerge_BG1_int8_NUM<
          11>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 12:
      bnProcKernelMerge_BG1_int8_NUM<
          12>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 13:
      bnProcKernelMerge_BG1_int8_NUM<
          13>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 14:
      bnProcKernelMerge_BG1_int8_NUM<
          14>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 15:
      bnProcKernelMerge_BG1_int8_NUM<
          15>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 16:
      bnProcKernelMerge_BG1_int8_NUM<
          16>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 17:
      bnProcKernelMerge_BG1_int8_NUM<
          17>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 18:
      bnProcKernelMerge_BG1_int8_NUM<
          18>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 19:
      bnProcKernelMerge_BG1_int8_NUM<
          19>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 20:
      bnProcKernelMerge_BG1_int8_NUM<
          20>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 21:
      bnProcKernelMerge_BG1_int8_NUM<
          21>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 22:
      bnProcKernelMerge_BG1_int8_NUM<
          22>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 23:
      bnProcKernelMerge_BG1_int8_NUM<
          23>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 24:
      bnProcKernelMerge_BG1_int8_NUM<
          24>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 25:
      bnProcKernelMerge_BG1_int8_NUM<
          25>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 26:
      bnProcKernelMerge_BG1_int8_NUM<
          26>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 27:
      bnProcKernelMerge_BG1_int8_NUM<
          27>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 28:
      bnProcKernelMerge_BG1_int8_NUM<
          28>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 29:
      bnProcKernelMerge_BG1_int8_NUM<
          29>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    case 30:
      bnProcKernelMerge_BG1_int8_NUM<
          30>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc);
      break;
    default:
      break;
  }
}

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

// ======================= //
//  BN_PROC PC Kernel BG1  //
// ======================= //


__device__ __forceinline__ void bnProcKernel_BG1_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
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
  const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
  uint32_t prevIdxWords = (MsgIdx * GrpNum * Zc) >> 2;
  uint32_t prev = bnProcBufPtr[prevIdxWords];

  // ---- Unrolled accumulation ----
  uint32_t MsgSum = bnProcBufPtr[0];
  uint32_t off = (GrpNum * Zc) >> 2;
#pragma unroll
  for (int i = 1; i < GrpIdx; ++i) {
    bnProcBufPtr += off;
    MsgSum = __vaddss4(MsgSum, *bnProcBufPtr);
  }

  // ---- Compute llrRes ----

  int32_t computed_llrRes = __vaddss4(MsgSum, ((const int32_t *)(d_llrProcBuf))[lane]);

  uint32_t BricksToBeGet = __vsubss4(computed_llrRes, prev);
  // ---- Write result ----
  moveBricks_forput_circ(d_cnProcBuf, lane * 4, (uint8_t *)&BricksToBeGet, Zc, circShift);



}

__device__ __forceinline__ void bnProcKernel_BG1_int8_Gn_last(const int8_t *__restrict__ d_bnProcBuf,
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
  const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
  uint32_t prevIdxWords = (MsgIdx * GrpNum * Zc) >> 2;
  uint32_t prev = bnProcBufPtr[prevIdxWords];

  // ---- Unrolled accumulation ----
  uint32_t MsgSum = bnProcBufPtr[0];
  uint32_t off = (GrpNum * Zc) >> 2;
#pragma unroll
  for (int i = 1; i < GrpIdx; ++i) {
    bnProcBufPtr += off;
    MsgSum = __vaddss4(MsgSum, *bnProcBufPtr);
  }

  // ---- Compute llrRes ----

  int32_t computed_llrRes = __vaddss4(MsgSum, ((const int32_t *)(d_llrProcBuf))[lane]);
  //  Only write to llrRes when MsgIdx == 1
  if (MsgIdx == 0) {
    ((int32_t *)(d_llrRes))[lane] = computed_llrRes;
  }

}
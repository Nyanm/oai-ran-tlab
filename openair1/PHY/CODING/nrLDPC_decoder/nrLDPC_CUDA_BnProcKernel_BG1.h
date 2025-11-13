#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"


// ======================= //
//  BN_PROC PC Kernel BG1  //
// ======================= //


template<int NUM>
__device__ __forceinline__ void bnProcKernelMerge_BG1_int8_NUM(
    const int8_t *__restrict__ d_bnProcBuf,
    int8_t *__restrict__ d_cnProcBuf,
    const int8_t *__restrict__ d_llrProcBuf,
    int8_t *__restrict__ d_llrRes,
    uint32_t lane,
    uint8_t MsgIdx,
    uint32_t BnIdx,
    uint32_t GrpNum,
    uint32_t circShift,
    uint32_t Zc)
{
    

//    int8_t *p_bnProcBufRes_BnIdx     = d_bnProcBufRes + baseBn;
//    const int8_t *p_llrProcBuf_BnIdx = d_llrProcBuf + baseBn;
//    int8_t *p_llrRes_BnIdx           = d_llrRes + baseBn;

    const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
    uint32_t prevIdxWords = ( MsgIdx  * GrpNum * Zc) >> 2;
    uint32_t prev = bnProcBufPtr[prevIdxWords];

    // ---- ① Unrolled accumulation ----
    uint32_t MsgSum = bnProcBufPtr[0];
    uint32_t off = (GrpNum * Zc) >> 2;
#pragma unroll
    for (int i = 1; i < NUM; ++i) {
	bnProcBufPtr += off;
	MsgSum = __vaddss4(MsgSum,*bnProcBufPtr);
    }

    // ---- ② Compute llrRes ----
//    int32_t computed_llrRes = __vaddss4(MsgSum,((const int32_t*)p_llrProcBuf_BnIdx)[lane]);
    int32_t computed_llrRes = __vaddss4(MsgSum,((const int32_t*)(d_llrProcBuf))[lane]);
    //  Only write to llrRes when MsgIdx == 1 
    if (MsgIdx == 0) {
      ((int32_t *)(d_llrRes))[lane]  = computed_llrRes;
    }

    uint32_t BricksToBeGet = __vsubss4(computed_llrRes, prev);//0x01010101*NUM;
    // ---- ④ Write result ----
    moveBricks_forput_circ(d_cnProcBuf, lane * 4, (uint8_t*)&BricksToBeGet, Zc, circShift);
}   

__device__ __forceinline__ void bnProcKernelMerge_BG1_int8_Gn(
    const int8_t *__restrict__ d_bnProcBuf,
    int8_t *__restrict__ d_cnProcBuf,
    const int8_t *__restrict__ d_llrProcBuf,
    int8_t *__restrict__ d_llrRes,
    uint32_t lane,
    uint8_t GrpIdx,
    uint8_t MsgIdx,
    uint32_t BnIdx,
    uint8_t GrpNum,
    uint32_t circShift,
    uint32_t Zc)
{
    switch (GrpIdx)
    {
    case 1:  bnProcKernelMerge_BG1_int8_NUM<1 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 2:  bnProcKernelMerge_BG1_int8_NUM<2 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 3:  bnProcKernelMerge_BG1_int8_NUM<3 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 4:  bnProcKernelMerge_BG1_int8_NUM<4 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 5:  bnProcKernelMerge_BG1_int8_NUM<5 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 6:  bnProcKernelMerge_BG1_int8_NUM<6 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 7:  bnProcKernelMerge_BG1_int8_NUM<7 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 8:  bnProcKernelMerge_BG1_int8_NUM<8 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 9:  bnProcKernelMerge_BG1_int8_NUM<9 >(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 10: bnProcKernelMerge_BG1_int8_NUM<10>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 11: bnProcKernelMerge_BG1_int8_NUM<11>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 12: bnProcKernelMerge_BG1_int8_NUM<12>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 13: bnProcKernelMerge_BG1_int8_NUM<13>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 14: bnProcKernelMerge_BG1_int8_NUM<14>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 15: bnProcKernelMerge_BG1_int8_NUM<15>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 16: bnProcKernelMerge_BG1_int8_NUM<16>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 17: bnProcKernelMerge_BG1_int8_NUM<17>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 18: bnProcKernelMerge_BG1_int8_NUM<18>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 19: bnProcKernelMerge_BG1_int8_NUM<19>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 20: bnProcKernelMerge_BG1_int8_NUM<20>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 21: bnProcKernelMerge_BG1_int8_NUM<21>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 22: bnProcKernelMerge_BG1_int8_NUM<22>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 23: bnProcKernelMerge_BG1_int8_NUM<23>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 24: bnProcKernelMerge_BG1_int8_NUM<24>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 25: bnProcKernelMerge_BG1_int8_NUM<25>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 26: bnProcKernelMerge_BG1_int8_NUM<26>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 27: bnProcKernelMerge_BG1_int8_NUM<27>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 28: bnProcKernelMerge_BG1_int8_NUM<28>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 29: bnProcKernelMerge_BG1_int8_NUM<29>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    case 30: bnProcKernelMerge_BG1_int8_NUM<30>(d_bnProcBuf, d_cnProcBuf, d_llrProcBuf, d_llrRes, lane, MsgIdx, BnIdx, GrpNum, circShift, Zc); break;
    default: break;
    }
}


/*
__device__ void bnProcKernelMerge_BG1_int8_Gn(
    const int8_t *__restrict__ d_bnProcBuf,
    int8_t *__restrict__ d_bnProcBufRes,
    const int8_t *__restrict__ d_llrProcBuf,
    int8_t *__restrict__ d_llrRes,
    int lane,
    int GrpIdx,
    int MsgIdx,
    int BnIdx,
    int GrpNum,
    int Zc)
{
    // base pointers for this BN index
    const int baseBn = (BnIdx - 1) * Zc; // bytes offset
    const int laneByte = lane * 4;

    const int8_t *p_bnProcBuf_BnIdx = d_bnProcBuf + baseBn;
    int8_t *p_bnProcBufRes_BnIdx = d_bnProcBufRes + baseBn;
    const int8_t *p_llrProcBuf_BnIdx = d_llrProcBuf + baseBn;
    int8_t *p_llrRes_BnIdx = d_llrRes + baseBn;

    // pointer to int32 words for lane
    const int32_t *bnProcBufPtr = (const int32_t *)(p_bnProcBuf_BnIdx + laneByte);

    // Sum messages across groups (MsgSum)
    int32_t MsgSum = bnProcBufPtr[0];

    // Note: if NUM is small and known, replace the loop with unrolled/template version
    //#pragma unroll using this makes it even slower
    for (int i = 1; i < GrpIdx; ++i) {
        // compute word index: (GrpNum * i * Zc) / 4
        int idx = (GrpNum * i * Zc) >> 2; // use >>2 if guaranteed divisible by 4
        int32_t v = bnProcBufPtr[idx];
        MsgSum = __vaddss4(MsgSum, v);
    }

    // llrProcBuf value (per-lane)
    int32_t llrProcVal = *(const int32_t *)(p_llrProcBuf_BnIdx + laneByte);

    // computed llrRes for this lane (what PC would write)
    int32_t computed_llrRes = __vaddss4(MsgSum, llrProcVal);

    // If you'd like other threads in the same block to see this newly computed llrRes,
    // you must write it out and synchronize. If the read of llrRes for next steps
    // is local to this same thread, it's enough to use computed_llrRes directly.
    if (MsgIdx == 1) {
        // write computed value to global llrRes
        *(int32_t *)(p_llrRes_BnIdx + laneByte) = computed_llrRes;
        // If some other *other threads in same block* need to read this just-written llrRes
        // *before* they proceed, you must call __syncthreads() here.
        // But avoid __syncthreads() unless you are sure all these threads belong to the same block.
        // __syncthreads();
    }

    // For computing MsgRes we must use the authoritative llrRes:
    // - if MsgIdx == 1: we should use computed_llrRes (we just computed it)
    // - else: use the existing llrRes in memory
    int32_t llrForMsg;
    llrForMsg = computed_llrRes;

    // prev message
    int prevIdxWords = ((MsgIdx - 1) * GrpNum * Zc) >> 2; // /4 to get words
    int32_t prevMsg = *(const int32_t *)(p_bnProcBuf_BnIdx + prevIdxWords*4 + laneByte);

    int32_t MsgRes = __vsubss4(llrForMsg, prevMsg);

    // write out bnProcBufRes
    *(int32_t *)(p_bnProcBufRes_BnIdx + prevIdxWords*4 + laneByte) = MsgRes;
}
*/
/*
__device__ void bnProcKernelMerge_BG1_int8_Gn(const int8_t *__restrict__ d_bnProcBuf,
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


    int32_t *bnProcBufPtr = (int32_t *)(d_bnProcBuf_BnIdx + lane * 4);

    int32_t MsgSum = bnProcBufPtr[0];

    for (uint8_t i = 1; i < NUM; i++) {
      int32_t ymm0 = bnProcBufPtr[(GrpNum * i * Zc) / 4];
      MsgSum = __vaddss4(MsgSum, ymm0);
    }

    int32_t llrData = *(const int32_t *)(d_llrProcBuf_BnIdx + lane * 4);

    int32_t ymm0Res = __vaddss4(MsgSum, llrData);
  if (MsgIdx == 1) {
    *(int32_t *)(d_llrRes_BnIdx + lane * 4) = ymm0Res;
  }

  //__syncthreads();

  ymm0Res = llrData;

  int32_t prevMsg = *(const int32_t *)(d_bnProcBuf_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4);

  int32_t MsgRes = __vsubss4(ymm0Res, prevMsg);

  *(int32_t *)(d_bnProcBufRes_BnIdx + (MsgIdx - 1) * GrpNum * Zc + lane * 4) = MsgRes;

}*/

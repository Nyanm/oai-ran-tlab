#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

// ======================= //
//  BN_PROC PC Kernel BG1  //
// ======================= //

// 辅助函数：将1个 packed int32 (4x int8) 解包并符号扩展为 2个 packed int32 (2x int16)
// 输入: [d, c, b, a] (8-bit)
// 输出: val_lo=[b_16, a_16], val_hi=[d_16, c_16]
__device__ __forceinline__ void unpack_and_sign_extend(uint32_t packed, uint32_t* val_lo, uint32_t* val_hi) {
    // 利用 int8_t -> int16_t 的强制转换自动进行符号扩展 (Sign Extension)
    // 这一点至关重要！不能只是补0，否则负数会变正数
    int16_t a = (int16_t)(int8_t)(packed & 0xFF);
    int16_t b = (int16_t)(int8_t)((packed >> 8) & 0xFF);
    int16_t c = (int16_t)(int8_t)((packed >> 16) & 0xFF);
    int16_t d = (int16_t)(int8_t)((packed >> 24) & 0xFF);

    // 重新打包成 [High16 | Low16] 格式供 __vaddss2 使用
    *val_lo = (uint16_t)a | ((uint32_t)(uint16_t)b << 16);
    *val_hi = (uint16_t)c | ((uint32_t)(uint16_t)d << 16);
}

// 辅助函数：将 2个 packed int32 (2x int16) 饱和截断并打包回 1个 packed int32 (4x int8)
__device__ __forceinline__ uint32_t saturate_and_pack(uint32_t val_lo, uint32_t val_hi) {
    // 提取 16位 部分
    int16_t a = (int16_t)(val_lo & 0xFFFF);
    int16_t b = (int16_t)((val_lo >> 16) & 0xFFFF);
    int16_t c = (int16_t)(val_hi & 0xFFFF);
    int16_t d = (int16_t)((val_hi >> 16) & 0xFFFF);

    // 手动饱和逻辑 (-128 ~ 127)
    auto saturate_i8 = [](int16_t v) -> uint8_t {
        if (v > 127) return 127;
        if (v < -128) return 128; // 0x80
        return (uint8_t)v;
    };

    uint32_t res = 0;
    res |= saturate_i8(a);
    res |= (uint32_t)saturate_i8(b) << 8;
    res |= (uint32_t)saturate_i8(c) << 16;
    res |= (uint32_t)saturate_i8(d) << 24;
    return res;
}

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
    // 指针初始化
    const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
    
    // 1. 加载 Intrinsic LLR (原始信道值) 作为累加器初始值
    // 这个值对于 Degree=1 的节点将直接作为输出
    uint32_t packed_intrinsic = ((const int32_t *)(d_llrProcBuf))[lane];
    
    // 2. 解包 Intrinsic 到 16-bit 累加器 (MsgSumLo, MsgSumHi)
    // 即使是 Degree=1，我们也需要计算 llrRes (Posterior)，所以累加还是要做的
    uint32_t MsgSumLo, MsgSumHi;
    unpack_and_sign_extend(packed_intrinsic, &MsgSumLo, &MsgSumHi);

    // 3. 循环累加 (Unrolled Accumulation)
    uint32_t off = (GrpNum * Zc) >> 2;
    const int32_t *currPtr = bnProcBufPtr;

    #pragma unroll
    for (int i = 0; i < GrpIdx; ++i) {
        uint32_t val = *currPtr;
        uint32_t val_lo, val_hi;
        
        unpack_and_sign_extend(val, &val_lo, &val_hi);

        // 使用 16-bit 饱和加法计算 Total Sum (Posterior LLR)
        MsgSumLo = __vaddss2(MsgSumLo, val_lo);
        MsgSumHi = __vaddss2(MsgSumHi, val_hi);

        currPtr += off; 
    }

    // 4. 计算饱和后的 Total LLR (用于更新 llrRes)
    uint32_t saturated_llr = saturate_and_pack(MsgSumLo, MsgSumHi);

    // 5. 更新 llrRes (仅当处理第一个消息组时写入)
    // 这一步对于所有 Degree 的节点都是一样的：Posterior = Intrinsic + Sum(Extrinsic_In)
    //if (MsgIdx == 0) {
    //    ((int32_t *)(d_llrRes))[lane] = saturated_llr;
    //}

    // ==========================================================
    // 6. 计算回传给 CN 的 Extrinsic 消息 (BricksToBeGet)
    // ==========================================================
    uint32_t BricksToBeGet;

    if (GrpIdx == 1) {
        // 【Degree 1 特殊路径】
        // 对于只连接 1 个 CN 的节点，CPU 代码直接赋值：p_bnProcBufRes[i] = p_llrProcBuf[i];
        // 这种情况下，发回给 CN 的信息就是原始的 Intrinsic LLR (剔除 CN 自身信息后只剩 Intrinsic)
        // 这样避免了 (Intrinsic + CN_Msg - CN_Msg) 过程中因饱和而产生的不可逆误差
        BricksToBeGet = packed_intrinsic;
    } 
    else {
        // 【Degree > 1 标准路径】
        // 模拟 CPU 行为：使用已饱和的 LLR 减去 Prev
        
        // 加载 Prev (自身上一轮发出的消息)
        uint32_t prevIdxWords = (MsgIdx * GrpNum * Zc) >> 2;
        uint32_t prev = bnProcBufPtr[prevIdxWords]; 
        
        // 使用 8-bit 饱和减法 (__vsubss4)
        // 逻辑：Extrinsic = Saturate(Total) - Prev
        BricksToBeGet = __vsubss4(saturated_llr, prev);
    }

    // 7. 写入 CN Buffer
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
    // 指针初始化
    const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
    
    // 1. 加载 Intrinsic LLR (原始信道值) 作为累加器初始值
    // 这个值对于 Degree=1 的节点将直接作为输出
    uint32_t packed_intrinsic = ((const int32_t *)(d_llrProcBuf))[lane];
    
    // 2. 解包 Intrinsic 到 16-bit 累加器 (MsgSumLo, MsgSumHi)
    // 即使是 Degree=1，我们也需要计算 llrRes (Posterior)，所以累加还是要做的
    uint32_t MsgSumLo, MsgSumHi;
    unpack_and_sign_extend(packed_intrinsic, &MsgSumLo, &MsgSumHi);

    // 3. 循环累加 (Unrolled Accumulation)
    uint32_t off = (GrpNum * Zc) >> 2;
    const int32_t *currPtr = bnProcBufPtr;

    #pragma unroll
    for (int i = 0; i < GrpIdx; ++i) {
        uint32_t val = *currPtr;
        uint32_t val_lo, val_hi;
        
        unpack_and_sign_extend(val, &val_lo, &val_hi);

        // 使用 16-bit 饱和加法计算 Total Sum (Posterior LLR)
        MsgSumLo = __vaddss2(MsgSumLo, val_lo);
        MsgSumHi = __vaddss2(MsgSumHi, val_hi);

        currPtr += off; 
    }

    // 4. 计算饱和后的 Total LLR (用于更新 llrRes)
    uint32_t saturated_llr = saturate_and_pack(MsgSumLo, MsgSumHi);

    // 5. 更新 llrRes (仅当处理第一个消息组时写入)
    // 这一步对于所有 Degree 的节点都是一样的：Posterior = Intrinsic + Sum(Extrinsic_In)
    if (MsgIdx == 0) {
        ((int32_t *)(d_llrRes))[lane] = saturated_llr;
    }


}



/*
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
  uint32_t computed_llrRes = ((const int32_t *)(d_llrProcBuf))[lane];
  uint32_t off = (GrpNum * Zc) >> 2;
#pragma unroll
  for (int i = 0; i < GrpIdx; ++i) {
    bnProcBufPtr += i*off;
    computed_llrRes = __vaddss4(computed_llrRes, *bnProcBufPtr);
  }

  // ---- Compute llrRes ----

  //int32_t computed_llrRes = __vaddss4(MsgSum, );

  uint32_t BricksToBeGet = __vsubss4(computed_llrRes, prev);
  // ---- Write result ----
    if (MsgIdx == 0) {
    ((int32_t *)(d_llrRes))[lane] = computed_llrRes;
  }
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
 */
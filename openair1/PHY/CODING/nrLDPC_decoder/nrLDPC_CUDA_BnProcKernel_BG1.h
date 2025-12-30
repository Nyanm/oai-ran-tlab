/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */
  /*! \file nrLDPC_CUDA_BnProcKernel_BG1.h
 * \brief Defines the kernels for bit node processing
 * \author Qizhi Pan, Raymond Knopp
 * \company EURECOM
 * \email: qizhi.pan@eurecom.fr, raymond.knopp@eurecom.fr
 * \date 2025-12-30
 * \version 1.0
 * \note 
 * \warning
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"


__device__ __forceinline__ void unpack_and_sign_extend(uint32_t packed, uint32_t* val_lo, uint32_t* val_hi) {

    int16_t a = (int16_t)(int8_t)(packed & 0xFF);
    int16_t b = (int16_t)(int8_t)((packed >> 8) & 0xFF);
    int16_t c = (int16_t)(int8_t)((packed >> 16) & 0xFF);
    int16_t d = (int16_t)(int8_t)((packed >> 24) & 0xFF);

    *val_lo = (uint16_t)a | ((uint32_t)(uint16_t)b << 16);
    *val_hi = (uint16_t)c | ((uint32_t)(uint16_t)d << 16);
}

__device__ __forceinline__ uint32_t saturate_and_pack(uint32_t val_lo, uint32_t val_hi) {

    int16_t a = (int16_t)(val_lo & 0xFFFF);
    int16_t b = (int16_t)((val_lo >> 16) & 0xFFFF);
    int16_t c = (int16_t)(val_hi & 0xFFFF);
    int16_t d = (int16_t)((val_hi >> 16) & 0xFFFF);

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
    const int32_t *bnProcBufPtr = (const int32_t *)(d_bnProcBuf) + lane;
    
    uint32_t packed_intrinsic = ((const int32_t *)(d_llrProcBuf))[lane];
    
    uint32_t MsgSumLo, MsgSumHi;
    unpack_and_sign_extend(packed_intrinsic, &MsgSumLo, &MsgSumHi);

    uint32_t off = (GrpNum * NR_LDPC_ZMAX) >> 2;
    const int32_t *currPtr = bnProcBufPtr;

    #pragma unroll
    for (int i = 0; i < GrpIdx; ++i) {
        uint32_t val = *currPtr;
        uint32_t val_lo, val_hi;
        
        unpack_and_sign_extend(val, &val_lo, &val_hi);

        MsgSumLo = __vaddss2(MsgSumLo, val_lo);
        MsgSumHi = __vaddss2(MsgSumHi, val_hi);

        currPtr += off; 
    }

    uint32_t saturated_llr = saturate_and_pack(MsgSumLo, MsgSumHi);

    uint32_t BricksToBeGet;

    if (GrpIdx == 1) {
        BricksToBeGet = packed_intrinsic;
    } 
    else {
        
        uint32_t prevIdxWords = (MsgIdx * GrpNum * NR_LDPC_ZMAX) >> 2;
        uint32_t prev = bnProcBufPtr[prevIdxWords]; 
        
        BricksToBeGet = __vsubss4(saturated_llr, prev);
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
    
    uint32_t packed_intrinsic = ((const int32_t *)(d_llrProcBuf))[lane];
    
    uint32_t MsgSumLo, MsgSumHi;
    unpack_and_sign_extend(packed_intrinsic, &MsgSumLo, &MsgSumHi);

    uint32_t off = (GrpNum * NR_LDPC_ZMAX) >> 2;
    const int32_t *currPtr = bnProcBufPtr;

    #pragma unroll
    for (int i = 0; i < GrpIdx; ++i) {
        uint32_t val = *currPtr;
        uint32_t val_lo, val_hi;
        
        unpack_and_sign_extend(val, &val_lo, &val_hi);

        MsgSumLo = __vaddss2(MsgSumLo, val_lo);
        MsgSumHi = __vaddss2(MsgSumHi, val_hi);

        currPtr += off; 
    }

    uint32_t saturated_llr = saturate_and_pack(MsgSumLo, MsgSumHi);

    if (MsgIdx == 0) {
        ((int32_t *)(d_llrRes))[lane] = saturated_llr;
    }


}

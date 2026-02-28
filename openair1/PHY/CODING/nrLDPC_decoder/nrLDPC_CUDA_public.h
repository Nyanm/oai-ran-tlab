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
/*! \file nrLDPC_CUDA_public.h
 * \brief Shared functions in CUDA implementation of LDPC decoder 
 * \author Qizhi Pan, Raymond Knopp
 * \company EURECOM
 * \email: qizhi.pan@eurecom.fr, raymond.knopp@eurecom.fr
 * \date 2025-12-30
 * \version 1.0
 * \note 
 * \warning
 */

#pragma once

#include "PHY/gpu_compat.h"
#include "PHY/gpu_simd_intrin_compat.h"
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

#define arrPos(a, b) a.d + b *a.dim2

#include <stdint.h>


__device__ __forceinline__ void moveBricks_invput_circ(int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       const uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
    uint32_t tmp = cshift + dstBuf_Offset;
    uint32_t pos = (tmp >= Z) ? tmp - Z : tmp;

    if (pos <= Z - 4) {
        uint32_t val = *(const uint32_t*)Four_Bricks;
        memcpy(dstBuf + pos, &val, 4);
    } 
    else {
        uint32_t bytes_at_end = Z - pos; // 1, 2, or 3
        
        #pragma unroll
        for(int i=0; i<4; i++) {
            if (i < bytes_at_end) {
                dstBuf[pos + i] = Four_Bricks[i];
            } else {
                dstBuf[i - bytes_at_end] = Four_Bricks[i];
            }
        }
    }
}

__device__ __forceinline__ void moveBricks_forput_circ(int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       const uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
    uint32_t tmp = dstBuf_Offset + Z - cshift;
    uint32_t pos = (tmp >= Z) ? tmp - Z : tmp;

    if (pos <= Z - 4) {
        uint32_t val = *(const uint32_t*)Four_Bricks;
        memcpy(dstBuf + pos, &val, 4);
    } 
    else {
        uint32_t bytes_at_end = Z - pos;
        
        #pragma unroll
        for(int i=0; i<4; i++) {
            if (i < bytes_at_end) {
                dstBuf[pos + i] = Four_Bricks[i];
            } else {
                dstBuf[i - bytes_at_end] = Four_Bricks[i];
            }
        }
    }
}

__device__ __forceinline__ void moveBricks_invget_circ(const int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
    uint32_t tmp = cshift + dstBuf_Offset;
    uint32_t pos = (tmp >= Z) ? tmp - Z : tmp;

    if (pos <= Z - 4) {
        uint32_t val;
        memcpy(&val, dstBuf + pos, 4);
        *(uint32_t*)Four_Bricks = val;
    } 
    else {
        uint32_t bytes_at_end = Z - pos;
        
        #pragma unroll
        for(int i=0; i<4; i++) {
            if (i < bytes_at_end) {
                Four_Bricks[i] = dstBuf[pos + i];
            } else {
                Four_Bricks[i] = dstBuf[i - bytes_at_end];
            }
        }
    }
}

__device__ __forceinline__ uint32_t __vxor4(const uint32_t a, uint32_t b)
{
  return a ^ b; 
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t a, uint32_t b)
{
    uint32_t mask = gpu_vcmplts4(b, 0); 
    uint32_t bneg = gpu_vneg4(a); 
    return (mask & bneg) | (~mask & a);
    //uint32_t is_zero_mask = __vcmpeq4(b, 0);
    //return result & (~is_zero_mask);
}


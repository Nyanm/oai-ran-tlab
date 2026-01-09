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

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"

#define arrPos(a, b) a.d + b *a.dim2

enum CircShiftDirection { FORWARD = 0, INVERSE = 1 };
enum CircShiftOp { PUT_BRICKS = 0, GET_BRICKS = 1 };

__device__ __forceinline__ void moveBricks_invput_circ(int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
  uint32_t pos = (cshift + dstBuf_Offset) % Z;

  switch (pos + 3 - Z) {
    case 0:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[pos + 2] = Four_Bricks[2];
      dstBuf[0] = Four_Bricks[3];
      break;
    case 1:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[0] = Four_Bricks[2];
      dstBuf[1] = Four_Bricks[3];
      break;
    case 2:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[0] = Four_Bricks[1];
      dstBuf[1] = Four_Bricks[2];
      dstBuf[2] = Four_Bricks[3];
      break;
    default:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[pos + 2] = Four_Bricks[2];
      dstBuf[pos + 3] = Four_Bricks[3];
      break;
  }
}
__device__ __forceinline__ void moveBricks_forput_circ(int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       const uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
  uint32_t pos = (dstBuf_Offset + Z - cshift ) % Z;

  switch (pos + 3 - Z) {
    case 0:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[pos + 2] = Four_Bricks[2];
      dstBuf[0] = Four_Bricks[3];
      break;
    case 1:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[0] = Four_Bricks[2];
      dstBuf[1] = Four_Bricks[3];
      break;
    case 2:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[0] = Four_Bricks[1];
      dstBuf[1] = Four_Bricks[2];
      dstBuf[2] = Four_Bricks[3];
      break;
    default:
      dstBuf[pos] = Four_Bricks[0];
      dstBuf[pos + 1] = Four_Bricks[1];
      dstBuf[pos + 2] = Four_Bricks[2];
      dstBuf[pos + 3] = Four_Bricks[3];
      break;
  }
}
__device__ __forceinline__ void moveBricks_invget_circ(int8_t *__restrict__ dstBuf,
                                                       uint32_t dstBuf_Offset,
                                                       uint8_t *__restrict__ Four_Bricks,
                                                       uint32_t Z,
                                                       uint32_t cshift)
{
  uint32_t pos = (cshift + dstBuf_Offset) % Z;

  switch (pos + 3 - Z) {
    case 0:
      Four_Bricks[0] = dstBuf[pos];
      Four_Bricks[1] = dstBuf[pos + 1];
      Four_Bricks[2] = dstBuf[pos + 2];
      Four_Bricks[3] = dstBuf[0];
      break;
    case 1:
      Four_Bricks[0] = dstBuf[pos];
      Four_Bricks[1] = dstBuf[pos + 1];
      Four_Bricks[2] = dstBuf[0];
      Four_Bricks[3] = dstBuf[1];
      break;
    case 2:
      Four_Bricks[0] = dstBuf[pos];
      Four_Bricks[1] = dstBuf[0];
      Four_Bricks[2] = dstBuf[1];
      Four_Bricks[3] = dstBuf[2];
      break;
    default:
      Four_Bricks[0] = dstBuf[pos];
      Four_Bricks[1] = dstBuf[pos + 1];
      Four_Bricks[2] = dstBuf[pos + 2];
      Four_Bricks[3] = dstBuf[pos + 3];
      break;
  }
}

__device__ __forceinline__ uint32_t __vxor4(const uint32_t a, uint32_t b)
{
  return a ^ b; 
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t a, uint32_t b)
{
    uint32_t mask = __vcmplts4(b, 0); 
    uint32_t bneg = __vneg4(a); 
    uint32_t result = (mask & bneg) | (~mask & a);
    uint32_t is_zero_mask = __vcmpeq4(b, 0);
    return result & (~is_zero_mask);
}


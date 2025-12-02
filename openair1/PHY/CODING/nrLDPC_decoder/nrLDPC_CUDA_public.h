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


__device__ __forceinline__ uint32_t __vxor4(const uint32_t *a, uint32_t *b)
{
  return a[0] ^ b[0]; // increase accuracy
}
/*
__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{
  uint32_t mask = __vcmples4(b[0] | 0x01010101, 0); // 0xFF / 0x00 per‑byte
  uint32_t bneg = __vneg4(a[0]);
  return (mask & bneg) | (~mask & a[0]); // Compute ±magnitude in two steps
}
*/
__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{
    uint32_t mask = __vcmplts4(b[0], 0); 
    uint32_t bneg = __vneg4(a[0]); 
    uint32_t result = (mask & bneg) | (~mask & a[0]);
    uint32_t is_zero_mask = __vcmpeq4(b[0], 0);
    return result & (~is_zero_mask);
}
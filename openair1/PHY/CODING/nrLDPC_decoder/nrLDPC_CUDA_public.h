#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"


#define arrPos(a, b) a.d + b *a.dim2

enum CircShiftDirection { FORWARD = 0, INVERSE = 1 };
enum CircShiftOp { PUT_BRICKS = 0, GET_BRICKS = 1 };

__device__ __forceinline__ void moveBricks_invput_circ(int8_t *__restrict__ dstBuf,
                                uint16_t dstBuf_Offset,
                                uint8_t *__restrict__ Four_Bricks,
                                uint16_t Z,
                                uint16_t cshift)
{
  int8_t *DstBuf = (int8_t *)dstBuf;
  uint16_t shift;

  shift = (cshift + dstBuf_Offset) % Z;

  uint16_t pos = shift;
  uintptr_t ptr = (uintptr_t)(DstBuf + pos);

  // put bricks
  if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
    *(uint32_t *)(DstBuf + pos) = *(const uint32_t *)(Four_Bricks);
  } else {
    switch (pos+3-Z) {
	    case 0 :
		    DstBuf[pos]   = Four_Bricks[0];
		    DstBuf[pos+1] = Four_Bricks[1];
		    DstBuf[pos+2] = Four_Bricks[2];
		    DstBuf[0]     = Four_Bricks[3];
		    break;
	    case 1 :
		    DstBuf[pos]   = Four_Bricks[0];
		    DstBuf[pos+1] = Four_Bricks[1];
		    DstBuf[0]     = Four_Bricks[2];
		    DstBuf[1]     = Four_Bricks[3];
		    break;
	    case 2 :
		    DstBuf[pos] = Four_Bricks[0];
		    DstBuf[0]   = Four_Bricks[1];
		    DstBuf[1]   = Four_Bricks[2];
		    DstBuf[2]   = Four_Bricks[3];
		    break;
            default:
		    DstBuf[pos]   = Four_Bricks[0];
		    DstBuf[pos+1] = Four_Bricks[1];
		    DstBuf[pos+2] = Four_Bricks[2];
		    DstBuf[pos+3] = Four_Bricks[3];
		    break;
    }
/*
    for (uint16_t j = 0; j < 4; j++) {
      DstBuf[(pos + j) % Z] = Four_Bricks[j];
    }*/
  }
}
__device__ __forceinline__ void moveBricks_invget_circ(int8_t *__restrict__ dstBuf,
                                uint16_t dstBuf_Offset,
                                uint8_t *__restrict__ Four_Bricks,
                                uint16_t Z,
                                uint16_t cshift)
{
  int8_t *DstBuf = (int8_t *)dstBuf;
  uint16_t shift;

  shift = (cshift + dstBuf_Offset) % Z;

  uint16_t pos = shift;
  uintptr_t ptr = (uintptr_t)(DstBuf + pos);

  // get bricks
  if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
    *(uint32_t *)(Four_Bricks) = *(const uint32_t *)(DstBuf + pos);
  } else {
    switch(pos+3-Z) {
	    case 0 :
		    Four_Bricks[0] = DstBuf[pos];
		    Four_Bricks[1] = DstBuf[pos+1];
		    Four_Bricks[2] = DstBuf[pos+2];
		    Four_Bricks[3] = DstBuf[0];
		    break;
	    case 1 :
		    Four_Bricks[0] = DstBuf[pos];
		    Four_Bricks[1] = DstBuf[pos+1];
		    Four_Bricks[2] = DstBuf[0];
		    Four_Bricks[3] = DstBuf[1];
		    break;
	    case 2 :
		    Four_Bricks[0] = DstBuf[pos];
		    Four_Bricks[1] = DstBuf[0];
		    Four_Bricks[2] = DstBuf[1];
		    Four_Bricks[3] = DstBuf[2];
		    break;
	    default :
		    Four_Bricks[0] = DstBuf[pos];
		    Four_Bricks[1] = DstBuf[pos+1];
		    Four_Bricks[2] = DstBuf[pos+2];
		    Four_Bricks[3] = DstBuf[pos+3];
		    break;
    }
/*	 
    for (uint16_t j = 0; j < 4; j++) {
      Four_Bricks[j] = DstBuf[(pos + j) % Z];
    }
  */   
  }
}
__device__ __forceinline__ void moveBricks_circ(int8_t *__restrict__ dstBuf,
                                uint16_t dstBuf_Offset,
                                uint8_t *__restrict__ Four_Bricks,
                                uint16_t Z,
                                uint16_t cshift,
                                CircShiftDirection dir,
                                CircShiftOp op)
{
  int8_t *DstBuf = (int8_t *)dstBuf;
  uint16_t shift;

  if (dir == FORWARD) {
    shift = (Z - ((cshift + dstBuf_Offset) % Z)) % Z;
  } else {
    shift = (cshift + dstBuf_Offset) % Z;
  }

  uint16_t pos = shift;
  uintptr_t ptr = (uintptr_t)(DstBuf + pos);

  if (op == PUT_BRICKS) {
    // put bricks
    if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
      *(uint32_t *)(DstBuf + pos) = *(const uint32_t *)(Four_Bricks);
    } else {
      for (uint16_t j = 0; j < 4; j++) {
        DstBuf[(pos + j) % Z] = Four_Bricks[j];
      }
    }
  } else if (op == GET_BRICKS) {
    // get bricks
    if ((pos + 3 < Z) && ((ptr & 0x3) == 0)) {
      *(uint32_t *)(Four_Bricks) = *(const uint32_t *)(DstBuf + pos);
    } else {
      for (uint16_t j = 0; j < 4; j++) {
        Four_Bricks[j] = DstBuf[(pos + j) % Z];
      }
    }
  }
}

__device__ __forceinline__ uint32_t __vxor4(const uint32_t *a, uint32_t *b)
{
  return a[0] ^ b[0]; // increase accuracy
}

__device__ __forceinline__ uint32_t __vsign4(const uint32_t *a, uint32_t *b)
{
  uint32_t mask = __vcmples4(b[0] | 0x01010101, 0); // 0xFF / 0x00 per‑byte
  uint32_t bneg = __vneg4(a[0]);
  return (mask & bneg) | (~mask & a[0]); // Compute ±magnitude in two steps
}

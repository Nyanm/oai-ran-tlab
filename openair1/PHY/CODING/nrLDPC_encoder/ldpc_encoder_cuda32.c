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

/*! \file ldpc_encoder_optim8segmulti.c
 * \brief Defines the optimized LDPC encoder
 * \author Florian Kaltenberger, Raymond Knopp, Kien le Trung (Eurecom)
 * \email openair_tech@eurecom.fr
 * \date 27-03-2018
 * \version 1.0
 * \note
 * \warning
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "assertions.h"
#include "common/utils/LOG/log.h"
#include "time_meas.h"
#include "openair1/PHY/CODING/nrLDPC_defs.h"
#include "PHY/sse_intrin.h"
#include "openair1/PHY/CODING/nrLDPC_extern.h"

#include "ldpc_encode_parity_check_cuda.c"
#include "ldpc_generate_coefficient.c"

#include <cuda_runtime.h>

#ifdef __AVX2__
simde__m256i input32_lut[32][256];
#else
#if 0
simde__m128i input32_luta[32][256];
simde__m128i input32_lutb[32][256];
#endif
#endif

uint32_t *cc0[4];
uint32_t *dd0[4];

int managed = 0, concurrent = 0, uva = 0, pageable = 0, pageable_uses_host = 0;

void cuda_support_init() {


    int dev = 0;
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);


    cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev);
    cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev);
    cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, dev);
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
    cudaDeviceGetAttribute(&pageable_uses_host, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);

    LOG_I(NR_PHY,"Device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    LOG_I(NR_PHY,"Unified Virtual Addressing (UVA): %s\n", uva ? "YES" : "NO");
    LOG_I(NR_PHY,"Managed (Unified) Memory:        %s\n", managed ? "YES" : "NO");
    LOG_I(NR_PHY,"Concurrent managed access:       %s\n", concurrent ? "YES" : "NO");
    LOG_I(NR_PHY,"Pageable memory access:          %s\n", pageable ? "YES" : "NO");
    LOG_I(NR_PHY,"Uses host page tables:           %s\n", pageable_uses_host ? "YES" : "NO");

  // initialize input and output memory
  if (!managed) {
    for (int i=0;i<4;i++) {
      cudaError_t err=cudaMalloc((void**)&cc0[i],22*384*sizeof(uint32_t));
      AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
    }
  }
  for (int i=0;i<4;i++) {
      cudaError_t err=cudaMalloc((void**)&dd0[i],46*384*sizeof(uint32_t));
      AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
  }
#if 0
  for (int i=0;i<256;i++) {
    input32_luta[0][i] = simde_mm_insert_epi32(input32_luta[0][i],(i&128)>>7,0);
    input32_luta[0][i] = simde_mm_insert_epi32(input32_luta[0][i],(i&64)>>6,1);
    input32_luta[0][i] = simde_mm_insert_epi32(input32_luta[0][i],(i&32)>>5,2);
    input32_luta[0][i] = simde_mm_insert_epi32(input32_luta[0][i],(i&16)>>4,3);
    input32_lutb[0][i] = simde_mm_insert_epi32(input32_lutb[0][i],(i&8)>>3,0);
    input32_lutb[0][i] = simde_mm_insert_epi32(input32_lutb[0][i],(i&4)>>2,1);
    input32_lutb[0][i] = simde_mm_insert_epi32(input32_lutb[0][i],(i&2)>>1,2);
    input32_lutb[0][i] = simde_mm_insert_epi32(input32_lutb[0][i],(i&1),3);

    for (int j=1;j<32;j++) {
      input32_luta[j][i]=simde_mm_slli_epi32(input32_luta[0][i],j);
      input32_lutb[j][i]=simde_mm_slli_epi32(input32_lutb[0][i],j);
    }
  }
#endif
}

int LDPCencoder32(uint8_t **input, uint32_t output[4][68*384], encoder_implemparams_t *impp)
{
  //set_log(PHY, 4);

  int Zc = impp->Zc;
  int Kb = impp->Kb;
  short block_length = impp->K;
  short BG = impp->BG;
  int nrows=0,ncols=0;
  int rate=3;
  int no_punctured_columns,removed_bit;

  if(impp->tinput != NULL) start_meas(impp->tinput);
  //determine number of bits in codeword
  if (BG==1)
    {
      nrows=46; //parity check bits
      ncols=22; //info bits
      rate=3;
    }
    else if (BG==2)
    {
      nrows=42; //parity check bits
      ncols=10; // info bits
      rate=5;
    }

#ifdef DEBUG_LDPC
  LOG_I(PHY,"ldpc_encoder_cuda32: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",BG,Zc,Kb,block_length,impp->n_segments);
  LOG_I(PHY,"ldpc_encoder_cuda32: PDU (seg 0) %x %x %x %x\n",input[0][0],input[0][1],input[0][2],input[0][3]);
#endif

  AssertFatal(Zc > 0, "no valid Zc found for block length %d\n", block_length);

  int n_inputs = (impp->n_segments/32)+(((impp->n_segments&31) > 0) ? 1: 0);
  uint32_t  cc[4][22*Zc]; //padded input, unpacked, max size

  // calculate number of punctured bits
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*rate)/Zc;
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length*rate);
  // clear input
  for (int i=0;i<n_inputs;i++) {
    memset(cc[i],0,22*Zc*sizeof(uint32_t));
  }

  //interleave up to 32 transport-block segements at a time

#if 0
  // unoptimized version of input processing
  for (; i_dword < block_length; i_dword++) {
    unsigned int i = i_dword;
    for (int j = 0; j < impp->n_segments; j++) {

      temp = (input[j][i/8]&(128>>(i&7)))>>(7-(i&7));
      cc[j>>5][i] |= (temp << (j&31));
    }
  }
#else
#ifdef __AVX2__

#else
#if 0
  simde__m128i temp128a,temp128b,*ccj;
  int i2=0; 
  uint8_t* inp;
  simde__m128i *luta,*lutb;
#if 1
  for (int j = 0; j < impp->n_segments; j++) {
    inp = input[j];
    luta=input32_luta[j&31];
    lutb=input32_lutb[j&31];    
    ccj=(simde__m128i*)(&cc[j>>5][0]);
    i2=0;
    for (int i=0; i < (block_length>>3); i++,i2+=2) {
       temp128a = luta[inp[i]];
       temp128b = lutb[inp[i]];
       ccj[i2]   = simde_mm_or_si128(ccj[i2],temp128a);
       ccj[i2+1] = simde_mm_or_si128(ccj[i2+1],temp128b);
    }
  }
#else
  i2=0;
  for (int i=0; i < (block_length>>3); i++,i2+=2) {
    luta=input32_luta[0];
    lutb=input32_lutb[0];    
    ccj=(simde__m128i*)(&cc[0][0]);
    simde__m128i cc128a,cc128b;
    temp128a = luta[input[0][i]];
    temp128b = lutb[input[0][i]];
    cc128a   = temp128a;
    cc128b   = temp128b;
    for (int j =1; j < impp->n_segments; j++) {
       luta=input32_luta[j];
       lutb=input32_lutb[j];    
       temp128a = luta[input[j][i]];
       temp128b = lutb[input[j][i]];
       cc128a   = simde_mm_or_si128(cc128a,temp128a);
       cc128b   = simde_mm_or_si128(cc128b,temp128b);
    }
    ccj[i2] = cc128a;
    ccj[i2+1] = cc128b;
  }

#endif
#else
  int i2=0;
  const int32_t ucShifta[32][4] = { 
	  {-7,-6,-5,-4}, // 0  
	  {-6,-5,-4,-3}, // 1
	  {-5,-4,-3,-2}, // 2 
	  {-4,-3,-2,-1}, // 3 
	  {-3,-2,-1,0}, // 4 
	  {-2,-1,0,1}, // 5
	  {-1,0,1,2}, // 6
	  {0,1,2,3}, // 7
	  {1,2,3,4}, // 8
	  {2,3,4,5}, // 9
	  {3,4,5,6}, // 10
	  {4,5,6,7}, // 11
	  {5,6,7,8}, // 12
	  {6,7,8,9}, // 13
	  {7,8,9,10}, // 14
	  {8,9,10,11}, // 15
	  {9,10,11,12}, // 16
	  {10,11,12,13}, // 17
	  {11,12,13,14}, // 18
	  {12,13,14,15}, // 19
	  {13,14,15,16}, // 20
	  {14,15,16,17}, // 21
	  {15,16,17,18}, // 22
	  {16,17,18,19}, // 23
	  {17,18,19,20}, // 24
	  {18,19,20,21}, // 25
	  {19,20,21,22}, // 26
	  {20,21,22,23}, // 27
	  {21,22,23,24}, // 28
	  {22,23,24,25}, // 29
	  {23,24,25,26}, // 30
	  {24,25,26,27}}; // 31
  const int32_t ucShiftb[32][4] = { 
	  {-3,-2,-1,0}, // 0 
	  {-2,-1,0,1}, // 1
	  {-1,0,1,2}, // 2
	  {0,1,2,3}, // 3
	  {1,2,3,4}, // 4
	  {2,3,4,5}, // 5
	  {3,4,5,6}, // 6 
	  {4,5,6,7}, // 7
	  {5,6,7,8}, // 8
	  {6,7,8,9}, // 9
	  {7,8,9,10}, // 10
	  {8,9,10,11}, // 11
	  {9,10,11,12}, // 12
	  {10,11,12,13}, // 13
	  {11,12,13,14}, // 14
	  {12,13,14,15}, // 15
	  {13,14,15,16}, // 16
	  {14,15,16,17}, // 17
	  {15,16,17,18}, // 18
	  {16,17,18,19}, // 19
	  {17,18,19,20}, // 20
	  {18,19,20,21}, // 21
	  {19,20,21,22}, // 22
	  {20,21,22,23}, // 23
	  {21,22,23,24}, // 24
	  {22,23,24,25}, // 25
	  {23,24,25,26}, // 26
	  {24,25,26,27}, // 27
	  {25,26,27,28}, // 28
	  {26,27,28,29}, // 29
	  {27,28,29,30}, // 30
	  {28,29,30,31}}; // 31
			      
  const int32_t ucShiftc[32][4] = { 
	  {-15,-14,-13,-12}, // 0
	  {-14,-13,-12,-11}, // 1
	  {-13,-12,-11,-10}, // 2
          {-12,-11,-10,-9}, // 3
	  {-11,-10,-9,-8}, // 4
	  {-10,-9,-8,-7}, // 5
	  {-9,-8,-7,-6}, // 6
	  {-8,-7,-6,-5}, // 7
	  {-7,-6,-5,-4}, // 8  
	  {-6,-5,-4,-3}, // 9
	  {-5,-4,-3,-2}, // 10 
	  {-4,-3,-2,-1}, // 11 
	  {-3,-2,-1,0}, // 12 
	  {-2,-1,0,1}, // 13
	  {-1,0,1,2}, // 14
	  {0,1,2,3}, // 15
	  {1,2,3,4}, // 16
	  {2,3,4,5}, // 17
	  {3,4,5,6}, // 18
	  {4,5,6,7}, // 19
	  {5,6,7,8}, // 20
	  {6,7,8,9}, // 21
	  {7,8,9,10}, // 22
	  {8,9,10,11}, // 23
	  {9,10,11,12}, // 24
	  {10,11,12,13}, // 25
	  {11,12,13,14}, // 26
	  {12,13,14,15}, // 27
	  {13,14,15,16}, // 28
	  {14,15,16,17}, // 29
	  {15,16,17,18}, // 30
	  {16,17,18,19}}; // 31
  const int32_t ucShiftd[32][4] = { 
	  {-11,-10,-9,-8}, // 0
	  {-10,-9,-8,-7}, // 1
	  {-9,-8,-7,-6}, // 2
	  {-8,-7,-6,-5}, // 3
	  {-7,-6,-5,-4}, // 4  
	  {-6,-5,-4,-3}, // 5
	  {-5,-4,-3,-2}, // 6
	  {-4,-3,-2,-1}, // 7 
	  {-3,-2,-1,0}, // 8
	  {-2,-1,0,1}, // 9
	  {-1,0,1,2}, // 10
	  {0,1,2,3}, // 11 
	  {1,2,3,4}, // 12
	  {2,3,4,5}, // 13
	  {3,4,5,6}, // 14 
	  {4,5,6,7}, // 15
	  {5,6,7,8}, // 16
	  {6,7,8,9}, // 17
	  {7,8,9,10}, // 18
	  {8,9,10,11}, // 19
	  {9,10,11,12}, // 20
	  {10,11,12,13}, // 21
	  {11,12,13,14}, // 22
	  {12,13,14,15}, // 23
	  {13,14,15,16}, // 24
	  {14,15,16,17}, // 25
	  {15,16,17,18}, // 26
	  {16,17,18,19}, // 27
	  {17,18,19,20}, // 28
	  {18,19,20,21}, // 29
	  {19,20,21,22}, // 30
	  {20,21,22,23}};// 31
			  
  const int32_t ucShifte[32][4] = { 
	  {-23,-22,-21,-20}, // 0 
	  {-22,-21,-20,-19}, // 1
	  {-21,-20,-19,-18}, // 2
	  {-20,-19,-18,-17}, // 3
	  {-19,-18,-17,-16}, // 4 
	  {-18,-17,-16,-15}, // 5
	  {-17,-16,-15,-14}, // 6
	  {-16,-15,-14,-13}, // 7
	  {-15,-14,-13,-12}, // 8
	  {-14,-13,-12,-11}, // 9
	  {-13,-12,-11,-10}, // 10
          {-12,-11,-10,-9}, // 11
	  {-11,-10,-9,-8}, // 12
	  {-10,-9,-8,-7}, // 13
	  {-9,-8,-7,-6}, // 14 
	  {-8,-7,-6,-5}, // 15
	  {-7,-6,-5,-4}, // 16 
	  {-6,-5,-4,-3}, // 17
	  {-5,-4,-3,-2}, // 18 
	  {-4,-3,-2,-1}, // 19 
	  {-3,-2,-1,0}, // 20 
	  {-2,-1,0,1}, // 21
	  {-1,0,1,2}, // 22
	  {0,1,2,3}, // 23
	  {1,2,3,4}, // 24
	  {2,3,4,5}, // 25
	  {3,4,5,6}, // 26
	  {4,5,6,7}, // 27
	  {5,6,7,8}, // 28
	  {6,7,8,9}, // 29
	  {7,8,9,10}, // 30
	  {8,9,10,11}}; // 31 
			  
  const int32_t ucShiftf[32][4] = { 
	  {-19,-18,-17,-16}, // 0 
	  {-18,-17,-16,-15}, // 1
	  {-17,-16,-15,-14}, // 2
	  {-16,-15,-14,-13}, // 3
	  {-15,-14,-13,-12}, // 4
	  {-14,-13,-12,-11}, // 5
	  {-13,-12,-11,-10}, // 6
          {-12,-11,-10,-9}, // 7
	  {-11,-10,-9,-8}, // 8
	  {-10,-9,-8,-7}, // 9  
	  {-9,-8,-7,-6}, // 10 
	  {-8,-7,-6,-5}, // 11
	  {-7,-6,-5,-4}, // 12 
	  {-6,-5,-4,-3}, // 13
	  {-5,-4,-3,-2}, // 14
	  {-4,-3,-2,-1}, // 15 
	  {-3,-2,-1,0}, // 16
	  {-2,-1,0,1}, // 17
	  {-1,0,1,2}, // 18
	  {0,1,2,3}, // 19
	  {1,2,3,4}, // 20
	  {2,3,4,5}, // 21
	  {3,4,5,6}, // 22 
	  {4,5,6,7}, // 23
	  {5,6,7,8}, // 24
	  {6,7,8,9}, // 25
	  {7,8,9,10}, // 26
	  {8,9,10,11}, // 27
	  {9,10,11,12}, // 28
	  {10,11,12,13}, // 29
	  {11,12,13,14}, // 30
	  {12,13,14,15}}; // 31
			  
  const int32_t ucShiftg[32][4] = { 
	  {-31,-30,-29,-28},
	  {-30,-29,-28,-27},
	  {-29,-28,-27,-26},
	  {-28,-27,-26,-25},
	  {-27,-26,-25,-24},
	  {-26,-25,-24,-23},
	  {-25,-24,-23,-22},
	  {-24,-23,-22,-21},
	  {-23,-22,-21,-20}, // 0 
	  {-22,-21,-20,-19}, // 1
	  {-21,-20,-19,-18}, // 2
	  {-20,-19,-18,-17}, // 3
	  {-19,-18,-17,-16}, // 4 
	  {-18,-17,-16,-15}, // 5
	  {-17,-16,-15,-14}, // 6
	  {-16,-15,-14,-13}, // 7
	  {-15,-14,-13,-12}, // 8
	  {-14,-13,-12,-11}, // 9
	  {-13,-12,-11,-10}, // 10
          {-12,-11,-10,-9}, // 11
	  {-11,-10,-9,-8}, // 12
	  {-10,-9,-8,-7}, // 13
	  {-9,-8,-7,-6}, // 14 
	  {-8,-7,-6,-5}, // 15
	  {-7,-6,-5,-4}, // 16 
	  {-6,-5,-4,-3}, // 17
	  {-5,-4,-3,-2}, // 18 
	  {-4,-3,-2,-1}, // 19 
	  {-3,-2,-1,0}, // 20 
	  {-2,-1,0,1}, // 21
	  {-1,0,1,2}, // 22
	  {0,1,2,3}}; // 23
			  
  const int32_t ucShifth[32][4] = { 
	  {-27,-26,-25,-24}, // 0
	  {-26,-25,-24,-23}, // 1
	  {-25,-24,-23,-22}, // 2
	  {-24,-23,-22,-21}, // 3
	  {-23,-22,-21,-20}, // 4 
	  {-22,-21,-20,-19}, // 5
	  {-21,-20,-19,-18}, // 6
	  {-20,-19,-18,-17}, // 7
	  {-19,-18,-17,-16}, // 8 
	  {-18,-17,-16,-15}, // 9
	  {-17,-16,-15,-14}, // 10
	  {-16,-15,-14,-13}, // 11
	  {-15,-14,-13,-12}, // 12
	  {-14,-13,-12,-11}, // 13
	  {-13,-12,-11,-10}, // 14
          {-12,-11,-10,-9}, // 15
	  {-11,-10,-9,-8}, // 16
	  {-10,-9,-8,-7}, // 17 
	  {-9,-8,-7,-6}, // 18 
	  {-8,-7,-6,-5}, // 19
	  {-7,-6,-5,-4}, // 20 
	  {-6,-5,-4,-3}, // 21
	  {-5,-4,-3,-2}, // 22
	  {-4,-3,-2,-1}, // 23 
	  {-3,-2,-1,0}, // 24 
	  {-2,-1,0,1}, // 25
	  {-1,0,1,2}, // 26
	  {0,1,2,3}, // 27
	  {1,2,3,4}, // 28
	  {2,3,4,5}, // 29
	  {3,4,5,6}, // 30
          {4,5,6,7}}; // 31 
  const uint32_t __attribute__ ((aligned (16))) masksa[4] = {0x80,0x40,0x20,0x10};
  const uint32_t __attribute__ ((aligned (16))) masksb[4] = {0x8,0x4,0x2,0x1};
  const uint32_t __attribute__ ((aligned (16))) masksc[4] = {0x8000,0x4000,0x2000,0x1000};
  const uint32_t __attribute__ ((aligned (16))) masksd[4] = {0x800,0x400,0x200,0x100};
  const uint32_t __attribute__ ((aligned (16))) maskse[4] = {0x800000,0x400000,0x200000,0x100000};
  const uint32_t __attribute__ ((aligned (16))) masksf[4] = {0x80000,0x40000,0x20000,0x10000};
  const uint32_t __attribute__ ((aligned (16))) masksg[4] = {0x80000000,0x40000000,0x20000000,0x10000000};
  const uint32_t __attribute__ ((aligned (16))) masksh[4] = {0x8000000,0x4000000,0x2000000,0x1000000};
  int32x4_t vshifta[32],vshiftb[32],vshiftc[32],vshiftd[32],vshifte[32],vshiftf[32],vshiftg[32],vshifth[32];
  uint32x4_t vmasksa  = vld1q_u32(masksa);
  uint32x4_t vmasksb  = vld1q_u32(masksb);
  uint32x4_t vmasksc  = vld1q_u32(masksc);
  uint32x4_t vmasksd  = vld1q_u32(masksd);
  uint32x4_t vmaskse  = vld1q_u32(maskse);
  uint32x4_t vmasksf  = vld1q_u32(masksf);
  uint32x4_t vmasksg  = vld1q_u32(masksg);
  uint32x4_t vmasksh  = vld1q_u32(masksh);
  uint32x4_t in;

  for (int n=0;n<32;n++) { 
    vshifta[n] = vld1q_s32(ucShifta[n]);
    vshiftb[n] = vld1q_s32(ucShiftb[n]);
    vshiftc[n] = vld1q_s32(ucShiftc[n]);
    vshiftd[n] = vld1q_s32(ucShiftd[n]);
    vshifte[n] = vld1q_s32(ucShifte[n]);
    vshiftf[n] = vld1q_s32(ucShiftf[n]);
    vshiftg[n] = vld1q_s32(ucShiftg[n]);
    vshifth[n] = vld1q_s32(ucShifth[n]);
  }
  i2=0;
  int j0=0,j1=0,j2=0,j3=0;
  if (impp->n_segments <= 32) {
	  j0=impp->n_segments;
  }
  else if (impp->n_segments <= 64) {
     j0=32;
     j1=impp->n_segments-32;
  }
  else if (impp->n_segments <= 96) {
     j0=32; j1=32;
     j2=impp->n_segments-64;
  }
  else if (impp->n_segments <= 128) {
     j0=32; j1=32; j2=32;
     j3=impp->n_segments-96;
  }
  uint32x4_t *ccp,cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7;
  for (int i=0; i < (block_length>>5); i++,i2+=8) {
    in = vdupq_n_u32(((uint32_t*)input[0])[i]);
    cc0 = vshlq_u32(vandq_u32(in,vmasksa),vshifta[0]);
    cc1 = vshlq_u32(vandq_u32(in,vmasksb),vshiftb[0]);
    cc2 = vshlq_u32(vandq_u32(in,vmasksc),vshiftc[0]);
    cc3 = vshlq_u32(vandq_u32(in,vmasksd),vshiftd[0]);
    cc4 = vshlq_u32(vandq_u32(in,vmaskse),vshifte[0]);
    cc5 = vshlq_u32(vandq_u32(in,vmasksf),vshiftf[0]);
    cc6 = vshlq_u32(vandq_u32(in,vmasksg),vshiftg[0]);
    cc7 = vshlq_u32(vandq_u32(in,vmasksh),vshifth[0]);
    for (int j = 1; j < j0; j++) {
      in = vdupq_n_u32(((uint32_t*)input[j])[i]);
      cc0 = vorrq_u32(cc0,vshlq_u32(vandq_u32(in,vmasksa),vshifta[j]));
      cc1 = vorrq_u32(cc1,vshlq_u32(vandq_u32(in,vmasksb),vshiftb[j]));
      cc2 = vorrq_u32(cc2,vshlq_u32(vandq_u32(in,vmasksc),vshiftc[j]));
      cc3 = vorrq_u32(cc3,vshlq_u32(vandq_u32(in,vmasksd),vshiftd[j]));
      cc4 = vorrq_u32(cc4,vshlq_u32(vandq_u32(in,vmaskse),vshifte[j]));
      cc5 = vorrq_u32(cc5,vshlq_u32(vandq_u32(in,vmasksf),vshiftf[j]));
      cc6 = vorrq_u32(cc6,vshlq_u32(vandq_u32(in,vmasksg),vshiftg[j]));
      cc7 = vorrq_u32(cc7,vshlq_u32(vandq_u32(in,vmasksh),vshifth[j]));
    }
    ccp=&((uint32x4_t *)cc[0])[i2];
    ccp[0] = cc0;
    ccp[1] = cc1;
    ccp[2] = cc2;
    ccp[3] = cc3;
    ccp[4] = cc4;
    ccp[5] = cc5;
    ccp[6] = cc6;
    ccp[7] = cc7;
    if (j1>0) {
      in = vdupq_n_u32(((uint32_t*)input[32])[i]);
      cc0 = vshlq_u32(vandq_u32(in,vmasksa),vshifta[0]);
      cc1 = vshlq_u32(vandq_u32(in,vmasksb),vshiftb[0]);
      cc2 = vshlq_u32(vandq_u32(in,vmasksc),vshiftc[0]);
      cc3 = vshlq_u32(vandq_u32(in,vmasksd),vshiftd[0]);
      cc4 = vshlq_u32(vandq_u32(in,vmaskse),vshifte[0]);
      cc5 = vshlq_u32(vandq_u32(in,vmasksf),vshiftf[0]);
      cc6 = vshlq_u32(vandq_u32(in,vmasksg),vshiftg[0]);
      cc7 = vshlq_u32(vandq_u32(in,vmasksh),vshifth[0]);
      for (int j = 1; j < j1; j++) {
        in = vdupq_n_u32(((uint32_t*)input[32+j])[i]);
        cc0 = vorrq_u32(cc0,vshlq_u32(vandq_u32(in,vmasksa),vshifta[j]));
        cc1 = vorrq_u32(cc1,vshlq_u32(vandq_u32(in,vmasksb),vshiftb[j]));
        cc2 = vorrq_u32(cc2,vshlq_u32(vandq_u32(in,vmasksc),vshiftc[j]));
        cc3 = vorrq_u32(cc3,vshlq_u32(vandq_u32(in,vmasksd),vshiftd[j]));
        cc4 = vorrq_u32(cc4,vshlq_u32(vandq_u32(in,vmaskse),vshifte[j]));
        cc5 = vorrq_u32(cc5,vshlq_u32(vandq_u32(in,vmasksf),vshiftf[j]));
        cc6 = vorrq_u32(cc6,vshlq_u32(vandq_u32(in,vmasksg),vshiftg[j]));
        cc7 = vorrq_u32(cc7,vshlq_u32(vandq_u32(in,vmasksh),vshifth[j]));
      }
      ccp=&((uint32x4_t *)cc[1])[i2];
      ccp[0] = cc0;
      ccp[1] = cc1;
      ccp[2] = cc2;
      ccp[3] = cc3;
      ccp[4] = cc4;
      ccp[5] = cc5;
      ccp[6] = cc6;
      ccp[7] = cc7;
    }
    if (j2>0) {
      in = vdupq_n_u32(((uint32_t*)input[64])[i]);
      cc0 = vshlq_u32(vandq_u32(in,vmasksa),vshifta[0]);
      cc1 = vshlq_u32(vandq_u32(in,vmasksb),vshiftb[0]);
      cc2 = vshlq_u32(vandq_u32(in,vmasksc),vshiftc[0]);
      cc3 = vshlq_u32(vandq_u32(in,vmasksd),vshiftd[0]);
      cc4 = vshlq_u32(vandq_u32(in,vmaskse),vshifte[0]);
      cc5 = vshlq_u32(vandq_u32(in,vmasksf),vshiftf[0]);
      cc6 = vshlq_u32(vandq_u32(in,vmasksg),vshiftg[0]);
      cc7 = vshlq_u32(vandq_u32(in,vmasksh),vshifth[0]);
      for (int j = 1; j < j2; j++) {
        in = vdupq_n_u32(((uint32_t*)input[64+j])[i]);
        cc0 = vorrq_u32(cc0,vshlq_u32(vandq_u32(in,vmasksa),vshifta[j]));
        cc1 = vorrq_u32(cc1,vshlq_u32(vandq_u32(in,vmasksb),vshiftb[j]));
        cc2 = vorrq_u32(cc2,vshlq_u32(vandq_u32(in,vmasksc),vshiftc[j]));
        cc3 = vorrq_u32(cc3,vshlq_u32(vandq_u32(in,vmasksd),vshiftd[j]));
        cc4 = vorrq_u32(cc4,vshlq_u32(vandq_u32(in,vmaskse),vshifte[j]));
        cc5 = vorrq_u32(cc5,vshlq_u32(vandq_u32(in,vmasksf),vshiftf[j]));
        cc6 = vorrq_u32(cc6,vshlq_u32(vandq_u32(in,vmasksg),vshiftg[j]));
        cc7 = vorrq_u32(cc7,vshlq_u32(vandq_u32(in,vmasksh),vshifth[j]));
      }
      ccp=&((uint32x4_t *)cc[2])[i2];
      ccp[0] = cc0;
      ccp[1] = cc1;
      ccp[2] = cc2;
      ccp[3] = cc3;
      ccp[4] = cc4;
      ccp[5] = cc5;
      ccp[6] = cc6;
      ccp[7] = cc7;
    }
    if (j3>0) {
      in = vdupq_n_u32(((uint32_t*)input[96])[i]);
      cc0 = vshlq_u32(vandq_u32(in,vmasksa),vshifta[0]);
      cc1 = vshlq_u32(vandq_u32(in,vmasksb),vshiftb[0]);
      cc2 = vshlq_u32(vandq_u32(in,vmasksc),vshiftc[0]);
      cc3 = vshlq_u32(vandq_u32(in,vmasksd),vshiftd[0]);
      cc4 = vshlq_u32(vandq_u32(in,vmaskse),vshifte[0]);
      cc5 = vshlq_u32(vandq_u32(in,vmasksf),vshiftf[0]);
      cc6 = vshlq_u32(vandq_u32(in,vmasksg),vshiftg[0]);
      cc7 = vshlq_u32(vandq_u32(in,vmasksh),vshifth[0]);
      for (int j = 1; j < j3; j++) {
        in = vdupq_n_u32(((uint32_t*)input[96+j])[i]);
        cc0 = vorrq_u32(cc0,vshlq_u32(vandq_u32(in,vmasksa),vshifta[j]));
        cc1 = vorrq_u32(cc1,vshlq_u32(vandq_u32(in,vmasksb),vshiftb[j]));
        cc2 = vorrq_u32(cc2,vshlq_u32(vandq_u32(in,vmasksc),vshiftc[j]));
        cc3 = vorrq_u32(cc3,vshlq_u32(vandq_u32(in,vmasksd),vshiftd[j]));
        cc4 = vorrq_u32(cc4,vshlq_u32(vandq_u32(in,vmaskse),vshifte[j]));
        cc5 = vorrq_u32(cc5,vshlq_u32(vandq_u32(in,vmasksf),vshiftf[j]));
        cc6 = vorrq_u32(cc6,vshlq_u32(vandq_u32(in,vmasksg),vshiftg[j]));
        cc7 = vorrq_u32(cc7,vshlq_u32(vandq_u32(in,vmasksh),vshifth[j]));
      }
      ccp=&((uint32x4_t *)cc[3])[i2];
      ccp[0] = cc0;
      ccp[1] = cc1;
      ccp[2] = cc2;
      ccp[3] = cc3;
      ccp[4] = cc4;
      ccp[5] = cc5;
      ccp[6] = cc6;
      ccp[7] = cc7;
    }
  }
#endif
#endif
#endif
  if(impp->tinput != NULL) stop_meas(impp->tinput);

  if (BG==1 && Zc==384)  {
    //parity check part
    if(impp->tparity != NULL) start_meas(impp->tparity);
    uint32_t *ccp[n_inputs];
    for (int s=0;s<n_inputs;s++) {
      ccp[s]=cc[s];
    }
    encode_parity_check_part_cuda(ccp, dd0, BG, Zc, Kb, ncols,n_inputs);
    if(impp->tparity != NULL) stop_meas(impp->tparity);
  }
  else {
	  AssertFatal(1==0,"Only BG1 Zc=384 for now\n");
  }


  if(impp->toutput != NULL) start_meas(impp->toutput);
  for (int s=0;s<n_inputs;s++) {
    memcpy(output[s],&cc[s][2*Zc],sizeof(uint32_t)*(block_length-(2*Zc)));
    cudaError_t err = cudaMemcpy(&output[s][block_length-(2*Zc)],dd0[s],sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit),2);
    AssertFatal(err == cudaSuccess, "dd0[%d] %p CUDA Error: %s\n", s, dd0[s],cudaGetErrorString(err)); 							
  }
  if(impp->toutput != NULL) stop_meas(impp->toutput);
  return 0;
}


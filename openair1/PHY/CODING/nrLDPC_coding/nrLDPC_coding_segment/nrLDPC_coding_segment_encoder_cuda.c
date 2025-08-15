/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
 * \brief Top-level routines for implementing LDPC encoding of transport channels
 */

#include "nr_rate_matching.h"
#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include "common/utils/nr/nr_common.h"
#include <openair2/UTIL/OPT/opt.h>

#include <syscall.h>

#define DEBUG_LDPC_ENCODING
//#define DEBUG_LDPC_ENCODING_FREE 1
/*
static void write_task_output(uint8_t *f,
                              uint32_t E,
                              uint8_t *f2,
                              uint32_t E2,
                              bool Eshift,
                              uint32_t E2_first_segment,
                              uint32_t nb_segments,
                              uint8_t *output,
                              uint32_t Eoffset)
{

#if defined(__AVX512VBMI__)
  uint64_t *output_p = (uint64_t*)output;
  __m512i inc = _mm512_set1_epi8(0x1);

  for (int i=0;i<E2;i+=64) {
    uint32_t Eoffset2 = Eoffset;
    __m512i bitperm = _mm512_set1_epi64(0x3830282018100800);
    if (i<E) {
      for (int j=0; j < E2_first_segment; j++) {
        // Note: Here and below for AVX2, we are using the 64-bit SIMD instruction
        // instead of C >>/<< because when the Eoffset2_bit is 64 or 0, the <<
        // and >> operations are undefined and in fact don't give "0" which is
        // what we want here. The SIMD version do give 0 when the shift is 64
        uint32_t Eoffset2_byte = Eoffset2 >> 6;
        uint32_t Eoffset2_bit = Eoffset2 & 63;
        __m64 tmp = (__m64)_mm512_bitshuffle_epi64_mask(((__m512i *)f)[i >> 6],bitperm);
        *(__m64*)(output_p + Eoffset2_byte)   = _mm_or_si64(*(__m64*)(output_p + Eoffset2_byte),_mm_slli_si64(tmp,Eoffset2_bit));
        *(__m64*)(output_p + Eoffset2_byte+1) = _mm_or_si64(*(__m64*)(output_p + Eoffset2_byte+1),_mm_srli_si64(tmp,(64-Eoffset2_bit)));
        Eoffset2 += E;
        bitperm = _mm512_add_epi8(bitperm ,inc);
      }
    } else {
      for (int j=0; j < E2_first_segment; j++) {
        Eoffset2 += E;
        bitperm = _mm512_add_epi8(bitperm ,inc);
      }
    }
    for (int j=E2_first_segment; j < nb_segments; j++) {
      uint32_t Eoffset2_byte = Eoffset2 >> 6;
      uint32_t Eoffset2_bit = Eoffset2 & 63;
      __m64 tmp = (__m64)_mm512_bitshuffle_epi64_mask(((__m512i *)f2)[i >> 6],bitperm);
      *(__m64*)(output_p + Eoffset2_byte)   = _mm_or_si64(*(__m64*)(output_p + Eoffset2_byte),_mm_slli_si64(tmp,Eoffset2_bit));
      *(__m64*)(output_p + Eoffset2_byte+1) = _mm_or_si64(*(__m64*)(output_p + Eoffset2_byte+1),_mm_srli_si64(tmp,(64-Eoffset2_bit)));
      Eoffset2 += E2;
      bitperm = _mm512_add_epi8(bitperm ,inc);
    }
    output_p++;
  }

#elif defined(__aarch64__)
  uint16_t *output_p = (uint16_t*)output;
  const int8_t __attribute__ ((aligned (16))) ucShift[8][16] = {
    {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7},     // segment 0
    {-1,0,1,2,3,4,5,6,-1,0,1,2,3,4,5,6},   // segment 1
    {-2,-1,0,1,2,3,4,5,-2,-1,0,1,2,3,4,5}, // segment 2
    {-3,-2,-1,0,1,2,3,4,-3,-2,-1,0,1,2,3,4}, // segment 3
    {-4,-3,-2,-1,0,1,2,3,-4,-3,-2,-1,0,1,2,3}, // segment 4
    {-5,-4,-3,-2,-1,0,1,2,-5,-4,-3,-2,-1,0,1,2}, // segment 5
    {-6,-5,-4,-3,-2,-1,0,1,-6,-5,-4,-3,-2,-1,0,1}, // segment 6
    {-7,-6,-5,-4,-3,-2,-1,0,-7,-6,-5,-4,-3,-2,-1,0}}; // segment 7
  const uint8_t __attribute__ ((aligned (16))) masks[16] = 
      {0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80};
  int8x16_t vshift[8];
  for (int n=0;n<8;n++) vshift[n] = vld1q_s8(ucShift[n]);
  uint8x16_t vmask  = vld1q_u8(masks);

  for (int i=0;i<E2;i+=16) {
    uint32_t Eoffset2 = Eoffset;
    if (i<E) {	
      for (int j=0; j < E2_first_segment; j++) {
        uint32_t Eoffset2_byte = Eoffset2 >> 4;
        uint32_t Eoffset2_bit = Eoffset2 & 15;
        uint8x16_t cshift = vandq_u8(vshlq_u8(((uint8x16_t*)f)[i >> 4],vshift[j]),vmask);
        int32_t tmp = (int)vaddv_u8(vget_low_u8(cshift));
        tmp += (int)(vaddv_u8(vget_high_u8(cshift))<<8);
        *(output_p + Eoffset2_byte)   |= (uint16_t)(tmp<<Eoffset2_bit);
        *(output_p + Eoffset2_byte+1) |= (uint16_t)(tmp>>(16-Eoffset2_bit));
        Eoffset2 += E;
      }
    } else {
      for (int j=0; j < E2_first_segment; j++) {
        Eoffset2 += E;
      }
    }
    for (int j=E2_first_segment; j < nb_segments; j++) {
      uint32_t Eoffset2_byte = Eoffset2 >> 4;
      uint32_t Eoffset2_bit = Eoffset2 & 15;
      uint8x16_t cshift = vandq_u8(vshlq_u8(((uint8x16_t*)f2)[i >> 4],vshift[j]),vmask);
      int32_t tmp = (int)vaddv_u8(vget_low_u8(cshift));
      tmp += (int)(vaddv_u8(vget_high_u8(cshift))<<8);
      *(output_p + Eoffset2_byte)   |= (uint16_t)(tmp<<Eoffset2_bit);
      *(output_p + Eoffset2_byte+1) |= (uint16_t)(tmp>>(16-Eoffset2_bit));
      Eoffset2 += E2;
    }
    output_p++;
  }
       
#else
  uint32_t *output_p = (uint32_t*)output;

  for (int i=0; i < E2; i += 32) {
    uint32_t Eoffset2 = Eoffset;
    if (i < E) {
      for (int j = 0; j < E2_first_segment; j++) {
        // Note: Here and below, we are using the 64-bit SIMD instruction
        // instead of C >>/<< because when the Eoffset2_bit is 64 or 0, the <<
        // and >> operations are undefined and in fact don't give "0" which is
        // what we want here. The SIMD version do give 0 when the shift is 64
        uint32_t Eoffset2_byte = Eoffset2 >> 5;
        uint32_t Eoffset2_bit = Eoffset2 & 31;
        int tmp = _mm256_movemask_epi8(_mm256_slli_epi16(((__m256i *)f)[i >> 5], 7 - j));
        __m64 tmp64 = _mm_set1_pi32(tmp);
        __m64 out64 = _mm_set_pi32(*(output_p + Eoffset2_byte + 1), *(output_p + Eoffset2_byte));
        __m64 tmp64b = _mm_or_si64(out64, _mm_slli_pi32(tmp64, Eoffset2_bit));
        __m64 tmp64c = _mm_or_si64(out64, _mm_srli_pi32(tmp64, (32 - Eoffset2_bit)));
        *(output_p + Eoffset2_byte) = _m_to_int(tmp64b);
        *(output_p + Eoffset2_byte + 1) = _m_to_int(_mm_srli_si64(tmp64c, 32));
        Eoffset2 += E;
      }
    } else {
      for (int j = 0; j < E2_first_segment; j++) {
        Eoffset2 += E;
      }
    } 
    for (int j = E2_first_segment; j < nb_segments; j++) {
      uint32_t Eoffset2_byte = Eoffset2 >> 5;
      uint32_t Eoffset2_bit = Eoffset2 & 31;
      int tmp = _mm256_movemask_epi8(_mm256_slli_epi16(((__m256i *)f2)[i >> 5], 7 - j));
      __m64 tmp64 = _mm_set1_pi32(tmp);
      __m64 out64 = _mm_set_pi32(*(output_p + Eoffset2_byte + 1), *(output_p + Eoffset2_byte));
      __m64 tmp64b = _mm_or_si64(out64, _mm_slli_pi32(tmp64, Eoffset2_bit));
      __m64 tmp64c = _mm_or_si64(out64, _mm_srli_pi32(tmp64, (32 - Eoffset2_bit)));
      *(output_p + Eoffset2_byte)  = _m_to_int(tmp64b);
      *(output_p + Eoffset2_byte + 1) = _m_to_int(_mm_srli_si64(tmp64c, 32));
      Eoffset2 += E2;
    }
    output_p++;
  }

  
#endif
}
*/


static void unpack_output(uint32_t *f,
                         uint32_t E,
                         uint32_t *f2,
                         uint32_t E2,
			 uint32_t E2_first_segment32,
                         uint32_t E2_first_segment,
                         uint32_t nb_segments,
                         uint8_t *output) {


  int s;
 // int s0;
  uint32_t *fp;
  int foffset;
  uint32_t *output_p = (uint32_t *)output;
//  printf("E %d, E2 %d, E2_first_segment %d, E2_first_segment32 %d, nb_segments %d\n",E,E2,E2_first_segment,E2_first_segment32,nb_segments);
  uint32_t bit_index = 0;
#if 1
  const int32_t ucShift0[32][4] = { {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}, {-12,-11,-10,-9}, {-13,-12,-11,-10}, {-14,-13,-12,-11}, {-15,-14,-13,-12}, {-16,-15,-14,-13}, {-17,-16,-15,-14}, {-18,-17,-16,-15}, {-19,-18,-17,-16}, {-20,-19,-18,-17}, {-21,-20,-19,-18}, {-22,-21,-20,-19}, {-23,-22,-21,-20}, {-24,-23,-22,-21}, {-25,-24,-23,-22}, {-26,-25,-24,-23}, {-27,-26,-25,-24}, {-28,-27,-26,-25}, {-29,-28,-27,-26}, {-30,-29,-28,-27}, {-31,-30,-29,-28}}; 

  const int32_t ucShift1[32][4] = { {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}, {-12,-11,-10,-9}, {-13,-12,-11,-10}, {-14,-13,-12,-11}, {-15,-14,-13,-12}, {-16,-15,-14,-13}, {-17,-16,-15,-14}, {-18,-17,-16,-15}, {-19,-18,-17,-16}, {-20,-19,-18,-17}, {-21,-20,-19,-18}, {-22,-21,-20,-19}, {-23,-22,-21,-20}, {-24,-23,-22,-21}, {-25,-24,-23,-22}, {-26,-25,-24,-23}, {-27,-26,-25,-24}}; 

  const int32_t ucShift2[32][4] = { {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}, {-12,-11,-10,-9}, {-13,-12,-11,-10}, {-14,-13,-12,-11}, {-15,-14,-13,-12}, {-16,-15,-14,-13}, {-17,-16,-15,-14}, {-18,-17,-16,-15}, {-19,-18,-17,-16}, {-20,-19,-18,-17}, {-21,-20,-19,-18}, {-22,-21,-20,-19},{-23,-22,-21,-20}}; 

  const int32_t ucShift3[32][4] = { {12,13,14,15}, {11,12,13,14}, {10,11,12,13}, {9,10,11,12}, {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}, {-12,-11,-10,-9}, {-13,-12,-11,-10}, {-14,-13,-12,-11}, {-15,-14,-13,-12}, {-16,-15,-14,-13}, {-17,-16,-15,-14}, {-18,-17,-16,-15}, {-19,-18,-17,-16}}; 

  const int32_t ucShift4[32][4] = { {16,17,18,19}, {15,16,17,18}, {14,15,16,17}, {13,14,15,16}, {12,13,14,15}, {11,12,13,14}, {10,11,12,13}, {9,10,11,12}, {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}, {-12,-11,-10,-9}, {-13,-12,-11,-10}, {-14,-13,-12,-11}, {-15,-14,-13,-12}}; 

  const int32_t ucShift5[32][4] = { {20,21,22,23}, {19,20,21,22}, {18,19,20,21}, {17,18,19,20}, {16,17,18,19}, {15,16,17,18}, {14,15,16,17}, {13,14,15,16}, {12,13,14,15}, {11,12,13,14}, {10,11,12,13}, {9,10,11,12}, {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}, {-8,-7,-6,-5}, {-9,-8,-7,-6}, {-10,-9,-8,-7}, {-11,-10,-9,-8}}; 

  const int32_t ucShift6[32][4] = { {24,25,26,27}, {23,24,25,26}, {22,23,24,25}, {21,22,23,24}, {20,21,22,23}, {19,20,21,22}, {18,19,20,21}, {17,18,19,20}, {16,17,18,19}, {15,16,17,18}, {14,15,16,17}, {13,14,15,16}, {12,13,14,15}, {11,12,13,14}, {10,11,12,13}, {9,10,11,12}, {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}, {-4,-3,-2,-1}, {-5,-4,-3,-2}, {-6,-5,-4,-3}, {-7,-6,-5,-4}}; 

  const int32_t ucShift7[32][4] = { {28,29,30,31}, {27,28,29,30}, {26,27,28,29}, {25,26,27,28}, {24,25,26,27}, {23,24,25,26}, {22,23,24,25}, {21,22,23,24}, {20,21,22,23}, {19,20,21,22}, {18,19,20,21}, {17,18,19,20}, {16,17,18,19}, {15,16,17,18}, {14,15,16,17}, {13,14,15,16}, {12,13,14,15}, {11,12,13,14}, {10,11,12,13}, {9,10,11,12}, {8,9,10,11},{7,8,9,10}, {6,7,8,9}, {5,6,7,8}, {4,5,6,7}, {3,4,5,6}, {2,3,4,5}, {1,2,3,4}, {0,1,2,3}, {-1,0,1,2},{-2,-1,0,1}, {-3,-2,-1,0}}; 
  const uint32_t __attribute__ ((aligned (16))) masks0[4] = {0x1,0x2,0x4,0x8};
  const uint32_t __attribute__ ((aligned (16))) masks1[4] = {0x10,0x20,0x40,0x80};
  const uint32_t __attribute__ ((aligned (16))) masks2[4] = {0x100,0x200,0x400,0x800};
  const uint32_t __attribute__ ((aligned (16))) masks3[4] = {0x1000,0x2000,0x4000,0x8000};
  const uint32_t __attribute__ ((aligned (16))) masks4[4] = {0x10000,0x20000,0x40000,0x80000};
  const uint32_t __attribute__ ((aligned (16))) masks5[4] = {0x100000,0x200000,0x400000,0x800000};
  const uint32_t __attribute__ ((aligned (16))) masks6[4] = {0x1000000,0x2000000,0x4000000,0x8000000};
  const uint32_t __attribute__ ((aligned (16))) masks7[4] = {0x10000000,0x20000000,0x40000000,0x80000000};
  int32x4_t vshift0[32],vshift1[32],vshift2[32],vshift3[32],vshift4[32],vshift5[32],vshift6[32],vshift7[32];
  for (int n=0;n<32;n++) {
	  vshift0[n] = vld1q_s32(ucShift0[n]);
	  vshift1[n] = vld1q_s32(ucShift1[n]);
	  vshift2[n] = vld1q_s32(ucShift2[n]);
	  vshift3[n] = vld1q_s32(ucShift3[n]);
	  vshift4[n] = vld1q_s32(ucShift4[n]);
	  vshift5[n] = vld1q_s32(ucShift5[n]);
	  vshift6[n] = vld1q_s32(ucShift6[n]);
	  vshift7[n] = vld1q_s32(ucShift7[n]);
  }
  uint32x4_t vmask0  = vld1q_u32(masks0);
  uint32x4_t vmask1  = vld1q_u32(masks1);
  uint32x4_t vmask2  = vld1q_u32(masks2);
  uint32x4_t vmask3  = vld1q_u32(masks3);
  uint32x4_t vmask4  = vld1q_u32(masks4);
  uint32x4_t vmask5  = vld1q_u32(masks5);
  uint32x4_t vmask6  = vld1q_u32(masks6);
  uint32x4_t vmask7  = vld1q_u32(masks7);
  uint32_t output_tmp=0;
  int s2=0;
  for (s = 0; s < E2_first_segment ; s++) {
    s2 = s&31;	  
    foffset = (s>>5)*E;
    fp = f+foffset;
    int i;
    if ((bit_index&31) == 0 ) {
      for (i = 0; i < (E>>5)<<5; i+=32) {
	uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
  	uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));

	*(output_p + (bit_index>>5))     = vaddvq_u32(cshift);
	bit_index+=32;
      }
      uint32_t Emod32=E&31;
      if (Emod32 != 0) {
        uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
  	uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));

	*(output_p + (bit_index>>5))     = vaddvq_u32(cshift)&((1<<Emod32)-1);
        bit_index+=Emod32;
      }
    }
    else {
      for (i = 0; i < (E>>5)<<5; i+=32) {
	uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
  	uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));
        uint32_t tmp = vaddvq_u32(cshift);
	*(output_p + (bit_index>>5))     |= (tmp<<(bit_index&31));
	*(output_p + (bit_index>>5)+1)   |= (tmp>>(32-(bit_index&31)));
	bit_index+=32;
      }
      uint32_t Emod32=E&31;
      if (Emod32 != 0) {
        uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
  	uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));
        uint32_t tmp = vaddvq_u32(cshift);
        tmp&=((1<<Emod32)-1);
        *(output_p + (bit_index>>5))     |= (tmp<<(bit_index&31));
	*(output_p + (bit_index>>5)+1)   |= (tmp>>(32-(bit_index&31)));
        bit_index+=Emod32;
      }
    }
  }
//  s0 = s;
  for ( ; s < nb_segments ; s++){
    s2 = s&31;	  
    foffset = ((s>>5)-E2_first_segment32)*E2;
    fp = f2+foffset;
    int i;
    if ((bit_index&31) == 0 ) {
      for (i = 0; i < (E2>>5)<<5; i+=32) {
	uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
  	uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));
	*(output_p + (bit_index>>5))     = vaddvq_u32(cshift);
	bit_index+=32;
      }
      uint32_t E2mod32=E2&31;
      if (E2mod32 != 0) {
        uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
        uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));
        *(output_p + (bit_index>>5))     = vaddvq_u32(cshift)&((1<<E2mod32)-1);
        bit_index+=E2mod32;
      }
    }
    else {
      for (i = 0; i < (E2>>5)<<5; i+=32) {
	uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
        uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[7],vshift7[s2]),vmask7));
        uint32_t tmp = vaddvq_u32(cshift);
	*(output_p + (bit_index>>5))     |= (tmp<<(bit_index&31));
	*(output_p + (bit_index>>5)+1)   |= (tmp>>(32-(bit_index&31)));
	bit_index+=32;
      }
      uint32_t E2mod32=E2&31;
      if (E2mod32 != 0) {
        uint32x4_t *fp128 = (uint32x4_t*)&fp[i];    
        uint32x4_t cshift = vandq_u32(vshlq_u32(fp128[0],vshift0[s2]),vmask0);
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[1],vshift1[s2]),vmask1));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[2],vshift2[s2]),vmask2));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[3],vshift3[s2]),vmask3));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[4],vshift4[s2]),vmask4));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[5],vshift5[s2]),vmask5));
  	cshift = vorrq_u32(cshift,vandq_u32(vshlq_u32(fp128[6],vshift6[s2]),vmask6));
        uint32_t tmp = vaddvq_u32(cshift);
        tmp&=((1<<E2mod32)-1);
        *(output_p + (bit_index>>5))     |= (tmp<<(bit_index&31));
	*(output_p + (bit_index>>5)+1)   |= (tmp>>(32-(bit_index&31)));
        bit_index+=E2mod32;
      }
    }
  }
#else // non SIMD version
  int segpos;	
  for (s = 0; s < E2_first_segment ; s++) {
    foffset = (s>>5)*E;
    fp = f+foffset;
    segpos = (1<<s);
    for (int i = 0; i < E; i++) {
      output_p[bit_index>>5]|=((fp[i] & segpos)!=0)<<(bit_index&31); 
      bit_index++;
    }
  }
  s0 = s;
  for ( ; s < nb_segments ; s++){
    foffset = ((s-s0)>>5)*E2;
    fp = f2+foffset;
    segpos = (1<<s);
    for (int i = 0; i < E2; i++) {
      output_p[bit_index>>5]|=((fp[i] & segpos)!=0)<<(bit_index&31); 
      bit_index++;
    }
  }
#endif  
}
/**
 * \typedef ldpc8blocks_args_t
 * \struct ldpc8blocks_args_s
 * \brief Arguments of an encoding task
 * encode up to 8 code blocks
 * \var nrLDPC_TB_encoding_parameters TB encoding parameters as defined in the coding library interface
 * \var impp encoder implementation specific parameters for the task
 * \var f first interleaver output to be filled by the task
 * \var f2 second interleaver output to be filled by the task
 * in case of a shift of E in the code blocks group processed by the task
 */

static void ldpcnblocks(nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters, encoder_implemparams_t impp)
{
 
  uint8_t mod_order = nrLDPC_TB_encoding_parameters->Qm;
  uint16_t nb_rb = nrLDPC_TB_encoding_parameters->nb_rb;
  uint32_t A = nrLDPC_TB_encoding_parameters->A;

  unsigned int G = nrLDPC_TB_encoding_parameters->G;
  LOG_D(PHY, "dlsch coding A %d K %d G %d (nb_rb %d, mod_order %d)\n", A, impp.K, G, nb_rb, (int)mod_order);

  // nrLDPC_encoder output is in "d"
  // let's make this interface happy!
  uint32_t d[4][68*384];
  uint8_t *c[nrLDPC_TB_encoding_parameters->C];

  
  for (int r = 0; r < nrLDPC_TB_encoding_parameters->C; r++)
    c[r] = nrLDPC_TB_encoding_parameters->segments[r].c;
  start_meas(&nrLDPC_TB_encoding_parameters->segments[impp.first_seg].ts_ldpc_encode);
  LDPCencoder32(c, d, &impp);
  stop_meas(&nrLDPC_TB_encoding_parameters->segments[impp.first_seg].ts_ldpc_encode);
  // Compute where to place in output buffer that is concatenation of all segments

#ifdef DEBUG_LDPC_ENCODING
  LOG_D(PHY, "rvidx in encoding = %d\n", nrLDPC_TB_encoding_parameters->rv_index);
#endif
  const uint32_t E = nrLDPC_TB_encoding_parameters->segments[0].E;
  uint32_t E2=E;
  uint32_t Emax = E;
  int n_seg   = nrLDPC_TB_encoding_parameters->C>>5;
  int n_seg2  = n_seg;
  if ((nrLDPC_TB_encoding_parameters->C & 31) > 0) n_seg2++;
  int r_shift = n_seg2; 
  int r_shift2 = nrLDPC_TB_encoding_parameters->C;
  for (int s=0;s<nrLDPC_TB_encoding_parameters->C;s++) {
      //printf("segment %d E %d\n",s,nrLDPC_TB_encoding_parameters->segments[s].E);	  
      if (nrLDPC_TB_encoding_parameters->segments[s].E != E) {
	 E2=nrLDPC_TB_encoding_parameters->segments[s].E;
         if(E2 > Emax)
           Emax = E2;
	 r_shift = s>>5;
	 r_shift2 = s;
	// printf("r_shift %d, r_shift2 %d\n",r_shift,r_shift2);
         break;
      }	 
  }    

  LOG_D(NR_PHY,
        "Rate Matching, Code segment %d...%d r_shift %d n_seg2 %d (coded bits (G) %u, E %d, E2 %d Filler bits %d, Filler offset %d mod_order %d, nb_rb "
          "%d,nrOfLayer %d)...\n",
        0,
        impp.n_segments-1,
	r_shift,
	n_seg2,
        G,
        E,E2,
        impp.F,
        impp.K - impp.F - 2 * impp.Zc,
        mod_order,
        nb_rb,
        nrLDPC_TB_encoding_parameters->nb_layers);
/*
  printf("Rate Matching, Code segment 0..%d r_shift %d r_shift2 %d n_seg2 %d (coded bits (G) %u, E %d, E2 %d Filler bits %d, Filler offset %d mod_order %d, nb_rb "
          "%d,nrOfLayer %d)...\n",
        impp.n_segments-1,
	r_shift,
	r_shift2,
	n_seg2,
        G,
        E,E2,
        impp.F,
        impp.K - impp.F - 2 * impp.Zc,
        mod_order,
        nb_rb,
        nrLDPC_TB_encoding_parameters->nb_layers);
*/

  uint32_t Tbslbrm = nrLDPC_TB_encoding_parameters->tbslbrm;

  uint32_t e[E*(r_shift+1)];
  uint32_t e2[E2*(n_seg2-r_shift)];
  uint32_t f[E*(r_shift+1)];
  uint32_t f2[E2*(n_seg2-r_shift)];

  // Interleaver outputs are stored in the output arrays
  uint8_t *output = nrLDPC_TB_encoding_parameters->output;

  start_meas(&nrLDPC_TB_encoding_parameters->segments[0].ts_rate_match);
  memset(e,0,sizeof(e));
  memset(f,0,sizeof(f));
  if (1/*r_shift < n_seg2*/) { 
    memset(e2,0,sizeof(e2));
    memset(f2,0,sizeof(f2));
  }

  for (int r=0;r<n_seg2;r++) {
    if (r<=r_shift)	  
      nr_rate_matching_ldpc32(Tbslbrm,
                              impp.BG,
                              impp.Zc,
                              d[r],
                              e+(r*E),
                              impp.n_segments,
                              impp.F,
                              impp.K - impp.F - 2 * impp.Zc,
                              nrLDPC_TB_encoding_parameters->rv_index,
                              E);
    if (r>=r_shift)	  
      nr_rate_matching_ldpc32(Tbslbrm,
                              impp.BG,
                              impp.Zc,
                              d[r],
                              e2+((r-r_shift)*E2),
                              impp.n_segments,
                              impp.F,
                              impp.K - impp.F - 2 * impp.Zc,
                              nrLDPC_TB_encoding_parameters->rv_index,
                              E2);
   /* 
    if (r==(n_seg2-1)) {
	    for (int i=0;i<16;i++) printf("rm: %x %x\n",d[n_seg2-1][i],e2[((n_seg2-1)*E2)+i]);
    }
    */
  }
  stop_meas(&nrLDPC_TB_encoding_parameters->segments[0].ts_rate_match);
  if (impp.K - impp.F - 2 * impp.Zc > E) {
    LOG_E(PHY,
          "dlsch coding A %d  Kr %d G %d (nb_rb %d, mod_order %d)\n",
          A,
          impp.K,
          G,
          nb_rb,
          (int)mod_order);

    LOG_E(NR_PHY,
          "Rate Matching, Code segments 0..%d (coded bits (G) %u, E %d, Kr %d, Filler bits %d, Filler offset %d mod_order %d, "
          "nb_rb %d)...\n",
          impp.n_segments,
          G,
          E,
          impp.K,
          impp.F,
          impp.K - impp.F - 2 * impp.Zc,
          mod_order,
          nb_rb);
  }
  
  //printf("interleaving r_shift %d, n_seg2 %d\n",r_shift,n_seg2);
  start_meas(&nrLDPC_TB_encoding_parameters->segments[0].ts_interleave);
  
  for (int r=0;r<=r_shift;r++)
    nr_interleaving_ldpc32(E,
                           mod_order,
                           e+E*r,
                           f+E*r);

  for (int r=r_shift;r<n_seg2;r++)
    nr_interleaving_ldpc32(E2,
                           mod_order,
                           e2+E2*(r-r_shift),
                           f2+E2*(r-r_shift));
/*
  for (int i=0;i<16;i++) printf("intl (f offset %d): %x %x\n",(n_seg2-1)*E2,e2[((n_seg2-1)*E2)+i],f2[((n_seg2-1)*E2)+i]);
  printf("-------------------\n");
  for (int i=E2-16;i<E2;i++) printf("intl (f offset %d): %x %x\n",(n_seg2-1)*E2,e2[((n_seg2-1)*E2)+i],f2[((n_seg2-1)*E2)+i]);
  */
  stop_meas(&nrLDPC_TB_encoding_parameters->segments[0].ts_interleave);

  if (impp.tconcat) start_meas(impp.tconcat);
  unpack_output(f,E,f2,E2,r_shift,r_shift2,nrLDPC_TB_encoding_parameters->C,output);
  if (impp.tconcat) stop_meas(impp.tconcat);

}

int nrLDPC_coding_encoder(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{

  for (int dlsch_id = 0; dlsch_id < nrLDPC_slot_encoding_parameters->nb_TBs; dlsch_id++) {
    nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters = &nrLDPC_slot_encoding_parameters->TBs[dlsch_id];

    encoder_implemparams_t common_segment_params = {
      .n_segments = nrLDPC_TB_encoding_parameters->C,
      .tinput = nrLDPC_slot_encoding_parameters->tinput,
      .tprep = nrLDPC_slot_encoding_parameters->tprep,
      .tparity = nrLDPC_slot_encoding_parameters->tparity,
      .toutput = nrLDPC_slot_encoding_parameters->toutput,
      .tconcat = nrLDPC_slot_encoding_parameters->tconcat,
      .Kb = nrLDPC_TB_encoding_parameters->Kb,
      .Zc = nrLDPC_TB_encoding_parameters->Z,
      .BG = nrLDPC_TB_encoding_parameters->BG,
      .output = nrLDPC_TB_encoding_parameters->output, 
      .K = nrLDPC_TB_encoding_parameters->K,
      .F = nrLDPC_TB_encoding_parameters->F,
    };

    LOG_D(NR_PHY,"Calling ldpcnblocks (C %d, Z %d, K %d)\n",common_segment_params.n_segments,common_segment_params.Zc,common_segment_params.K);
    ldpcnblocks(nrLDPC_TB_encoding_parameters, common_segment_params);


  }

  return 0;
}

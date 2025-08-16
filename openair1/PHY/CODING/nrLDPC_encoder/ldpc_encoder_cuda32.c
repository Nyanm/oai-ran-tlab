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
#define USE_UMEM 1


int LDPCencoder(uint8_t **input, uint8_t *output, encoder_implemparams_t *impp)
{
  //set_log(PHY, 4);

  int Zc = impp->Zc;
  int Kb = impp->Kb;
  short block_length = impp->K;
  short BG = impp->BG;
  uint32_t *out32=(uint32_t*)output;
  int nrows=0,ncols=0;
  int rate=3;
  int no_punctured_columns,removed_bit;
  //Table of possible lifting sizes
  uint8_t temp;
  int simd_size;
  unsigned int macro_segment, macro_segment_end;

//  printf("input %p output %p\n",input[0],output);

  macro_segment = impp->first_seg;
  macro_segment_end = (impp->n_segments > impp->first_seg + 32) ? impp->first_seg + 32 : impp->n_segments;
  ///printf("macro_segment: %d\n", macro_segment);
  ///printf("macro_segment_end: %d\n", macro_segment_end );

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
  LOG_D(PHY,"ldpc_encoder_cuda32: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",BG,Zc,Kb,block_length,impp->n_segments);
  LOG_D(PHY,"ldpc_encoder_cuda32: PDU (seg 0) %x %x %x %x\n",input[0][0],input[0][1],input[0][2],input[0][3]);
#endif

  AssertFatal(Zc > 0, "no valid Zc found for block length %d\n", block_length);

  uint32_t  cc[22*Zc]; //padded input, unpacked, max size
		       //
#if USE_UMEM 
  uint32_t *dd;

  cudaError_t err=cudaMalloc((void**)&dd,46*Zc*sizeof(uint32_t));
  if (err != cudaSuccess) printf("CUDA Error: %s_1\n", cudaGetErrorString(err)); 							
#else
 uint32_t dd[46*Zc];
#endif
  // calculate number of punctured bits
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*rate)/Zc;
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length*rate);
  //printf("%d\n",no_punctured_columns);
  //printf("%d\n",removed_bit);
  // unpack input
  memset(cc,0,sizeof(cc));
#if USE_UMEM 
  err = cudaMemset(dd,0,46*Zc*sizeof(uint32_t));
  if (err != cudaSuccess) printf("CUDA Error: %s_2\n", cudaGetErrorString(err)); 							
#else
  memset(dd,0,sizeof(dd));
#endif
  if(impp->tinput != NULL) start_meas(impp->tinput);

  //interleave up to 32 transport-block segements at a time

  unsigned int i_dword = 0;

#if 0 //defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__)
  const __m512i masks5[8] = { _mm512_set1_epi8(0x1), _mm512_set1_epi8(0x2),
                              _mm512_set1_epi8(0x4), _mm512_set1_epi8(0x8),
                              _mm512_set1_epi8(0x10), _mm512_set1_epi8(0x20),
                              _mm512_set1_epi8(0x40), _mm512_set1_epi8(0x80)};
  const __m512i zero512 = _mm512_setzero_si512();
  const uint8_t perm[64]__attribute__((aligned(64))) = {7,6,5,4,3,2,1,0,         15,14,13,12,11,10,9,8,
                                                        23,22,21,20,19,18,17,16, 31,30,29,28,27,26,25,24,
                                                        39,38,37,36,35,34,33,32, 47,46,45,44,43,42,41,40,
                                                        55,54,53,52,51,50,49,48, 63,62,61,60,59,58,57,56};
  register __m512i c512;

  for (; i_byte < ((block_length >> 6) << 6); i_byte += 64) {
    unsigned int i = i_byte >> 6;
    c512 = _mm512_mask_blend_epi8(((uint64_t *)&input[macro_segment][0])[i], zero512, masks5[0]);
    for (int j = macro_segment + 1; j < macro_segment_end; j++) {
      c512 = _mm512_or_si512(c512, _mm512_mask_blend_epi8(((uint64_t *)&input[j][0])[i], zero512, masks5[j - macro_segment]));
    }
    c512 = _mm512_permutexvar_epi8(*(__m512i*)perm, c512);
    ((__m512i *)cc)[i] = c512;
  }
#endif

#if 0//ndef __aarch64__
  simde__m256i shufmask = simde_mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,0x0101010101010101, 0x0000000000000000);
  simde__m256i andmask  = simde_mm256_set1_epi64x(0x0102040810204080);  // every 8 bits -> 8 bytes, pattern repeats.
  simde__m256i zero256   = simde_mm256_setzero_si256();
  simde__m256i masks[8];
  register simde__m256i c256;
  masks[0] = simde_mm256_set1_epi8(0x1);
  masks[1] = simde_mm256_set1_epi8(0x2);
  masks[2] = simde_mm256_set1_epi8(0x4);
  masks[3] = simde_mm256_set1_epi8(0x8);
  masks[4] = simde_mm256_set1_epi8(0x10);
  masks[5] = simde_mm256_set1_epi8(0x20);
  masks[6] = simde_mm256_set1_epi8(0x40);
  masks[7] = simde_mm256_set1_epi8(0x80);

  for (; i_byte < ((block_length >> 5 ) << 5); i_byte += 32) {
    unsigned int i = i_byte >> 5;
    c256 = simde_mm256_and_si256(simde_mm256_cmpeq_epi8(simde_mm256_andnot_si256(simde_mm256_shuffle_epi8(simde_mm256_set1_epi32(((uint32_t*)input[macro_segment])[i]), shufmask),andmask),zero256),masks[0]);
    for (int j=macro_segment+1; j < macro_segment_end; j++) {    
      c256 = simde_mm256_or_si256(simde_mm256_and_si256(simde_mm256_cmpeq_epi8(simde_mm256_andnot_si256(simde_mm256_shuffle_epi8(simde_mm256_set1_epi32(((uint32_t*)input[j])[i]), shufmask),andmask),zero256),masks[j-macro_segment]),c256);
    }
    ((simde__m256i *)cc)[i] = c256;
  }
#endif

#if 0//def __aarch64__
  // s0_0 s1_0 s2_0 ... s31_0 s0_1 ... s31_1 ... s0_3 ... s31_3
  // s0_4 s1_4 s2_4 ....s31_4 s0_5 ... s31_5 ... s0_7 ... s31_7
  simde__m128i shufmask = simde_mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
  simde__m128i andmask  = simde_mm_set1_epi64x(0x0102040810204080);  // every 8 bits -> 8 bytes, pattern repeats.
  simde__m128i zero128   = simde_mm_setzero_si128();
  simde__m128i masks[8];
  register simde__m128i c128;
  masks[0] = simde_mm_set1_epi8(0x1);
  masks[1] = simde_mm_set1_epi8(0x2);
  masks[2] = simde_mm_set1_epi8(0x4);
  masks[3] = simde_mm_set1_epi8(0x8);
  masks[4] = simde_mm_set1_epi8(0x10);
  masks[5] = simde_mm_set1_epi8(0x20);
  masks[6] = simde_mm_set1_epi8(0x40);
  masks[7] = simde_mm_set1_epi8(0x80);

  for (; i_byte < ((block_length >> 4 ) << 4); i_byte += 16) {
    unsigned int i = i_byte >> 4;
    c128 = simde_mm_and_si128(simde_mm_cmpeq_epi8(simde_mm_andnot_si128(simde_mm_shuffle_epi8(simde_mm_set1_epi16(((uint16_t*)input[macro_segment])[i]), shufmask),andmask),zero128),masks[0]);
    for (int j=macro_segment+1; j < macro_segment_end; j++) {    
      c128 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_cmpeq_epi8(simde_mm_andnot_si128(simde_mm_shuffle_epi8(simde_mm_set1_epi32(((uint16_t*)input[j])[i]), shufmask),andmask),zero128),masks[j-macro_segment]),c128);
    }
    ((simde__m128i *)cc)[i] = c128;
  }
#endif

  for (; i_dword < block_length; i_dword++) {
    unsigned int i = i_dword;
    for (int j = macro_segment; j < macro_segment_end; j++) {

      temp = ((input[j][i/8]&((1<<7))>>(i&7)))>>(7-(i&7));
      cc[i] |= (temp << (j-macro_segment));
    }
  }

  if(impp->tinput != NULL) stop_meas(impp->tinput);

  if (BG==1 && Zc==384)  {
    // extend matrix
    if(impp->tprep != NULL) start_meas(impp->tprep);
    if(impp->tprep != NULL) stop_meas(impp->tprep);
    //parity check part

//    printf("calling encode_parity_check_part_cuda cc %p dd %p\n",cc,dd);
    if(impp->tparity != NULL) start_meas(impp->tparity);
    encode_parity_check_part_cuda(cc, dd, BG, Zc, Kb, ncols);
    if(impp->tparity != NULL) stop_meas(impp->tparity);
  }
  else {
	  AssertFatal(1==0,"Only BG1 Zc=384 for now\n");
  }

if(impp->toutput != NULL) start_meas(impp->toutput);
  memcpy(out32,&cc[2*Zc],sizeof(uint32_t)*(block_length-(2*Zc)));
//  printf("cudaMemcpy: dst %p, src %p, length %d, block_length %d, nrows %d, no_punctured_columns\n",
//         &out32[block_length-(2*Zc)],dd,sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit),block_length,nrows,no_punctured_columns);
 // uint32_t dummy[((nrows-no_punctured_columns) * Zc-removed_bit)];
#if USE_UMEM
  err = cudaMemcpy(&out32[block_length-(2*Zc)],dd,sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit),2);
  if (err != cudaSuccess) printf("CUDA Error: %s_3\n", cudaGetErrorString(err)); 			
#else
memcpy(&out32[block_length-(2*Zc)],dd,sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit));
  				
#endif
#if USE_UMEM
cudaFree(dd);
//#else
#endif
  if(impp->toutput != NULL) stop_meas(impp->toutput);
  return 0;
}


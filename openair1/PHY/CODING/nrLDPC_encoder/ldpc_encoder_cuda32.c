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

simde__m128i input32_luta[32][256];
simde__m128i input32_lutb[32][256];
uint32_t *dd0[4];

void init_input32_luts() {

  for (int i=0;i<4;i++) {
    cudaError_t err=cudaMalloc((void**)&dd0[i],46*384*sizeof(uint32_t));
    AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
  }

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
  //Table of possible lifting sizes
  uint8_t temp;

//  printf("input %p output %p\n",input[0],output);


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
  LOG_D(PHY,"ldpc_encoder_cuda32: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",BG,Zc,Kb,block_length,impp->n_segments);
  LOG_D(PHY,"ldpc_encoder_cuda32: PDU (seg 0) %x %x %x %x\n",input[0][0],input[0][1],input[0][2],input[0][3]);
#endif

  AssertFatal(Zc > 0, "no valid Zc found for block length %d\n", block_length);

  int n_inputs = (impp->n_segments/32)+(impp->n_segments&31) > 0 ? 1: 0;
  uint32_t  cc[4][22*Zc]; //padded input, unpacked, max size

//#ifndef USE_UMEM 
  uint32_t *dd[n_inputs];

  for (int i=0;i<n_inputs;i++) {
    //cudaError_t err=cudaMalloc((void**)&dd[i],46*Zc*sizeof(uint32_t));
    //AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
    dd[i]=dd0[i];
  }
//#else
// uint32_t dd[4][46*Zc];
//#endif
  // calculate number of punctured bits
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*rate)/Zc;
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length*rate);
  //printf("%d\n",no_punctured_columns);
  //printf("%d\n",removed_bit);
  // unpack input
  for (int i=0;i<n_inputs;i++) {
    memset(cc[i],0,sizeof(cc[i]));
//#ifndef USE_UMEM 
/*
    cudaError_t err = cudaMemset(dd[i],0,46*Zc*sizeof(uint32_t));
    AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
*/    
//#else
//    memset(dd[i],0,sizeof(dd));
//#endif
  }

  //interleave up to 32 transport-block segements at a time

#if 0
  unsigned int i_dword = 0;
#endif

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

#if 0
  for (; i_dword < block_length; i_dword++) {
    unsigned int i = i_dword;
    for (int j = 0; j < impp->n_segments; j++) {

      temp = (input[j][i/8]&(128>>(i&7)))>>(7-(i&7));
      cc[j>>5][i] |= (temp << (j&31));
    }
  }
#else
  simde__m128i temp128a,temp128b,*ccj;
  int i2=0; 
  uint8_t* inp;
  simde__m128i *luta,*lutb;

  for (int j = 0; j < impp->n_segments; j++) {
    inp = input[j];
    luta=input32_luta[j];
    lutb=input32_lutb[j];    
    ccj=(simde__m128i*)(cc[j>>5]);
    i2=0;
    for (int i=0; i < (block_length>>3); i++,i2+=2) {
       temp128a = luta[inp[i]];
       temp128b = lutb[inp[i]];
       ccj[i2]   = simde_mm_or_si128(ccj[i2],temp128a);
       ccj[i2+1] = simde_mm_or_si128(ccj[i2+1],temp128b);
    }
  }
#endif

  if(impp->tinput != NULL) stop_meas(impp->tinput);

  if (BG==1 && Zc==384)  {
    //parity check part

//    printf("calling encode_parity_check_part_cuda cc %p dd %p\n",cc,dd);
    if(impp->tparity != NULL) start_meas(impp->tparity);
    uint32_t *ccp[n_inputs];/*,*ddp[n_inputs];*/
    for (int s=0;s<n_inputs;s++) {
      ccp[s]=cc[s];
//      ddp[s]=dd[s];
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
//  printf("cudaMemcpy: dst %p, src %p, length %d, block_length %d, nrows %d, no_punctured_columns\n",
//         &out32[block_length-(2*Zc)],dd,sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit),block_length,nrows,no_punctured_columns);
 // uint32_t dummy[((nrows-no_punctured_columns) * Zc-removed_bit)];
//#ifdef USE_UMEM
//    memcpy(&output[s][block_length-(2*Zc)],dd[s],sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit));
//#else
    cudaError_t err = cudaMemcpy(&output[s][block_length-(2*Zc)],dd0[s],sizeof(uint32_t)*((nrows-no_punctured_columns) * Zc-removed_bit),2);
    AssertFatal(err == cudaSuccess, "dd0[%d] %p CUDA Error: %s\n", s, dd0[s],cudaGetErrorString(err)); 							
//#endif
//#ifdef USE_UMEM
//#else
//    cudaFree(dd[s]);
//#endif
  }
  if(impp->toutput != NULL) stop_meas(impp->toutput);
  return 0;
}


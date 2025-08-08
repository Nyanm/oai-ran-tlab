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

#ifdef __AVX2__
simde__m256i input32_lut[32][256];
#else
simde__m128i input32_luta[32][256];
simde__m128i input32_lutb[32][256];
#endif
#ifndef USE_UMEM
uint32_t *cc0[4];
#endif
uint32_t *dd0[4];

void init_input32_luts() {

  // initialize input and output memory
#ifndef USE_UMEM
  for (int i=0;i<4;i++) {
    cudaError_t err=cudaMalloc((void**)&cc0[i],22*384*sizeof(uint32_t));
    AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
  }
#endif
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

  // calculate number of punctured bits
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*rate)/Zc;
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length*rate);
  // clear input
  for (int i=0;i<n_inputs;i++) {
    memset(cc[i],0,sizeof(cc[i]));
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


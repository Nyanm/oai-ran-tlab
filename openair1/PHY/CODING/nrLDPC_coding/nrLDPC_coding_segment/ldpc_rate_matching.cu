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

#include "nr_rate_matching.h"
#include "common/utils/LOG/log.h"

__device__ __forceinline__ int clamp_i16_to_i8(int x)
{
    // x is int (promoted)
    x = (x < -128) ? -128 : x;
    x = (x >  127) ?  127 : x;
    return x;
}

__device__ __forceinline__ uint32_t packs_2x16_to_4x8(uint32_t a, uint32_t b)
{
    int a0 = (int)(int16_t)(a & 0xFFFFu);
    int a1 = (int)(int16_t)(a >> 16);
    int b0 = (int)(int16_t)(b & 0xFFFFu);
    int b1 = (int)(int16_t)(b >> 16);

    uint32_t o0 = (uint8_t)(int8_t)clamp_i16_to_i8(a0);
    uint32_t o1 = (uint8_t)(int8_t)clamp_i16_to_i8(a1);
    uint32_t o2 = (uint8_t)(int8_t)clamp_i16_to_i8(b0);
    uint32_t o3 = (uint8_t)(int8_t)clamp_i16_to_i8(b1);

    return (o0) | (o1 << 8) | (o2 << 16) | (o3 << 24);
}
__global__ void rm(int Ncb,int ind0, int E1, int E2, int r_firstE2, int Foffset, int F, int clear, int seglen, int K, int Z, uint32_t **d, uint32_t **e, uint32_t *llr_buffer) {

     int ind = (int)(blockIdx.x * blockDim.x + threadIdx.x);
     int r = (int)blockIdx.y;

     if (ind>= Foffset && ind < Foffset+F) {
	     llr_buffer[seglen*r + ind-Foffset+K-F] = 0x7f7f7f7f;
//	     if (r==0) printf("writing 0x7f7f7f7f to position %d (ind %d, K %d, Foffset %d, F %d)\n",seglen*r + ind-Foffset+K-F,ind,K,Foffset,F);
	     return;
     }

     if (ind >= Ncb) return;
     if (clear == 1) { d[r][2*ind] = 0; d[r][(2*ind)+1]=0; }

     int E;
     if (r<r_firstE2) E=E1; else E=E2;

     int ind1=ind0,ind2;
     int k=0;

//     if (r==0 && threadIdx.x == 0) printf("check 1a: ind %d ind1 %d Foffset %d\n",ind,ind1,Foffset);
     if (ind1 < Foffset) {
       int ind2 = ind1 + min(Foffset-ind1,E);

//       if (r==0 && threadIdx.x == 0) printf("check 1b: ind %d ind1 %d ind2 %d\n",ind,ind1,ind2);
       if (ind >= ind1 && ind < ind2) {
	       d[r][2*ind] = __vaddss2(d[r][2*ind],e[r][2*(ind-ind1)]);
	       d[r][(2*ind)+1] = __vaddss2(d[r][(2*ind)+1],e[r][2*(ind-ind1)+1]);
       }
//       if (r==0 && threadIdx.x == 0 && ind >= ind1 && ind < ind2) printf("write 1. ind %d, ind1 %d, ind2 %d,k %d/E %d\n",ind,ind1,ind2,ind-ind1,E);
       k=ind2-ind1;
       ind1 = ind2;
     }

//     if (r==0 && threadIdx.x == 0) printf("check 2a: ind %d ind1 %d Foffset %d Foffset+F %d\n",ind,ind1,Foffset,Foffset+F);
     if (ind1 >= Foffset && ind1 < Foffset + F) ind1 = Foffset + F;
     ind2 = ind1 + min(Ncb-ind1,E-k);

//     if (r==0 && threadIdx.x == 0) printf("check 2b: ind %d ind1 %d ind2 %d\n",ind,ind1,ind2);
     if (ind >= ind1 && ind < ind2)  { 
	     d[r][2*ind]     = __vaddss2(d[r][2*ind],e[r][2*(k+(ind-ind1))]);
	     d[r][(2*ind)+1] = __vaddss2(d[r][(2*ind)+1],e[r][2*(k+(ind-ind1))+1]);
     }
//     if (r==0 && threadIdx.x == 0 && ind >= ind1 && ind < ind2) printf("write 2. ind %d, ind1 %d, ind2 %d, k %d/E %d\n",ind,ind1,ind2,k+ind-ind1,E);
     k+=(ind2-ind1);

//     if (r==0 && threadIdx.x == 0) printf("check k %d E %d\n",k,E);
     while (k < E) {
	ind2 = min(Foffset,E-k);
	if (ind < ind2) {
	 	d[r][2*ind]     = __vaddss2(d[r][2*ind],e[r][2*(k+ind)]);
	 	d[r][(2*ind)+1] = __vaddss2(d[r][(2*ind)+1],e[r][2*(k+ind)+1]);
	}
//        if (r==0 && threadIdx.x == 0 && ind < ind2 && ind >= ind1) printf("3. ind %d, ind2 %d, k %d/E %d\n",ind,ind2,k+ind,E);
	k+=ind2;

	ind1=Foffset+F;
	ind2 = ind1 + min(Ncb-ind1,E-k);
	if (ind >= ind1 && ind < ind2 && k < E) {
		d[r][2*ind]     = __vaddss2(d[r][2*ind],e[r][2*(k+ind-ind1)]);
		d[r][(2*ind)+1] = __vaddss2(d[r][(2*ind)+1],e[r][2*(k+ind-ind1)+1]);
	}
//        if (r==0 && threadIdx.x == 0 && ind < ind && ind >= ind1) printf("4. ind %d, ind1 %d, ind2 %d, k %d/E %d\n",ind,ind1,ind2,k+ind-ind1,E);
	k+=(ind2-ind1);
     }
       // note the offset here is such that when ind < Foffset = Kprime - 2Z, the output is put in position r*seglen + (2Z ... Kprime) and when ind > Foffset+F, it is in potiion r*seglent + (Kprime+F = K .. 2Z+(66*Z)=seglen  
     llr_buffer[r*seglen + 2*Z + ind] = packs_2x16_to_4x8(d[r][(2*ind)], d[r][(2*ind)+1]);
//     if (r==0 && threadIdx.x == 0) printf("writing %x to position %d (ind %d)\n",llr_buffer[r*seglen + 2*Z + ind],r*seglen + 2*Z + ind,ind);
}	
 
static const uint8_t index_k0[2][4] = {{0, 17, 33, 56}, {0, 13, 25, 43}};
extern "C" int nr_rate_matching_ldpc_rx_cuda(uint32_t Tbslbrm,
                                             uint8_t BG,
                                             uint16_t Z,
                                             int16_t **d,
                                             int16_t **soft_input,
					     int8_t *llr_buffer,
					     uint32_t K,
                                             uint8_t C,
                                             uint8_t rvidx,
                                             uint8_t clear,
                                             uint32_t E1,
				             uint32_t E2,
				             uint32_t r_firstE2,
                                             uint32_t F,
                                             uint32_t Foffset)
{
  if (C == 0 || C>132) {
    LOG_E(PHY, "nr_rate_matching: invalid parameter C %d\n", C);
    return -1;
  }

  //Bit selection
  uint32_t N = (BG == 1) ? (66 * Z) : (50 * Z);
  uint32_t Ncb;
  if (Tbslbrm == 0)
    Ncb = N;
  else {
    uint32_t Nref = (3 * Tbslbrm / (2 * C)); //R_LBRM = 2/3
    Ncb = min(N, Nref);
  }

  uint32_t ind = (index_k0[BG - 1][rvidx] * Ncb / N) * Z;


  int nthreads=384;
  dim3 nblocks(((Ncb>>2) + nthreads-1)/nthreads,C);
//  printf("rm: Ncb %d, ind %d, E1 %d, E2 %d, Foffset %d, F %d, K %d, Z %d\n",Ncb, ind, E1, E2, Foffset, F,K,Z);
  rm<<<nblocks, nthreads>>>(Ncb/4,ind/4,E1/4,E2/4,r_firstE2,Foffset/4,F/4,clear,68*Z/4,K/4,Z/4,(uint32_t**)d,(uint32_t**)soft_input,(uint32_t*)llr_buffer);
  cudaDeviceSynchronize();
  return(0);
} 



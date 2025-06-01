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

/*!\file ldpc_encode_parity_check.c
 * \brief Parity check function used by ldpc encoders
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
#include <cuda_runtime.h>

static void encode_parity_check_part_cuda(uint32_t *cc, uint32_t *d, short BG,short Zc,short Kb, int ncols)
{
  uint32_t *c=NULL;

  cudaError_t err;
  err = cudaMalloc((void**)&c,2 * 22 * Zc * sizeof(uint32_t));
  if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err)); 							
  for (int i1 = 0; i1 < ncols; i1++)   {
    cudaMemcpy(&c[2 * i1 * Zc], &cc[i1 * Zc], Zc * sizeof(uint32_t),1);
    cudaMemcpy(&c[(2 * i1 + 1) * Zc], &cc[i1 * Zc], Zc * sizeof(uint32_t),1);
  }
  if (BG == 1) {
    switch (Zc) {
      case 176:
      case 192:
      case 208:
      case 224:
      case 240:
      case 256:
      case 288:
      case 320:
      case 352:
	AssertFatal(1==0,"BG %d Zc %d not supported yet for CUDA\n",BG, Zc);
        break;
      case 384:
	ldpc_BG1_Zc384_cuda32(c, d);
        break;
      default:
        AssertFatal(false, "BG %d Zc %d is not supported yet\n", BG, Zc);
    }
  } else if (BG == 2) {
    switch (Zc) {
      case 72:
      case 80:
      case 88:
      case 96:
      case 104:
      case 112:
      case 120:
      case 128:
      case 144:
      case 160:
      case 176:
      case 192:
      case 208:
      case 224:
      case 240:
      case 256:
      case 288:
      case 320:
      case 352:
      case 384:
      default:
        AssertFatal(false , "BG %d Zc %d is not supported yet\n", BG, Zc);
    }
  } else
    AssertFatal(false, "BG %d is not supported\n", BG);
  cudaFree(c);
}




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

/*! \file ldpc_encoder_cuda32.c
 * \brief Defines the optimized LDPC encoder for NVidia GPUs
 * \email openair_tech@eurecom.fr
 * \date 11-30-2025
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

#include "ldpc_generate_coefficient.c"

#include <cuda_runtime.h>

//#define DEBUG_LDPC 1

#include "ldpc_encode_parity_check_cuda.c"
uint32_t *c_dev;
uint32_t **c_host;
uint32_t *c_devh[4];
uint32_t *d_dev;
uint32_t **d_host;
uint32_t *d_devh[4];
uint32_t *input_dev;
uint32_t **input_host;
uint32_t *input_devh[128];
int managed = 0, concurrent = 0, uva = 0, pageable = 0, pageable_uses_host = 0, register_host = 0;

#define USE_GPU_FOR_INPUT 1

int cuda_support_set = 0;

extern cudaStream_t encoderStreams[4];

void cuda_support_init() {

    int dev = 0;
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev);
    cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, dev);
    cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, dev);
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
    cudaDeviceGetAttribute(&pageable_uses_host, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);
    cudaDeviceGetAttribute(&register_host, cudaDevAttrHostRegisterSupported,dev);

    LOG_I(NR_PHY,"Device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    LOG_I(NR_PHY,"Unified Virtual Addressing (UVA): %s\n", uva ? "YES" : "NO");
    LOG_I(NR_PHY,"Managed (Unified) Memory:        %s\n", managed ? "YES" : "NO");
    LOG_I(NR_PHY,"Concurrent managed access:       %s\n", concurrent ? "YES" : "NO");
    LOG_I(NR_PHY,"Pageable memory access:          %s\n", pageable ? "YES" : "NO");
    LOG_I(NR_PHY,"Uses host page tables:           %s\n", pageable_uses_host ? "YES" : "NO");
    LOG_I(NR_PHY,"Host Register supported:         %s\n", register_host ? "YES" : "NO");

  // initialize input and output memory
  // REVISED LOGIC:
  // 1. GH200 has pageable=1. It MUST go to 'else' (Zero-Copy) for Encoder to work.
  // 2. L40S has pageable=0. It MUST go to 'if' (Malloc) to avoid "Illegal Memory Access".
  // 3. We ignore 'register_host' for the decision if 'pageable' is missing, 
  //    because L40S has register_host=1 but fails at Zero-Copy.
  if (!pageable) {
    LOG_I(NR_PHY,"[L40S/Legacy Fix] No Pageable Memory Access detected. Allocating discrete VRAM.\n");
    
    cudaError_t err=cudaMalloc((void **)&c_dev,4*sizeof(uint32_t*));
    AssertFatal(err == cudaSuccess,"CUDA Error (c_dev): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&c_host,4*sizeof(uint32_t*),cudaHostAllocDefault);
    AssertFatal(err == cudaSuccess,"CUDA Error (c_host): %s\n", cudaGetErrorString(err));
    for (int i=0;i<4;i++) {
      err=cudaMalloc((void**)&c_devh[i],2*22*384*sizeof(uint32_t));
      AssertFatal(err == cudaSuccess,"CUDA Error (c_devh[%d]): %s\n", i,cudaGetErrorString(err));
      err=cudaHostAlloc((void**)&c_host[i],2*22*384*sizeof(uint32_t),cudaHostAllocDefault);
      AssertFatal(err == cudaSuccess,"CUDA Error (chost[%d]): %s\n", i,cudaGetErrorString(err));
    }
    err = cudaMemcpy(c_dev,c_devh,4*sizeof(uint32_t*),cudaMemcpyHostToDevice);
    AssertFatal(err == cudaSuccess,"CUDA Error (memcpy c_devh -> c_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void**)&d_dev,4*sizeof(uint32_t*));
    AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&d_host,4*sizeof(uint32_t*),cudaHostAllocDefault);
    AssertFatal(err == cudaSuccess,"CUDA Error (d_host): %s\n", cudaGetErrorString(err));
    for (int i=0;i<4;i++) {
      err=cudaMalloc((void**)&d_devh[i],68*384*sizeof(uint32_t));
      AssertFatal(err == cudaSuccess,"CUDA Error (d_devh[%d]: %s\n", i,cudaGetErrorString(err));
      err=cudaHostAlloc((void**)&d_host[i],68*384*sizeof(uint32_t),cudaHostAllocDefault);
      AssertFatal(err == cudaSuccess,"CUDA Error (d_host[%d]): %s\n", i,cudaGetErrorString(err));
    }
    err=cudaMemcpy(d_dev,d_devh,4*sizeof(uint32_t*),cudaMemcpyHostToDevice);
    AssertFatal(err == cudaSuccess,"CUDA Error (memcpy d_devh -> d_dev): %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void**)&input_dev,128*sizeof(uint8_t*));
    AssertFatal(err == cudaSuccess,"CUDA Error: %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&input_host,128*sizeof(uint8_t*),cudaHostAllocDefault);
    AssertFatal(err == cudaSuccess,"CUDA Error (cc_host): %s\n", cudaGetErrorString(err));
    for (int i=0;i<128;i++) {
      err=cudaMalloc((void**)&input_devh[i],(8448/8)*sizeof(uint8_t));
      AssertFatal(err == cudaSuccess,"CUDA Error (input_devh[%d]: %s\n", i,cudaGetErrorString(err));
      err=cudaHostAlloc((void**)&input_host[i],(8448/8)*sizeof(uint8_t),cudaHostAllocDefault);
      AssertFatal(err == cudaSuccess,"CUDA Error (input_host[%d]): %s\n", i,cudaGetErrorString(err));
    }
    err=cudaMemcpy(input_dev,input_devh,128*sizeof(uint8_t*),cudaMemcpyHostToDevice);
    AssertFatal(err == cudaSuccess,"CUDA Error (memcpy cc_devh -> d_dev): %s\n", cudaGetErrorString(err));
  }
  else {
    LOG_I(NR_PHY,"Allocating c,d,cc arrays for CPU/GPU shared-memory (Zero-Copy Path)\n");
    cudaError_t err=cudaHostAlloc((void **)&c_host,4*sizeof(uint32_t*),cudaHostAllocMapped|cudaHostAllocPortable);
    AssertFatal(err == cudaSuccess,"CUDA Error (c_host): %s\n", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&c_dev, c_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error (c_dev): %s\n", cudaGetErrorString(err));
    LOG_I(NR_PHY,"c_host %p, c_dev %p\n",c_host,c_dev);
    for (int i=0;i<4;i++) {
      err=cudaHostAlloc((void**)&c_host[i],2*22*384*sizeof(uint32_t),cudaHostAllocMapped);
      AssertFatal(err == cudaSuccess,"CUDA Error (c_host[%d]): %s\n", i,cudaGetErrorString(err));
      err = cudaHostGetDevicePointer((void**)&c_devh[i], c_host[i], 0);
      AssertFatal(err == cudaSuccess,"CUDA Error (c_devh[%d]): %s\n", i,cudaGetErrorString(err));
    }
    err=cudaMemcpy(c_dev,c_devh,4*sizeof(uint32_t*),cudaMemcpyHostToDevice);
    AssertFatal(err == cudaSuccess,"CUDA Error (memcpy c_devh -> c_dev): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&d_host,4*sizeof(uint32_t*),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (d_host): %s\n", cudaGetErrorString(err));
    err=cudaHostGetDevicePointer((void**)&d_dev, d_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error cudaHostGetDevicePointer(d_dev): %s\n", cudaGetErrorString(err));
    LOG_I(NR_PHY,"d_host %p, d_dev %p\n",d_host,d_dev);
    for (int i=0;i<4;i++) {
      err=cudaHostAlloc((void**)&d_host[i],68*384*sizeof(uint32_t),cudaHostAllocMapped);
      AssertFatal(err == cudaSuccess,"CUDA Error (d_host[%d]): %s\n", i,cudaGetErrorString(err));
      err=cudaHostGetDevicePointer((void**)&d_devh[i], d_host[i], 0);
      AssertFatal(err == cudaSuccess,"CUDA Error (cudaHostGetDevicePointer) d_devh[%d]: %s\n", i,cudaGetErrorString(err));
    }
    err=cudaMemcpy(d_dev,d_devh,4*sizeof(uint32_t*),cudaMemcpyHostToDevice);
    AssertFatal(err == cudaSuccess,"CUDA Error (memcpy d_devh -> d_dev): %s\n", cudaGetErrorString(err));
    err=cudaHostAlloc((void **)&input_host,128*sizeof(uint8_t*),cudaHostAllocMapped);
    AssertFatal(err == cudaSuccess,"CUDA Error (input_host): %s\n", cudaGetErrorString(err));
    err=cudaHostGetDevicePointer((void**)&input_dev, input_host, 0);
    AssertFatal(err == cudaSuccess,"CUDA Error cudaHostGetDevicePointer(cc_host): %s\n", cudaGetErrorString(err));
    LOG_I(NR_PHY,"input_host %p, input_dev %p\n",input_host,input_dev);
  }


  cuda_support_set=1;
}

uint32_t **LDPCencoder32(uint8_t **input, encoder_implemparams_t *impp)
{
  //set_log(PHY, 4);

  int Zc = impp->Zc;
  int Kb = impp->Kb;
  short block_length = impp->K;
  short BG = impp->BG;
  int nrows=46,ncols=22;
  int rate=3;
  int no_punctured_columns,removed_bit;

  int encoder_stream=0;

  AssertFatal(BG==1,"BG %d is not supported for CUDA version\n",BG);
  AssertFatal(Zc==384,"Zc %d is not supported for CUDA version \n", Zc);
 
  if(impp->tinput != NULL) start_meas(impp->tinput);

#ifdef DEBUG_LDPC
  LOG_I(PHY,"ldpc_encoder_cuda32: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",BG,Zc,Kb,block_length,impp->n_segments);
  LOG_I(PHY,"ldpc_encoder_cuda32: PDU (seg 0) %x %x %x %x\n",input[0][0],input[0][1],input[0][2],input[0][3]);
#endif

  int n_inputs = (impp->n_segments/32)+(((impp->n_segments&31) > 0) ? 1: 0);
//  uint32_t  cc[4][22*Zc]; //padded input, unpacked, max size

  // calculate number of punctured bits
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*rate)/Zc;
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length*rate);
#ifdef USE_GPU_FOR_INPUT
  if (!pageable && !register_host) {
    for (int r=0;r<impp->n_segments;r++) {
        cudaMemcpy(input_devh[r],input[r],block_length>>3,cudaMemcpyHostToDevice);
    }
  }
  ldpc_input(pageable||register_host? input : input_dev,(uint32_t**)c_dev,impp->n_segments,&encoderStreams[encoder_stream]);
#else 
  ldpc_input32(input,(uint32_t**)c_dev,n_inputs,block_length,impp->n_segments); 
#endif
  if(impp->tinput != NULL) stop_meas(impp->tinput);
  //parity check part
  if(impp->tparity != NULL) start_meas(impp->tparity);
  encode_parity_check_part_cuda((uint32_t**)c_dev, (uint32_t**)d_dev, BG, Zc, Kb, ncols,n_inputs,&encoderStreams[encoder_stream]);
  if(impp->tparity != NULL) stop_meas(impp->tparity);
  
  return d_host;
}


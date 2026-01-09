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
/*! \file nrLDPC_CUDA_shared_param.h
 * \brief Shared parameters in CUDA implementation of LDPC decoder 
 * \author Qizhi Pan, Raymond Knopp
 * \company EURECOM
 * \email: qizhi.pan@eurecom.fr, raymond.knopp@eurecom.fr
 * \date 2025-12-30
 * \version 1.0
 * \note 
 * \warning
 */

#pragma once
#include <cuda_runtime.h>
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#ifdef __cplusplus
extern "C" {
#endif

#define num_TotalThreads_BG1_R13 30336
#define num_TotalThreads_BG1_R23 13824
#define num_TotalThreads_llr_llrRes 6528
#define RowLength 96 //Zc = 384/4 = 96

#define num_TotalBlocks_BG1_R13 316
#define num_TotalBlocks_BG1_R23 144
#define num_TotalBlocks_llr_llrRes 68


extern cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL];
extern cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL];
extern bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL];

#ifdef __cplusplus
}
#endif

typedef struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
}KernelLaunchConfig;


typedef struct {
    int idxBn;
    int idxCn;
    int preBuf;
    int circShift;
    int8_t dd;
} DumpEntry;

typedef struct {
    int8_t* p_llr_ptr;     
    int8_t* p_out_ptr;      
} ldpc_cuda_bridge_t;
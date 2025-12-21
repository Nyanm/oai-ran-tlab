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

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
 * \brief Top-level routines for decoding LDPC transport channels
 */

// [from gNB coding]
#include "nr_rate_matching.h"
#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "defs.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"

#include <stdalign.h>
#include <stdint.h>
#include <syscall.h>
#include <time.h>
// #define gNB_DEBUG_TRACE

#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000 // 26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX = 68*384
// #define DEBUG_CRC
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#ifdef DEBUG_CRC
#define PRINT_CRC_CHECK(a) a
#else
#define PRINT_CRC_CHECK(a)
#endif

#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"

#include <cuda_runtime.h>

//-------------------------Debug Function-----------------------
void dumpAssUltra(int8_t* cnProcBufRes, const char* filename)
{
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < 3 * 27000; i++) { //only dump the first 3 segments
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}

void dumpAssUltraInput(int8_t* cnProcBufRes, const char* filename)
{
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < 3 * 68 * 384; i++) { //only dump the first 3 segments
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}


static int8_t *g_llrBuffer = NULL;
static int8_t *g_decodedBitsBig = NULL;

#define MAX_LDPC_LLR_SIZE (OAI_LDPC_DECODER_MAX_NUM_LLR * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4)
#define MAX_LDPC_OUT_SIZE (8448 * MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER * 4) // BG1 K*Z_max

void nr_process_decode_segment_cuda(nrLDPC_TB_decoding_parameters_t *segs)
{
  // arg points to RDATA array (nrLDPC_decoding_parameters_t *RDATA)
  DevAssert(segs != NULL);

  // use seg0 as canonical
  const int C = segs->C;
  const int Z = segs->Z;
  const int Kc = segs->BG == 2 ? 52 : 68;
  const int K  = segs->K;
  const int Kprime = K - segs->F;
  const int segLen = Kc*Z; // int16 length; after packing we store int8 [segLen]
  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;
  // allocate big buffers on heap
  //the big buffer is now globally declared
  int8_t *llrBuffer = g_llrBuffer;
  int8_t *decodedBitsBig = g_decodedBitsBig;
  ///int8_t *llrBuffer = (int8_t*)alloca((size_t)C * OAI_LDPC_DECODER_MAX_NUM_LLR * sizeof(int8_t));
  ///if (!llrBuffer) { LOG_E(PHY,"alloc llrBuffer failed\n"); return; }

  ///int8_t *decodedBitsBig = (int8_t*)alloca(C * K * sizeof(int8_t));
  ///if (!decodedBitsBig) { LOG_E(PHY,"alloc decodedBitsBig failed\n"); return; }


  // Phase 1: per-segment deinterleave+rate-match and pack into llrBuffer
  // prepare int16 z (local)
  int16_t *z_local = (int16_t*)alloca(sizeof(int16_t) * segLen); // segLen is safe small
//  printf("Qm : %d\n",segs->Qm);								 
  for (int r = 0; r < C; ++r) {
    // deinterleave
    start_meas(&segs->segments[0].ts_deinterleave);
    int16_t *harq_e = (int16_t*)alloca(sizeof(int16_t) * segs->segments[r].E);
//    for (int i=0;i<segs->segments[r].E;i++) printf("llr_in[%d] %d\n",i,segs->segments[r].llr[i]);
    nr_deinterleaving_ldpc(segs->segments[r].E, segs->Qm, harq_e, segs->segments[r].llr);
//    for (int i=0;i<16;i++) printf("harq_e[%d] %d\n",i,harq_e[i]);
    stop_meas(&segs->segments[0].ts_deinterleave);
    // rate matching
    start_meas(&segs->segments[0].ts_rate_unmatch);
    if (nr_rate_matching_ldpc_rx(segs->tbslbrm,
                                 segs->BG,
                                 Z,
                                 segs->segments[r].d,
                                 harq_e,
                                 C,             // TB segments count
                                 segs->rv_index,
                                 *segs->segments[r].d_to_be_cleared,
                                 segs->segments[r].E,
                                 segs->F,
                                 Kprime - 2 * Z) == -1) {
      stop_meas(&segs->segments[0].ts_rate_unmatch);
      LOG_E(PHY,"rate matching failed seg %d\n", r);
      memset(segs->segments[r].c, 0, K);
      //*rdata->decodeSuccess = false;
      continue; // skip this segment
    }
    stop_meas(&segs->segments[0].ts_rate_unmatch);
    start_meas(&segs->segments[0].ts_seg_prep);
    *segs->segments[r].d_to_be_cleared = false;

    memset(z_local,0,sizeof(int16_t)*2*Z);
    memset(z_local + Kprime,127,sizeof(int16_t)*segs->F);
    memcpy(z_local + 2*Z, segs->segments[r].d, (size_t)(Kprime - 2*Z)*sizeof(int16_t));
    memcpy(z_local + K, segs->segments[r].d + (K - 2*Z), (size_t)(Kc*Z - K)*sizeof(int16_t));

    // pack int16 -> int8 into llrBuffer[r * segLen]
    simde__m128i *pv = (simde__m128i*)z_local;
    simde__m128i *pl = (simde__m128i*)(llrBuffer + (size_t)r*segLen);
    int vecCount = ((Kc * Z) >> 4);
    for (int j=0, idx=0; j<vecCount; ++j, idx+=2) {
      pl[j] = simde_mm_packs_epi16(pv[idx], pv[idx+1]);
    }
    stop_meas(&segs->segments[0].ts_seg_prep);
//    for (int i=0;i<(vecCount<<4);i++) printf("channel llr %d : %d\n",i,((int8_t*)pl)[i]);
  }

  start_meas(&segs->segments[0].ts_ldpc_decode);
  
  t_nrLDPC_dec_params decParams = {.check_crc = check_crc};
  decParams.Z = Z;
  decParams.R = segs->segments[0].R;
  decParams.BG = segs->BG;
  decParams.crc_type = crcType(C, segs->A);
  decParams.Kprime = lenWithCrc(C, segs->A);
  decParams.n_segments = C;
  decParams.outMode=nrLDPC_outMode_BIT;
  decParams.numMaxIter = segs->max_ldpc_iterations;
  // Phase 2: call batch GPU decoder (you must implement this API)
  int decodeIterations = LDPCdecoder_cuda(&decParams, llrBuffer, decodedBitsBig, p_procTime, segs->abort_decode);
  stop_meas(&segs->segments[0].ts_ldpc_decode);
//  dumpAssUltraInput(llrBuffer, "dlsim_decoder_input_cuda_GH.txt");
//  dumpAssUltra(decodedBitsBig, "dlsim_decoder_output_cuda_GH.txt");
  //dumpASS(decodedBitsBig, "dlsim_decoded_bits.txt");
  if (decodeIterations > segs->max_ldpc_iterations) {
    LOG_E(PHY,"LDPCdecoder_cuda_batch failed\n");
    // mark failures
    for (int r=0;r<C;r++) { memset(segs->segments[r].c,0,K>>3); segs->segments[r].decodeSuccess=false; }
    stop_meas(&segs->segments[0].ts_ldpc_decode);
    printf("Decoder failed\n");
    //free(iterUsed); 
    //free(decodedBitsBig); free(llrBuffer);
    return;
  }


  // Phase 3: scatter results and set decodeSuccess
  for (int r=0; r<C; ++r) {
    // check segment CRC here

    if (decodeIterations <= segs->max_ldpc_iterations) {
      memcpy(segs->segments[r].c, decodedBitsBig + (size_t)r*K, K>>3);
      segs->segments[r].decodeSuccess = true;
  //    for (int i=0;i<(K>>3);i++) printf("byte %d %x\n",i,segs->segments[r].c[i]);
    } else {
      memset(segs->segments[r].c, 0, K>>3);
      segs->segments[r].decodeSuccess = false;
    }
  }

  //free(iterUsed);
//  printf("decodedBitsBig %p, llrBuffer %p\n",decodedBitsBig,llrBuffer);
//  if (decodedBitsBig) free(decodedBitsBig);
//  if (llrBuffer) free(llrBuffer);
}


int32_t nrLDPC_coding_init_cuda(void)
{
  cuda_support_init();

  if (g_llrBuffer == NULL) {
      cudaMallocManaged((void**)&g_llrBuffer, MAX_LDPC_LLR_SIZE, cudaMemAttachGlobal);
      cudaMallocManaged((void**)&g_decodedBitsBig, MAX_LDPC_OUT_SIZE, cudaMemAttachGlobal);
  }

  LDPCinit_cuda(g_llrBuffer, g_decodedBitsBig);
  return 0;
}

int32_t nrLDPC_coding_shutdown_cuda(void)
{
  LDPCshutdown_cuda();
  return 0;
}


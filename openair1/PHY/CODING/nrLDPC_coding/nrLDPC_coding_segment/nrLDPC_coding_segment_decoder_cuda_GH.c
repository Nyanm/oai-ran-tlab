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
#include "SCHED_NR/fapi_nr_l1.h"
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

/**
 * \typedef nrLDPC_decoding_parameters_t
 * \struct nrLDPC_decoding_parameters_s
 * \brief decoding parameter of transport blocks
 * \var decoderParms decoder parameters
 * \var Qm modulation order
 * \var Kc ratio between the number of columns in the parity check matrix and the lifting size
 * it is fixed for a given base graph while the lifting size is chosen to have a sufficient number of columns
 * \var rv_index
 * \var max_number_iterations maximum number of LDPC iterations
 * \var abort_decode pointer to decode abort flag
 * \var tbslbrm transport block size LBRM in bytes
 * \var A Transport block size (This is A from 38.212 V15.4.0 section 5.1)
 * \var K Code block size at decoder output
 * \var Z lifting size
 * \var F filler bits size
 * \var r segment index in TB
 * \var E input llr segment size
 * \var C number of segments
 * \var llr input llr segment array
 * \var d Pointers to code blocks before LDPC decoding (38.212 V15.4.0 section 5.3.2)
 * \var d_to_be_cleared
 * pointer to the flag used to clear d properly
 * when true, clear d after rate dematching
 * \var c Pointers to code blocks after LDPC decoding (38.212 V15.4.0 section 5.2.2)
 * \var decodeSuccess pointer to the flag indicating that the decoding of the segment was successful
 * \var ans pointer to task answer used by the thread pool to detect task completion
 * \var p_ts_deinterleave pointer to deinterleaving time stats
 * \var p_ts_rate_unmatch pointer to rate unmatching time stats
 * \var p_ts_ldpc_decode pointer to decoding time stats
 */
//-------------------------Debug Function-----------------------
typedef struct nrLDPC_decoding_parameters_s {
  t_nrLDPC_dec_params decoderParms;

  uint8_t Qm;

  uint8_t Kc;
  uint8_t rv_index;
  decode_abort_t *abort_decode;

  uint32_t tbslbrm;
  uint32_t A;
  uint32_t K;
  uint32_t Z;
  uint32_t F;

  uint32_t C;

  int E;
  short *llr;
  int16_t *d;
  bool *d_to_be_cleared;
  uint8_t *c;
  bool *decodeSuccess;

  task_ans_t *ans;

  time_stats_t *p_ts_deinterleave;
  time_stats_t *p_ts_rate_unmatch;
  time_stats_t *p_ts_ldpc_decode;
} nrLDPC_decoding_parameters_t;

static void nr_process_decode_segment(void *arg)
{
  nrLDPC_decoding_parameters_t *rdata = (nrLDPC_decoding_parameters_t *)arg;
  t_nrLDPC_dec_params *p_decoderParms = &rdata->decoderParms;
  const int K = rdata->K;
  const int Kprime = K - rdata->F;
  const int A = rdata->A;
  const int E = rdata->E;
  const int Qm = rdata->Qm;
  const int rv_index = rdata->rv_index;
  const uint8_t Kc = rdata->Kc;
  short *ulsch_llr = rdata->llr;
  int8_t llrProcBuf[OAI_LDPC_DECODER_MAX_NUM_LLR] __attribute__((aligned(32)));

  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;

  ////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// nr_deinterleaving_ldpc ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////// ulsch_llr =====> ulsch_harq->e //////////////////////////////

  start_meas(rdata->p_ts_deinterleave);

  /// code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
  int16_t harq_e[E];

  nr_deinterleaving_ldpc(E, Qm, harq_e, ulsch_llr);

  //////////////////////////////////////////////////////////////////////////////////////////

  stop_meas(rdata->p_ts_deinterleave);

  start_meas(rdata->p_ts_rate_unmatch);

  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// nr_rate_matching_ldpc_rx ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////// ulsch_harq->e =====> ulsch_harq->d /////////////////////////

  if (nr_rate_matching_ldpc_rx(rdata->tbslbrm,
                               p_decoderParms->BG,
                               p_decoderParms->Z,
                               rdata->d,
                               harq_e,
                               rdata->C,
                               rv_index,
                               *rdata->d_to_be_cleared,
                               E,
                               rdata->F,
                               K - rdata->F - 2 * (p_decoderParms->Z))
      == -1) {
    stop_meas(rdata->p_ts_rate_unmatch);
    LOG_E(PHY, "nrLDPC_coding_segment_decoder.c: Problem in rate_matching\n");

    // Task completed
    completed_task_ans(rdata->ans);
    return;
  }
  stop_meas(rdata->p_ts_rate_unmatch);

  *rdata->d_to_be_cleared = false;

  p_decoderParms->crc_type = crcType(rdata->C, A);
  p_decoderParms->Kprime = lenWithCrc(rdata->C, A);
  p_decoderParms->n_segments = rdata->C;

  // set first 2*Z_c bits to zeros

  int16_t z[68 * 384 + 16] __attribute__((aligned(16)));

  start_meas(rdata->p_ts_ldpc_decode);

  memset(z, 0, 2 * rdata->Z * sizeof(*z));
  // set Filler bits
  memset(z + Kprime, 127, rdata->F * sizeof(*z));
  // Move coded bits before filler bits
  memcpy(z + 2 * rdata->Z, rdata->d, (Kprime - 2 * rdata->Z) * sizeof(*z));
  // skip filler bits
  memcpy(z + K, rdata->d + (K - 2 * rdata->Z), (Kc * rdata->Z - K) * sizeof(*z));
  // Saturate coded bits before decoding into 8 bits values
  simde__m128i *pv = (simde__m128i *)&z;
  int8_t l[68 * 384 + 16] __attribute__((aligned(16)));
  simde__m128i *pl = (simde__m128i *)&l;
  for (int i = 0, j = 0; j < ((Kc * rdata->Z) >> 4) + 1; i += 2, j++) {
    pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
  }
  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// nrLDPC_decoder /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////// pl =====> llrProcBuf //////////////////////////////////
  int decodeIterations =  
    LDPCdecoder(p_decoderParms, 0, 0, 0, l, llrProcBuf, p_procTime, rdata->abort_decode);
  if (decodeIterations < p_decoderParms->numMaxIter) {
    memcpy(rdata->c, llrProcBuf, K >> 3);
    *rdata->decodeSuccess = true;
  } else {
    memset(rdata->c, 0, K >> 3);
    *rdata->decodeSuccess = false;
  }
  stop_meas(rdata->p_ts_ldpc_decode);

  // Task completed
  completed_task_ans(rdata->ans);
}

static void nr_process_decode_segment_cuda(void *arg)
{
  // arg points to RDATA array (nrLDPC_decoding_parameters_t *RDATA)
  nrLDPC_decoding_parameters_t *RDATA = (nrLDPC_decoding_parameters_t *)arg;
  DevAssert(RDATA != NULL);

  // use seg0 as canonical
  nrLDPC_decoding_parameters_t *seg0 = &RDATA[0];
  const int C = seg0->C;
  const int Z = seg0->Z;
  const int Kc = seg0->Kc;
  const int K  = seg0->K;
  const int bytesPerSeg = K >> 3;
  const int segLen = Kc * Z + 16; // int16 length; after packing we store int8 [segLen]
  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;
  // allocate big buffers on heap
  int8_t *llrBuffer = (int8_t*)aligned_alloc(16, (size_t)MAX_NUM_DLSCH_SEGMENTS_DL * segLen * sizeof(int8_t));
  if (!llrBuffer) { LOG_E(PHY,"alloc llrBuffer failed\n"); return; }

  int8_t *decodedBitsBig = (int8_t*)aligned_alloc(16, MAX_NUM_DLSCH_SEGMENTS_DL * K * sizeof(int8_t));
  if (!decodedBitsBig) { free(llrBuffer); LOG_E(PHY,"alloc decodedBitsBig failed\n"); return; }

//  int *iterUsed = (int*)calloc(C, sizeof(int));
  //if (!iterUsed) { free(llrBuffer); free(decodedBitsBig); LOG_E(PHY,"alloc iterUsed failed\n"); return; }

  // Phase 1: per-segment deinterleave+rate-match and pack into llrBuffer
  for (int r = 0; r < C; ++r) {
    nrLDPC_decoding_parameters_t *rdata = &RDATA[r];
    // deinterleave
     start_meas(rdata->p_ts_deinterleave);
    int16_t *harq_e = (int16_t*)alloca(sizeof(int16_t) * rdata->E);

    nr_deinterleaving_ldpc(rdata->E, rdata->Qm, harq_e, rdata->llr);
  stop_meas(rdata->p_ts_deinterleave);
    // rate matching
    start_meas(rdata->p_ts_rate_unmatch);
    if (nr_rate_matching_ldpc_rx(rdata->tbslbrm,
                                 rdata->decoderParms.BG,
                                 rdata->Z,
                                 rdata->d,
                                 harq_e,
                                 rdata->C,             // TB segments count
                                 rdata->rv_index,
                                 *rdata->d_to_be_cleared,
                                 rdata->E,
                                 rdata->F,
                                 rdata->K - rdata->F - 2 * (rdata->decoderParms.Z)) == -1) {
      stop_meas(rdata->p_ts_rate_unmatch);
      LOG_E(PHY,"rate matching failed seg %d\n", r);
      memset(rdata->c, 0, bytesPerSeg);
      //*rdata->decodeSuccess = false;
      completed_task_ans(rdata->ans);
      continue; // skip this segment
    }
    stop_meas(rdata->p_ts_rate_unmatch);
    *rdata->d_to_be_cleared = false;
    start_meas(rdata->p_ts_ldpc_decode);

    // prepare int16 z (local)
    int16_t *z_local = (int16_t*)alloca(sizeof(int16_t) * segLen); // segLen is safe small
    // zero and fill z_local as before (use loops for clarity)
    for (int i=0;i<segLen;i++) z_local[i]=0;
    if (K - rdata->F > 2 * rdata->Z) {
      memcpy(z_local + 2*rdata->Z, rdata->d, (size_t)(rdata->K - rdata->F - 2*rdata->Z)*sizeof(int16_t));
    }
    for (int i=0;i<rdata->F;i++) z_local[rdata->K - rdata->F + i] = 127; // filler
    if (Kc * Z > K) {
      memcpy(z_local + K, rdata->d + (K - 2*rdata->Z), (size_t)(Kc*Z - K)*sizeof(int16_t));
    }

    // pack int16 -> int8 into llrBuffer[r * segLen]
    simde__m128i *pv = (simde__m128i*)z_local;
    simde__m128i *pl = (simde__m128i*)(llrBuffer + (size_t)r*segLen);
    int vecCount = ((Kc * Z) >> 4) + 1;
    for (int j=0, idx=0; j<vecCount; ++j, idx+=2) {
      pl[j] = simde_mm_packs_epi16(pv[idx], pv[idx+1]);
    }
    stop_meas(rdata->p_ts_ldpc_decode);
  }
  start_meas(RDATA->p_ts_ldpc_decode);
  (&RDATA->decoderParms)->crc_type = crcType(RDATA->C, RDATA->A);
  (&RDATA->decoderParms)->Kprime = lenWithCrc(RDATA->C, RDATA->A);
  (&RDATA->decoderParms)->n_segments = RDATA->C;
  // Phase 2: call batch GPU decoder (you must implement this API)
  int decodeIterations = LDPCdecoder(&RDATA->decoderParms, 0, 0, 0, llrBuffer, decodedBitsBig, p_procTime, RDATA->abort_decode);
  //printf("Decoder done\n");
  //dumpASS(decodedBitsBig, "dlsim_decoded_bits.txt");
  if (decodeIterations > (&seg0->decoderParms)->numMaxIter) {
    LOG_E(PHY,"LDPCdecoder_cuda_batch failed\n");
    // mark failures
    for (int r=0;r<C;r++) { memset(RDATA[r].c,0,bytesPerSeg); *RDATA[r].decodeSuccess=false; completed_task_ans(RDATA[r].ans); }
    stop_meas(RDATA->p_ts_ldpc_decode);
    printf("Decoder failed\n");
    //free(iterUsed); 
    //free(decodedBitsBig); free(llrBuffer);
    return;
  }
  stop_meas(RDATA->p_ts_ldpc_decode);

  // Phase 3: scatter results and set decodeSuccess
  for (int r=0; r<C; ++r) {
    nrLDPC_decoding_parameters_t *rdata = &RDATA[r];
    if (decodeIterations <= rdata->decoderParms.numMaxIter) {
      memcpy(rdata->c, decodedBitsBig + (size_t)r*bytesPerSeg, bytesPerSeg);
      *rdata->decodeSuccess = true;
    } else {
      memset(rdata->c, 0, bytesPerSeg);
      *rdata->decodeSuccess = false;
    }
    completed_task_ans(rdata->ans);
  }

  //free(iterUsed);
  free(decodedBitsBig);
  free(llrBuffer);
}



int nrLDPC_prepare_TB_decoding(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters,
                               int pusch_id,
                               thread_info_tm_t *t_info)
{
    nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];

    *nrLDPC_TB_decoding_parameters->processedSegments = 0;
    t_nrLDPC_dec_params decParams = {.check_crc = check_crc};
    decParams.BG = nrLDPC_TB_decoding_parameters->BG;
    decParams.Z = nrLDPC_TB_decoding_parameters->Z;
    decParams.numMaxIter = nrLDPC_TB_decoding_parameters->max_ldpc_iterations;
    decParams.outMode = 0;

    // Create a large RDATA array to store all the segment information
    // The size of RDATA will be equal to the number of segments (C)
    nrLDPC_decoding_parameters_t *RDATA = calloc(nrLDPC_TB_decoding_parameters->C, sizeof(nrLDPC_decoding_parameters_t));
    if (!RDATA) {
      perror("calloc RDATA");
      return -1;
    }
    // Fill the RDATA array with segment-specific parameters
    for (int r = 0; r < nrLDPC_TB_decoding_parameters->C; r++) {
        nrLDPC_decoding_parameters_t *rdata = &RDATA[r];
        rdata->ans = t_info->ans;

        // Set the parameters for each segment
        decParams.R = nrLDPC_TB_decoding_parameters->segments[r].R;
        rdata->decoderParms = decParams;
        rdata->llr = nrLDPC_TB_decoding_parameters->segments[r].llr;
        rdata->Kc = decParams.BG == 2 ? 52 : 68;  // Set Kc based on BG
        rdata->C = nrLDPC_TB_decoding_parameters->C;
        rdata->E = nrLDPC_TB_decoding_parameters->segments[r].E;
        rdata->A = nrLDPC_TB_decoding_parameters->A;
        rdata->Qm = nrLDPC_TB_decoding_parameters->Qm;
        rdata->K = nrLDPC_TB_decoding_parameters->K;
        rdata->Z = nrLDPC_TB_decoding_parameters->Z;
        rdata->F = nrLDPC_TB_decoding_parameters->F;
        rdata->rv_index = nrLDPC_TB_decoding_parameters->rv_index;
        rdata->tbslbrm = nrLDPC_TB_decoding_parameters->tbslbrm;
        rdata->abort_decode = nrLDPC_TB_decoding_parameters->abort_decode;
        rdata->d = nrLDPC_TB_decoding_parameters->segments[r].d;
        rdata->d_to_be_cleared = nrLDPC_TB_decoding_parameters->segments[r].d_to_be_cleared;
        rdata->c = nrLDPC_TB_decoding_parameters->segments[r].c;
        rdata->decodeSuccess = &nrLDPC_TB_decoding_parameters->segments[r].decodeSuccess;
        rdata->p_ts_deinterleave = &nrLDPC_TB_decoding_parameters->segments[r].ts_deinterleave;
        rdata->p_ts_rate_unmatch = &nrLDPC_TB_decoding_parameters->segments[r].ts_rate_unmatch;
        rdata->p_ts_ldpc_decode = &nrLDPC_TB_decoding_parameters->segments[r].ts_ldpc_decode;

        // The segment's parameters are now stored in the RDATA array
    }

    // Launch the CUDA kernel for parallel processing of all segments
    // You will need to adjust this part to pass the RDATA array to your CUDA kernel
    // The kernel can then process each segment independently using CUDA's parallel execution model
    nr_process_decode_segment_cuda(RDATA);
    // Log the total number of segments being processed
    LOG_D(PHY, "Added all blocks to decode, in total: %d segments\n", nrLDPC_TB_decoding_parameters->C);

    // Cleanup the RDATA array after use
    free(RDATA);

    return nrLDPC_TB_decoding_parameters->C;
}



int32_t nrLDPC_coding_init(void)
{
  cuda_support_init();
  LDPCinit_cuda();
  return 0;
}

int32_t nrLDPC_coding_shutdown(void)
{
  LDPCshutdown_cuda();
  return 0;
}

int32_t nrLDPC_coding_decoder(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  int nbSegments = 0;
  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];
    nbSegments += nrLDPC_TB_decoding_parameters->C;
  }
  nrLDPC_decoding_parameters_t arr[nbSegments];
  task_ans_t ans;
  init_task_ans(&ans, nbSegments);
  thread_info_tm_t t_info = {.buf = (uint8_t *)arr, .len = 0, .cap = nbSegments, .ans = &ans};

  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    (void)nrLDPC_prepare_TB_decoding(nrLDPC_slot_decoding_parameters, pusch_id, &t_info);
  }

  // Execute thread pool tasks
  join_task_ans(t_info.ans);

  for (int pusch_id = 0; pusch_id < nrLDPC_slot_decoding_parameters->nb_TBs; pusch_id++) {
    nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];
    for (int r = 0; r < nrLDPC_TB_decoding_parameters->C; r++) {
      if (nrLDPC_TB_decoding_parameters->segments[r].decodeSuccess) {
        *nrLDPC_TB_decoding_parameters->processedSegments = *nrLDPC_TB_decoding_parameters->processedSegments + 1;
      }
    }
  }
  return 0;
}

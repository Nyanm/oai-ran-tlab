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

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_armral/nrLDPC_coding_armral_decoder.c
 * \brief Top-level routines for decoding LDPC transport channels using the Arm RAN Acceleration Library
 * \author Romain Beurdouche
 * \date 2025
 * \company EURECOM
 * \email romain.beurdouche@eurecom.fr
 * \note ArmRAL available at https://git.gitlab.arm.com/networking/ral.git
 * \warning
 */

// [from gNB coding]
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
#include <armral.h>
// #define gNB_DEBUG_TRACE

#define OAI_LDPC_DECODER_MAX_NUM_LLR 27000 // 26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX = 68*384
// #define DEBUG_CRC
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
 * \var A Transport block size (This is A from 38.212 V15.4.0 section 5.1)
 * \var K Code block size at decoder output
 * \var Z lifting size
 * \var F filler bits size
 * \var C number of segments
 * \var E input llr segment size
 * \var BG Base graph index (BG1: 1, BG2: 2)
 * \var max_number_iterations maximum number of LDPC iterations
 * \var tbslbrm transport block size LBRM in bytes
 * \var Qm modulation order
 * \var rv_index
 * \var llr input llr segment array
 * \var d Pointers to code blocks before LDPC decoding (38.212 V15.4.0 section 5.3.2)
 * \var d_to_be_cleared
 * pointer to the flag used to clear d properly
 * when true, clear d after rate dematching
 * \var c Pointers to code blocks after LDPC decoding (38.212 V15.4.0 section 5.2.2)
 * \var decodeSuccess pointer to the flag indicating that the decoding of the segment was successful
 * \var ans pointer to task answer used by the thread pool to detect task completion
 * \var abort_decode pointer to decode abort flag
 * \var p_ts_rate_unmatch pointer to rate unmatching time stats
 * \var p_ts_ldpc_decode pointer to decoding time stats
 */
typedef struct nrLDPC_decoding_parameters_s {
  uint32_t A;
  uint32_t K;
  uint32_t Z;
  uint32_t F;

  uint32_t C;
  uint32_t E;

  uint8_t BG;
  uint32_t max_number_iterations;

  uint32_t tbslbrm;
  uint32_t Qm;
  uint8_t rv_index;

  short *llr;
  int16_t *d;
  bool *d_to_be_cleared;
  uint8_t *c;
  bool *decodeSuccess;

  task_ans_t *ans;
  // decode_abort_t *abort_decode;

  time_stats_t *p_ts_rate_unmatch;
  time_stats_t *p_ts_ldpc_decode;
} nrLDPC_decoding_parameters_t;

static void nr_process_decode_segment(void *arg)
{
  nrLDPC_decoding_parameters_t *rdata = (nrLDPC_decoding_parameters_t *)arg;
  const uint32_t A = rdata->A;
  const uint32_t K = rdata->K;
  const uint32_t Z = rdata->Z;
  const uint32_t F = rdata->F;
  const uint32_t C = rdata->C;
  const uint32_t Kprime = K - rdata->F;
  const uint32_t E = rdata->E;
  const uint32_t max_number_iterations = rdata->max_number_iterations;
  const uint32_t Qm = rdata->Qm;
  const uint32_t rv_index = rdata->rv_index;
  short *ulsch_llr = rdata->llr;
  uint8_t llrProcBuf[OAI_LDPC_DECODER_MAX_NUM_LLR] __attribute__((aligned(32)));

  start_meas(rdata->p_ts_rate_unmatch);

  armral_ldpc_graph_t armral_bg = rdata->BG == 2 ? LDPC_BASE_GRAPH_2 : LDPC_BASE_GRAPH_1;

  uint32_t Nref = 3 * rdata->tbslbrm / (2 * C); // R_LBRM = 2/3

  armral_modulation_type armral_mod = ARMRAL_MOD_QPSK;
  switch (Qm) {
    case 2:
      armral_mod = ARMRAL_MOD_QPSK;
      break;
    case 4:
      armral_mod = ARMRAL_MOD_16QAM;
      break;
    case 6:
      armral_mod = ARMRAL_MOD_64QAM;
      break;
    case 8:
      armral_mod = ARMRAL_MOD_256QAM;
      break;
    default:
      LOG_E(PHY, "Modulation order not supported: Qm = %d\n", Qm);
      break;
  }

  int size_f = ceil_mod(E, 16);
  int8_t f[size_f] __attribute__((aligned(64)));
  memset(f, 0, size_f * sizeof(int8_t));
  for (int i = 0; i < (size_f >> 4); i++) {
    ((simde__m128i *)f)[i] = simde_mm_packs_epi16(((simde__m128i *)ulsch_llr)[2 * i], ((simde__m128i *)ulsch_llr)[2 * i + 1]);
  }

  if (*rdata->d_to_be_cleared) {
    memset(rdata->d, 0, 68 * 384 * sizeof(*rdata->d));
    *rdata->d_to_be_cleared = false;
  }

  armral_status status_rate_recovery =
      armral_ldpc_rate_recovery(armral_bg, Z, E, Nref, F, K, rv_index, armral_mod, f, (int8_t *)rdata->d);
  if (status_rate_recovery == ARMRAL_ARGUMENT_ERROR) {
    LOG_E(PHY, "argument error in armral rate recovery\n");
  } else if (status_rate_recovery == ARMRAL_RESULT_FAIL) {
    LOG_E(PHY, "failure in armral rate recovery\n");
  }

  stop_meas(rdata->p_ts_rate_unmatch);
  start_meas(rdata->p_ts_ldpc_decode);

  int crc_type = crcType(C, A);
  uint32_t crc_len = 0;
  switch (crc_type) {
    case CRC24_A:
    case CRC24_B:
      crc_len = 24;
      break;

    case CRC16:
      crc_len = 16;
      break;

    case CRC8:
      crc_len = 8;
      break;

    default:
      AssertFatal(1, "Invalid crc_type \n");
  }
  uint32_t crc_idx = Kprime - crc_len;

  armral_status status_decoding =
      armral_ldpc_decode_block((int8_t *)rdata->d, armral_bg, Z, crc_idx, max_number_iterations, llrProcBuf);
  if (status_decoding == ARMRAL_ARGUMENT_ERROR) {
    LOG_E(PHY, "argument error in armral decoding\n");
  }

  if (check_crc(llrProcBuf, Kprime, crc_type)) {
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

int nrLDPC_prepare_TB_decoding(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters,
                               int pusch_id,
                               thread_info_tm_t *t_info)
{
  nrLDPC_TB_decoding_parameters_t *nrLDPC_TB_decoding_parameters = &nrLDPC_slot_decoding_parameters->TBs[pusch_id];

  *nrLDPC_TB_decoding_parameters->processedSegments = 0;

  for (int r = 0; r < nrLDPC_TB_decoding_parameters->C; r++) {
    nrLDPC_decoding_parameters_t *rdata = &((nrLDPC_decoding_parameters_t *)t_info->buf)[t_info->len];
    DevAssert(t_info->len < t_info->cap);
    rdata->ans = t_info->ans;
    t_info->len += 1;

    rdata->A = nrLDPC_TB_decoding_parameters->A;
    rdata->K = nrLDPC_TB_decoding_parameters->K;
    rdata->Z = nrLDPC_TB_decoding_parameters->Z;
    rdata->F = nrLDPC_TB_decoding_parameters->F;
    rdata->C = nrLDPC_TB_decoding_parameters->C;
    rdata->E = nrLDPC_TB_decoding_parameters->segments[r].E;
    rdata->BG = nrLDPC_TB_decoding_parameters->BG;
    rdata->max_number_iterations = nrLDPC_TB_decoding_parameters->max_ldpc_iterations;
    rdata->tbslbrm = nrLDPC_TB_decoding_parameters->tbslbrm;
    rdata->Qm = nrLDPC_TB_decoding_parameters->Qm;
    rdata->rv_index = nrLDPC_TB_decoding_parameters->rv_index;
    rdata->llr = nrLDPC_TB_decoding_parameters->segments[r].llr;
    rdata->d = nrLDPC_TB_decoding_parameters->segments[r].d;
    rdata->d_to_be_cleared = nrLDPC_TB_decoding_parameters->segments[r].d_to_be_cleared;
    rdata->c = nrLDPC_TB_decoding_parameters->segments[r].c;
    rdata->decodeSuccess = &nrLDPC_TB_decoding_parameters->segments[r].decodeSuccess;
    // rdata->abort_decode = nrLDPC_TB_decoding_parameters->abort_decode;
    rdata->p_ts_rate_unmatch = &nrLDPC_TB_decoding_parameters->segments[r].ts_rate_unmatch;
    rdata->p_ts_ldpc_decode = &nrLDPC_TB_decoding_parameters->segments[r].ts_ldpc_decode;

    task_t t = {.func = &nr_process_decode_segment, .args = rdata};
    pushTpool(nrLDPC_slot_decoding_parameters->threadPool, t);

    LOG_D(PHY, "Added a block to decode, in pipe: %d\n", r);
  }
  return nrLDPC_TB_decoding_parameters->C;
}

int32_t nrLDPC_coding_init(void)
{
  return 0;
}

int32_t nrLDPC_coding_shutdown(void)
{
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

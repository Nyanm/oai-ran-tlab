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

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
 * \brief Top-level routines for implementing LDPC encoding of transport channels
 */

#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include "common/utils/nr/nr_common.h"
#include <openair2/UTIL/OPT/opt.h>

#include <syscall.h>
#include "armral.h"

#define DEBUG_LDPC_ENCODING
//#define DEBUG_LDPC_ENCODING_FREE 1

typedef struct ldpc8blocks_args_s {
  nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters;
  uint32_t macro_num;
  time_stats_t *toutput;
  task_ans_t *ans;
} ldpc8blocks_args_t;

static void ldpc8blocks(void *p)
{
  ldpc8blocks_args_t *args = (ldpc8blocks_args_t *)p;
  nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters = args->nrLDPC_TB_encoding_parameters;

  uint32_t A = nrLDPC_TB_encoding_parameters->A;
  uint32_t K = nrLDPC_TB_encoding_parameters->K;
  uint32_t Z = nrLDPC_TB_encoding_parameters->Z;
  uint32_t F = nrLDPC_TB_encoding_parameters->F;
  uint32_t C = nrLDPC_TB_encoding_parameters->C;
  uint8_t Qm = nrLDPC_TB_encoding_parameters->Qm;
  uint16_t nb_rb = nrLDPC_TB_encoding_parameters->nb_rb;

  unsigned int G = nrLDPC_TB_encoding_parameters->G;
  LOG_D(PHY, "dlsch coding A %d K %d G %d (nb_rb %d, Qm %d)\n", A, K, G, nb_rb, (int)Qm);

  armral_ldpc_graph_t armral_bg = nrLDPC_TB_encoding_parameters->BG == 2 ? LDPC_BASE_GRAPH_2 : LDPC_BASE_GRAPH_1;

  uint32_t Nref = 3 * nrLDPC_TB_encoding_parameters->tbslbrm / (2 * C); // R_LBRM = 2/3

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

  // 384 (lifting size in bits) / 8 = 48 (lifting size in bytes)
  uint8_t d[68 * 48] __attribute__((aligned(64)));
  uint8_t f[68 * 48] __attribute__((aligned(64)));

  unsigned int macro_segment = 8 * args->macro_num;
  unsigned int macro_segment_end = (C > 8 * (args->macro_num + 1)) ? 8 * (args->macro_num + 1) : C;
  unsigned int offset_output_bit = 0;

  // Calculate initial offset in output buffer
  for (int r = 0; r < macro_segment; r++) {
    offset_output_bit += nrLDPC_TB_encoding_parameters->segments[r].E;
  }

  for (int r = macro_segment; r < macro_segment_end; r++) {
    start_meas(&nrLDPC_TB_encoding_parameters->segments[r].ts_ldpc_encode);

    memset(d, 0, 68 * 48 * sizeof(*d));
    armral_status status_encoding = armral_ldpc_encode_block(nrLDPC_TB_encoding_parameters->segments[r].c, armral_bg, Z, F, d);
    if (status_encoding == ARMRAL_ARGUMENT_ERROR) {
      LOG_E(PHY, "argument error in armral encoding\n");
    } else if (status_encoding == ARMRAL_RESULT_FAIL) {
      LOG_E(PHY, "failure in armral encoding\n");
    }

    stop_meas(&nrLDPC_TB_encoding_parameters->segments[r].ts_ldpc_encode);

    start_meas(&nrLDPC_TB_encoding_parameters->segments[r].ts_rate_match);

    unsigned int E = nrLDPC_TB_encoding_parameters->segments[r].E;
    LOG_D(NR_PHY,
          "Rate Matching, Code segment %d/%d (coded bits (G) %u, E %d, Filler bits %d, Filler offset %d Qm %d, nb_rb "
          "%d,nrOfLayer %d)...\n",
          r,
          C,
          G,
          E,
          F,
          K - F - 2 * Z,
          Qm,
          nb_rb,
          nrLDPC_TB_encoding_parameters->nb_layers);

    if (K - F - 2 * Z > E) {
      LOG_E(PHY, "dlsch coding A %d  Kr %d G %d (nb_rb %d, Qm %d)\n", A, K, G, nb_rb, Qm);

      LOG_E(NR_PHY,
            "Rate Matching, Code segments %d/%d (coded bits (G) %u, E %d, Kr %d, Filler bits %d, Filler offset %d Qm %d, "
            "nb_rb %d)...\n",
            macro_segment,
            C,
            G,
            E,
            K,
            F,
            K - F - 2 * Z,
            Qm,
            nb_rb);
    }

    memset(f, 0, 68 * 48 * sizeof(*f));
    armral_status status_rate_matching =
        armral_ldpc_rate_matching(armral_bg, Z, E, Nref, F, K, nrLDPC_TB_encoding_parameters->rv_index, armral_mod, d, f);
    if (status_rate_matching == ARMRAL_ARGUMENT_ERROR) {
      LOG_E(PHY, "argument error in armral rate matching\n");
    } else if (status_rate_matching == ARMRAL_RESULT_FAIL) {
      LOG_E(PHY, "failure in armral rate matching\n");
    }

    stop_meas(&nrLDPC_TB_encoding_parameters->segments[r].ts_rate_match);

    if (args->toutput != NULL)
      start_meas(args->toutput);

    unsigned int f_reverse_size = (E & 7) == 0 ? E >> 3 : (E >> 3) + 1;
    reverse_bits_u8(f, f_reverse_size, f);

    if ((offset_output_bit & 7) == 0) {
      LOG_D(PHY, "encoder output aligned on byte, using memcpy\n");
      unsigned int output_copy_size = (E & 7) == 0 ? E >> 3 : (E >> 3) + 1;
      memcpy(&nrLDPC_TB_encoding_parameters->output[offset_output_bit >> 3], f, output_copy_size);
    } else {
      LOG_D(PHY, "encoder output NOT aligned on byte, using Neon\n");
      uint64_t *f_64 = (uint64_t *)f;
      unsigned int output_offset_64 = (offset_output_bit >> 3) - ((offset_output_bit >> 3) & 7);
      uint64_t *output_64 = (uint64_t *)&nrLDPC_TB_encoding_parameters->output[output_offset_64];
      unsigned int nb_vec = (E & 63) == 0 ? E >> 6 : (E >> 6) + 1;
      int64_t shift_bit_low = offset_output_bit & 63;
      int64x1_t shift_bit_low_64x1 = vld1_s64(&shift_bit_low);
      int64_t shift_bit_high = shift_bit_low - 64;
      int64x1_t shift_bit_high_64x1 = vld1_s64(&shift_bit_high);
      for (int i = 0; i < nb_vec; i++) {
        uint64x1_t f_64x1 = vld1_u64(&f_64[i]);

        uint64x1_t f_64x1_low = vshl_u64(f_64x1, shift_bit_low_64x1);
        uint64x1_t output_64x1_low = vld1_u64(&output_64[i]);
        output_64[i] = (uint64_t)vorr_u64(output_64x1_low, f_64x1_low);

        uint64x1_t f_64x1_high = vshl_u64(f_64x1, shift_bit_high_64x1);
        uint64x1_t output_64x1_high = vld1_u64(&output_64[i + 1]);
        output_64[i + 1] = (uint64_t)vorr_u64(output_64x1_high, f_64x1_high);
      }
    }

    if (args->toutput != NULL)
      stop_meas(args->toutput);

    // Increment offset in output buffer
    offset_output_bit += E;
    // TODO Manage race condition every 8 segment end
  }

  // Task running in // completed
  completed_task_ans(args->ans);
}

static int nrLDPC_prepare_TB_encoding(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters,
                                      int dlsch_id,
                                      thread_info_tm_t *t_info)
{
  nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters = &nrLDPC_slot_encoding_parameters->TBs[dlsch_id];
  uint32_t C = nrLDPC_TB_encoding_parameters->C;

  size_t const n_seg = (C / 8 + ((C & 7) == 0 ? 0 : 1));

  for (int j = 0; j < n_seg; j++) {
    ldpc8blocks_args_t *perJobImpp = &((ldpc8blocks_args_t *)t_info->buf)[t_info->len];
    DevAssert(t_info->len < t_info->cap);
    perJobImpp->ans = t_info->ans;
    t_info->len += 1;

    perJobImpp->macro_num = j;
    perJobImpp->nrLDPC_TB_encoding_parameters = nrLDPC_TB_encoding_parameters;
    perJobImpp->toutput = nrLDPC_slot_encoding_parameters->toutput;

    task_t t = {.func = ldpc8blocks, .args = perJobImpp};
    pushTpool(nrLDPC_slot_encoding_parameters->threadPool, t);
  }
  return n_seg;
}

int nrLDPC_coding_encoder(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  int nbTasks = 0;

  for (int dlsch_id = 0; dlsch_id < nrLDPC_slot_encoding_parameters->nb_TBs; dlsch_id++) {
    nrLDPC_TB_encoding_parameters_t *nrLDPC_TB_encoding_parameters = &nrLDPC_slot_encoding_parameters->TBs[dlsch_id];
    uint32_t C = nrLDPC_TB_encoding_parameters->C;
    size_t n_seg = (C / 8 + ((C & 7) == 0 ? 0 : 1));
    nbTasks += n_seg;
  }
  ldpc8blocks_args_t arr[nbTasks];
  task_ans_t ans;
  init_task_ans(&ans, nbTasks);
  thread_info_tm_t t_info = {.buf = (uint8_t *)arr, .len = 0, .cap = nbTasks, .ans = &ans};

  int nbEncode = 0;
  for (int dlsch_id = 0; dlsch_id < nrLDPC_slot_encoding_parameters->nb_TBs; dlsch_id++) {
    nbEncode += nrLDPC_prepare_TB_encoding(nrLDPC_slot_encoding_parameters, dlsch_id, &t_info);
  }
  if (nbEncode < nbTasks) {
    completed_many_task_ans(&ans, nbTasks - nbEncode);
  }
  // Execute thread pool tasks
  join_task_ans(&ans);

  return 0;
}

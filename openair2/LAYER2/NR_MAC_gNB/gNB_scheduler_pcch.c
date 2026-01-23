/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief gNB PCCH (paging) scheduling procedures
 */

#include <stdbool.h>
#include <stdint.h>
#include "assertions.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "common/utils/ds/byte_array.h"
#include "openair2/RRC/NR/MESSAGES/asn1_msg.h"

static void free_pcch_item(nr_mac_pcch_item_t *item)
{
  DevAssert(item);
  free(item->sdu);
  item->sdu = NULL;
  item->sdu_len = 0;
  item->ue_id = 0;
}

/** @brief FIFO head: one SDU held outside spsc_q between dequeue and TX. */
static nr_mac_pcch_item_t *pcch_head_get(NR_COMMON_channels_t *cc)
{
  if (!cc->pcch_pending.sdu) {
    if (!spsc_q_get(&cc->pcch_queue, &cc->pcch_pending, sizeof(cc->pcch_pending)))
      return NULL;
  }
  return &cc->pcch_pending;
}

void nr_mac_pcch_queue_free(NR_COMMON_channels_t *cc)
{
  DevAssert(cc);
  if (cc->pcch_pending.sdu)
    free_pcch_item(&cc->pcch_pending);
  nr_mac_pcch_item_t item;
  while (spsc_q_get(&cc->pcch_queue, &item, sizeof(item)))
    free_pcch_item(&item);
  spsc_q_free(&cc->pcch_queue);
}

void nr_mac_pcch_queue_init(NR_COMMON_channels_t *cc)
{
  DevAssert(cc);
  const bool ok = spsc_q_alloc(&cc->pcch_queue, NR_PCCH_MAX_PAGING_RECORDS, sizeof(nr_mac_pcch_item_t));
  AssertFatal(ok, "failed to allocate PCCH queue\n");
}

static void pcch_queue_push(spsc_q_t *q, module_id_t module_id, nr_mac_pcch_item_t *item)
{
  DevAssert(q);
  DevAssert(item);
  DevAssert(item->sdu);
  DevAssert(item->sdu_len > 0);

  if (!spsc_q_put(q, item, sizeof(*item))) {
    free_pcch_item(item);
    LOG_W(NR_MAC, "[gNB %d] PCCH queue full, dropping new SDU\n", module_id);
  }
}

/** @brief Build NR Paging (PCCH) SDU for ng-5G-S-TMSI and enqueue it in the MAC PCCH queue. */
void nr_mac_pcch_enqueue(module_id_t module_id, uint64_t fiveg_s_tmsi, uint16_t ue_id)
{
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  DevAssert(mac);
  const int CC_id = 0;
  NR_COMMON_channels_t *cc = &mac->common_channels[CC_id];

  nr_paging_params_t params = {
      .ue_identity_type = NR_PagingUE_Identity_PR_ng_5G_S_TMSI,
      .ue_identity = {.fiveg_s_tmsi = fiveg_s_tmsi & ((1ULL << 48) - 1)},
      .access_type = false,
  };
  byte_array_t msg = do_NR_Paging(&params);
  if (!msg.buf || msg.len == 0) {
    LOG_E(NR_MAC, "[gNB %d] PCCH paging: do_NR_Paging failed for 5G-S-TMSI=0x%012lx\n", module_id, fiveg_s_tmsi);
    return;
  }

  AssertFatal(msg.len <= UINT16_MAX, "PCCH SDU length %zu exceeds uint16_t\n", msg.len);
  nr_mac_pcch_item_t item = {
      .sdu = msg.buf,
      .sdu_len = msg.len,
      .ue_id = ue_id % 1024,
  };
  msg.buf = NULL;
  msg.len = 0;

  NR_SCHED_LOCK(&mac->sched_lock);
  pcch_queue_push(&cc->pcch_queue, module_id, &item);
  NR_SCHED_UNLOCK(&mac->sched_lock);

  LOG_I(NR_MAC, "[gNB %d] PCCH SDU enqueued length=%zu UE_ID=%u (5G-S-TMSI=0x%012lx)\n", module_id, msg.len, ue_id, fiveg_s_tmsi);
  free_byte_array(msg);
}

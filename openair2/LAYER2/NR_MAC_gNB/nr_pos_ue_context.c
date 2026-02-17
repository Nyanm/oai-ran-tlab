/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */
#include "nr_mac_gNB.h"
#include "tree.h"
#include "T.h"
#include "common/utils/T/T.h"
#include "common/utils/utils.h"
#include "common/utils/LOG/log.h"

static RB_HEAD(pos_act_ue_map, positioning_activation_info_s) pos_act_ue_head = RB_INITIALIZER(&pos_act_ue_head);

/* Generate the tree management functions prototypes */
RB_PROTOTYPE(pos_act_ue_map, positioning_activation_info_s, entries, pos_act_ue_compare_rnti);

static int pos_act_ue_compare_rnti(positioning_activation_info_t *p1, positioning_activation_info_t *p2)
{
  if (p1->rnti > p2->rnti) {
    return 1;
  }

  if (p1->rnti < p2->rnti) {
    return -1;
  }

  return 0;
}

/* Generate the tree management functions */
RB_GENERATE(pos_act_ue_map, positioning_activation_info_s, entries, pos_act_ue_compare_rnti);

void pos_act_store_ue_context(const rnti_t rnti)
{
  LOG_I(NR_MAC, "Create positioning UE context for rnti : %x\n", rnti);
  positioning_activation_info_t *ue_info_p = calloc_or_fail(1, sizeof(*ue_info_p));
  ue_info_p->rnti = rnti;

  if (RB_INSERT(pos_act_ue_map, &pos_act_ue_head, ue_info_p))
    LOG_E(NR_MAC, "Bug in UE uniq number allocation %u, we try to add a existing UE\n", ue_info_p->rnti);
  return;
}

struct positioning_activation_info_s *pos_act_get_ue_context(rnti_t rnti)
{
  positioning_activation_info_t temp = {.rnti = rnti};
  return RB_FIND(pos_act_ue_map, &pos_act_ue_head, &temp);
}

struct positioning_activation_info_s *pos_act_detach_ue_context(rnti_t rnti)
{
  positioning_activation_info_t *tmp = pos_act_get_ue_context(rnti);
  if (tmp == NULL) {
    LOG_E(NR_MAC, "Trying to free a NULL UE context, %x\n", rnti);
    return NULL;
  }
  RB_REMOVE(pos_act_ue_map, &pos_act_ue_head, tmp);
  return tmp;
}

void pos_act_free_ue_context(positioning_activation_info_t *ue_info)
{
  if (!ue_info)
    return;
  free(ue_info);
}

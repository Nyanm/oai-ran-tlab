/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */
#include <stdint.h>
#include "tree.h"
#include "ds/byte_array.h"
#include "common/platform_types.h"

#ifndef POS_ACT_UE_CONTEXT_H_
#define POS_ACT_UE_CONTEXT_H_

typedef struct positioning_activation_info_s {
  /* Tree related data */
  RB_ENTRY(positioning_activation_info_s) entries;
  rnti_t rnti;
} positioning_activation_info_t;

void pos_act_store_ue_context(const rnti_t rnti);
positioning_activation_info_t *pos_act_get_ue_context(rnti_t rnti);
positioning_activation_info_t *pos_act_detach_ue_context(rnti_t rnti);
void pos_act_free_ue_context(positioning_activation_info_t *ue_info);

#endif /* POS_ACT_UE_CONTEXT_H_ */

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

/* \file nr_ue_scheduler_sl_v2x.c
 * \brief Rel-16 NR sidelink V2X semi-persistent resource selection and LBT gate
 */

#include <stdlib.h>
#include <string.h>

#include "NR_MAC_UE/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"

#define NR_SL_V2X_DEFAULT_PROB_RESOURCE_KEEP 0.8
#define NR_SL_V2X_DEFAULT_LBT_ED_THRESHOLD_DBM -72
#define NR_SL_V2X_DEFAULT_MAX_LBT_FAILURES 4
#define NR_SL_V2X_SLSCH_LCID 4

static bool nr_sl_v2x_is_configured(const NR_UE_MAC_INST_t *mac)
{
  return mac != NULL && mac->SL_MAC_PARAMS != NULL && mac->sl_tx_res_pool != NULL && mac->SL_MAC_PARAMS->sl_TxPool[0] != NULL;
}

static double nr_sl_v2x_random_unit(void)
{
  return (double)(lrand48() & 0x7fffffff) / (double)0x7fffffff;
}

static void nr_sl_v2x_reset_selected_list(NR_UE_MAC_INST_t *mac, const sl_resource_info_t *resource)
{
  if (mac->sl_candidate_resources == NULL) {
    mac->sl_candidate_resources = calloc(1, sizeof(*mac->sl_candidate_resources));
    init_list(mac->sl_candidate_resources, sizeof(*resource), 1);
  } else if (mac->sl_candidate_resources->data == NULL || mac->sl_candidate_resources->element_size != sizeof(*resource)) {
    init_list(mac->sl_candidate_resources, sizeof(*resource), 1);
  } else {
    mac->sl_candidate_resources->size = 0;
  }

  if (resource != NULL)
    push_back(mac->sl_candidate_resources, (void *)resource);
}

static uint16_t nr_sl_v2x_next_reselection_counter(NR_UE_MAC_INST_t *mac)
{
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  uint16_t rri = sl_mac->mac_tx_params.rri;
  uint16_t counter = get_random_reselection_counter(rri);

  if (counter == 0)
    counter = 1;

  sl_mac->mac_tx_params.resel_counter = counter;
  mac->sl_resel_counter = counter;
  mac->sl_c_resel = counter;
  return counter;
}

static void nr_sl_v2x_release_sps_resource(NR_UE_MAC_INST_t *mac)
{
  nr_sl_v2x_sps_state_t *sps = &mac->sl_v2x_scheduler.sps;
  sps->active = false;
  sps->next_abs_slot = -1;
  sps->released++;
  nr_sl_v2x_reset_selected_list(mac, NULL);
}

static bool nr_sl_v2x_pick_candidate(const List_t *candidates, sl_resource_info_t *selected)
{
  if (candidates == NULL || candidates->size == 0)
    return false;

  size_t idle_count = 0;
  for (size_t i = 0; i < candidates->size; i++) {
    const sl_resource_info_t *candidate = (const sl_resource_info_t *)((const char *)candidates->data + i * candidates->element_size);
    if (!candidate->slot_busy)
      idle_count++;
  }

  size_t target = 0;
  if (idle_count > 0) {
    target = lrand48() % idle_count;
    for (size_t i = 0; i < candidates->size; i++) {
      const sl_resource_info_t *candidate = (const sl_resource_info_t *)((const char *)candidates->data + i * candidates->element_size);
      if (!candidate->slot_busy && target-- == 0) {
        *selected = *candidate;
        return true;
      }
    }
  }

  *selected = *(const sl_resource_info_t *)((const char *)candidates->data + (lrand48() % candidates->size) * candidates->element_size);
  return true;
}

static bool nr_sl_v2x_activate_sps_resource(NR_UE_MAC_INST_t *mac, const frameslot_t *frame_slot)
{
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  uint8_t mu = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  List_t *candidates = get_candidate_resources((frameslot_t *)frame_slot, mac, &mac->sl_sensing_data, &mac->sl_transmit_history);

  sl_resource_info_t selected = {0};
  bool selected_ok = nr_sl_v2x_pick_candidate(candidates, &selected);
  if (candidates != NULL) {
    free_list_mem(candidates);
    free(candidates);
  }

  if (!selected_ok)
    return false;

  nr_sl_v2x_sps_state_t *sps = &mac->sl_v2x_scheduler.sps;
  sps->active = true;
  sps->pool_id = 0;
  sps->p_rsvp_slots = time_to_slots(mu, sl_mac->mac_tx_params.rri);
  if (sps->p_rsvp_slots == 0)
    sps->p_rsvp_slots = 1;
  sps->c_resel = nr_sl_v2x_next_reselection_counter(mac);
  sps->c_resel_initial = sps->c_resel;
  sps->resource = selected;
  sps->current_resource = selected;
  sps->next_abs_slot = normalize(&selected.sfn, mu);
  sps->reselections++;
  nr_sl_v2x_reset_selected_list(mac, &selected);

  LOG_D(NR_MAC,
        "SL-V2X SPS selected %4d.%2d subch %d len %d C_resel %d RRI-slots %d\n",
        selected.sfn.frame,
        selected.sfn.slot,
        selected.sl_subchan_start,
        selected.sl_subchan_len,
        sps->c_resel,
        sps->p_rsvp_slots);
  return true;
}

static void nr_sl_v2x_refresh_config(NR_UE_MAC_INST_t *mac)
{
  nr_sl_v2x_scheduler_t *scheduler = &mac->sl_v2x_scheduler;
  scheduler->enabled = true;

  if (mac->m_slProbResourceKeep <= 0.0 || mac->m_slProbResourceKeep > 1.0)
    mac->m_slProbResourceKeep = NR_SL_V2X_DEFAULT_PROB_RESOURCE_KEEP;

  scheduler->lbt.enabled = true;
  if (scheduler->lbt.mode == NR_SL_V2X_LBT_DISABLED)
    scheduler->lbt.mode = NR_SL_V2X_LBT_TYPE2;
  if (scheduler->lbt.energy_detection_threshold_dbm == 0)
    scheduler->lbt.energy_detection_threshold_dbm = NR_SL_V2X_DEFAULT_LBT_ED_THRESHOLD_DBM;
  if (mac->SL_MAC_PARAMS->mac_tx_params.sl_thresh_rsrp != 0)
    scheduler->lbt.energy_detection_threshold_dbm = mac->SL_MAC_PARAMS->mac_tx_params.sl_thresh_rsrp;
  if (scheduler->lbt.max_consecutive_failures == 0)
    scheduler->lbt.max_consecutive_failures = NR_SL_V2X_DEFAULT_MAX_LBT_FAILURES;
}

void nr_ue_sl_v2x_init_scheduler(NR_UE_MAC_INST_t *mac)
{
  AssertFatal(mac != NULL, "mac cannot be NULL\n");
  nr_sl_v2x_scheduler_t *scheduler = &mac->sl_v2x_scheduler;
  memset(scheduler, 0, sizeof(*scheduler));
  scheduler->enabled = true;
  scheduler->lbt.enabled = true;
  scheduler->lbt.mode = NR_SL_V2X_LBT_TYPE2;
  scheduler->lbt.energy_detection_threshold_dbm = NR_SL_V2X_DEFAULT_LBT_ED_THRESHOLD_DBM;
  scheduler->lbt.max_consecutive_failures = NR_SL_V2X_DEFAULT_MAX_LBT_FAILURES;
  scheduler->lbt.last_failure.frame = -1;
  scheduler->lbt.last_failure.slot = -1;
  scheduler->sps.next_abs_slot = -1;
  mac->m_slProbResourceKeep = NR_SL_V2X_DEFAULT_PROB_RESOURCE_KEEP;
}

sl_resource_info_t *nr_ue_sl_v2x_select_resource(NR_UE_MAC_INST_t *mac,
                                                 const frameslot_t *frame_slot,
                                                 sl_sidelink_slot_type_t slot_type)
{
  if (!nr_sl_v2x_is_configured(mac) || slot_type != SIDELINK_SLOT_TYPE_TX)
    return NULL;

  nr_sl_v2x_refresh_config(mac);
  if (!mac->sl_v2x_scheduler.enabled)
    return mac->sl_candidate_resources ? get_resource_element(mac->sl_candidate_resources, *frame_slot) : NULL;

  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  uint8_t mu = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  nr_sl_v2x_sps_state_t *sps = &mac->sl_v2x_scheduler.sps;
  int64_t current_abs_slot = normalize((frameslot_t *)frame_slot, mu);

  if (!sps->active && !nr_sl_v2x_activate_sps_resource(mac, frame_slot))
    return NULL;

  if (current_abs_slot > sps->next_abs_slot) {
    int64_t missed = ((current_abs_slot - sps->next_abs_slot) / sps->p_rsvp_slots) + 1;
    sps->next_abs_slot += missed * sps->p_rsvp_slots;
  }

  if (current_abs_slot != sps->next_abs_slot)
    return NULL;

  sps->current_resource = sps->resource;
  sps->current_resource.sfn = *frame_slot;
  return &sps->current_resource;
}

static bool nr_sl_v2x_has_pending_tx(NR_UE_MAC_INST_t *mac, const frameslot_t *frame_slot)
{
  if (mac->sl_tx_res_pool != NULL && mac->sl_tx_res_pool->sl_PSFCH_Config_r16 && is_feedback_scheduled(mac, frame_slot->frame, frame_slot->slot))
    return true;

  for (int i = 0; i < MAX_SL_UE_CONNECTIONS && mac->sl_info.list[i] != NULL; i++) {
    NR_SL_UE_sched_ctrl_t *sched_ctrl = &mac->sl_info.list[i]->UE_sched_ctrl;
    if (sched_ctrl->retrans_sl_harq.head >= 0)
      return true;
    if (sched_ctrl->sched_csi_report.active &&
        sched_ctrl->sched_csi_report.frame == frame_slot->frame &&
        sched_ctrl->sched_csi_report.slot == frame_slot->slot)
      return true;
  }

  mac_rlc_status_resp_t rlc_status = nr_mac_rlc_status_ind(mac->ue_id, frame_slot->frame, NR_SL_V2X_SLSCH_LCID);
  return rlc_status.bytes_in_buffer > 0;
}

static bool nr_sl_v2x_sensed_resource_busy(NR_UE_MAC_INST_t *mac, const sl_resource_info_t *resource, const frameslot_t *frame_slot)
{
  sl_nr_ue_mac_params_t *sl_mac = mac->SL_MAC_PARAMS;
  uint8_t mu = sl_mac->sl_phy_config.sl_config_req.sl_bwp_config.sl_scs;
  int64_t cycle_slots = nr_slots_per_frame[mu] * 1024;
  int64_t target_abs = normalize((frameslot_t *)frame_slot, mu);
  int16_t threshold = mac->sl_v2x_scheduler.lbt.energy_detection_threshold_dbm;

  for (size_t i = 0; i < mac->sl_sensing_data.size; i++) {
    const sensing_data_t *sensed = (const sensing_data_t *)((const char *)mac->sl_sensing_data.data + i * mac->sl_sensing_data.element_size);
    if (sensed->sl_rsrp <= threshold)
      continue;
    if (!overlapped_resource(sensed->subch_start, sensed->subch_len, resource->sl_subchan_start, resource->sl_subchan_len))
      continue;

    int64_t sensed_abs = normalize((frameslot_t *)&sensed->frame_slot, mu);
    int64_t diff = (target_abs - sensed_abs + cycle_slots) % cycle_slots;
    if (diff == 0)
      return true;

    uint16_t p_rsvp_slots = sensed->rsvp ? time_to_slots(mu, sensed->rsvp) : 0;
    if (p_rsvp_slots > 0 && diff > 0 && (diff % p_rsvp_slots) == 0)
      return true;
  }

  return false;
}

bool nr_ue_sl_v2x_lbt_allows_tx(NR_UE_MAC_INST_t *mac,
                                const sl_resource_info_t *resource,
                                const frameslot_t *frame_slot)
{
  if (!nr_sl_v2x_is_configured(mac) || resource == NULL || !mac->sl_v2x_scheduler.lbt.enabled)
    return true;
  if (!nr_sl_v2x_has_pending_tx(mac, frame_slot))
    return true;

  nr_sl_v2x_lbt_state_t *lbt = &mac->sl_v2x_scheduler.lbt;
  lbt->attempts++;

  if (!nr_sl_v2x_sensed_resource_busy(mac, resource, frame_slot))
    return true;

  lbt->failures++;
  lbt->consecutive_failures++;
  lbt->last_failure = *frame_slot;
  LOG_D(NR_MAC,
        "SL-V2X LBT busy at %4d.%2d subch %d len %d RSRP threshold %d failures %d/%d\n",
        frame_slot->frame,
        frame_slot->slot,
        resource->sl_subchan_start,
        resource->sl_subchan_len,
        lbt->energy_detection_threshold_dbm,
        lbt->consecutive_failures,
        lbt->max_consecutive_failures);

  if (lbt->consecutive_failures >= lbt->max_consecutive_failures) {
    LOG_D(NR_MAC, "SL-V2X LBT releasing SPS resource after consecutive failures\n");
    nr_sl_v2x_release_sps_resource(mac);
    lbt->consecutive_failures = 0;
  }

  return false;
}

void nr_ue_sl_v2x_notify_tx_result(NR_UE_MAC_INST_t *mac,
                                   const sl_resource_info_t *resource,
                                   const frameslot_t *frame_slot,
                                   bool transmitted)
{
  if (!nr_sl_v2x_is_configured(mac) || resource == NULL || !mac->sl_v2x_scheduler.sps.active || !transmitted)
    return;

  nr_sl_v2x_scheduler_t *scheduler = &mac->sl_v2x_scheduler;
  nr_sl_v2x_sps_state_t *sps = &scheduler->sps;
  scheduler->lbt.successes++;
  scheduler->lbt.consecutive_failures = 0;

  if (sps->c_resel > 0)
    sps->c_resel--;
  mac->sl_c_resel = sps->c_resel;
  mac->sl_resel_counter = sps->c_resel;
  mac->SL_MAC_PARAMS->mac_tx_params.resel_counter = sps->c_resel;

  if (sps->c_resel == 0) {
    if (nr_sl_v2x_random_unit() <= mac->m_slProbResourceKeep) {
      sps->c_resel = nr_sl_v2x_next_reselection_counter(mac);
      sps->c_resel_initial = sps->c_resel;
      sps->kept++;
      LOG_D(NR_MAC,
            "SL-V2X SPS keeping resource %4d.%2d subch %d for C_resel %d\n",
            sps->resource.sfn.frame,
            sps->resource.sfn.slot,
            sps->resource.sl_subchan_start,
            sps->c_resel);
    } else {
      LOG_D(NR_MAC,
            "SL-V2X SPS releasing resource %4d.%2d subch %d after C_resel expiry\n",
            frame_slot->frame,
            frame_slot->slot,
            resource->sl_subchan_start);
      nr_sl_v2x_release_sps_resource(mac);
      return;
    }
  }

  sps->next_abs_slot += sps->p_rsvp_slots;
}

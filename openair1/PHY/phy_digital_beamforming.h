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

#ifndef __PHY_DIGITAL_BEAMFORMING__H__
#define __PHY_DIGITAL_BEAMFORMING__H__

#include "defs_RU.h"

void fill_rx_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_ul_tti_request_number_of_pdus_t *ul_pdu);
struct grid_slot_entry *get_grid_slot(struct grid_slots_head *head, const uint32_t frame, const uint32_t slot);
void remove_grid_slot(struct grid_slots_head *head, const uint32_t frame, const uint32_t slot);
void apply_rx_beamforming(RU_t *ru, const uint32_t frame, const uint32_t slot);
void apply_tx_beamforming(RU_t *ru, const uint32_t frame, const uint32_t slot);
void process_rx_grid_info_bf(RU_t *ru, const uint32_t frame, const uint32_t slot);

#endif

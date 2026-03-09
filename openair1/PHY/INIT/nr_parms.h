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
#ifndef _NR_PARMS_H_
#define _NR_PARMS_H_
#include "fapi_nr_ue_interface.h"
#include "openair1/PHY/defs_nr_UE.h"

#ifdef __cplusplus
extern "C" {
#endif

void nr_init_frame_parms(nfapi_nr_config_request_scf_t *config, NR_DL_FRAME_PARMS *frame_parms);
int nr_init_frame_parms_ue(NR_DL_FRAME_PARMS *frame_parms, fapi_nr_config_request_t *config, uint16_t nr_band);
void nr_init_frame_parms_ue_sa(NR_DL_FRAME_PARMS *frame_parms, const nrUE_cell_params_t *cell);
int nr_init_frame_parms_ue_sl(NR_DL_FRAME_PARMS *fp,
                              sl_nr_phy_config_request_t *config,
                              int threequarter_fs,
                              uint32_t ofdm_offset_divisor);
void nr_dump_frame_parms(NR_DL_FRAME_PARMS *frame_parms);

#ifdef __cplusplus
}
#endif

#endif

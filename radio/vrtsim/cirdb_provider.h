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

/*
 * In-process CIR DB provider for VRTSIM
 */

#ifndef OAI_VRTSIM_CIRDB_PROVIDER_H
#define OAI_VRTSIM_CIRDB_PROVIDER_H

#include <stdint.h>
#include <stdbool.h>
#include "SIMULATION/TOOLS/sim.h"  // channel_desc_t, struct complexf

#ifdef __cplusplus
extern "C" {
#endif

/* Optional override for the DB location.
 * Pass either a file path to cir_db.bin, or a directory that contains cir_db.bin.
 * Call this once before cirdb_connect. Safe to skip if you rely on defaults.
 */
void cirdb_set_path(const char *path);

void cirdb_connect(int id,
                   int num_tx_antennas,
                   int num_rx_antennas,
                   channel_desc_t **channel_desc_out);

void cirdb_stop(void);

void cirdb_set_path_override(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* OAI_VRTSIM_CIRDB_PROVIDER_H */

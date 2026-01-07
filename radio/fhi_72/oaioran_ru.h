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

#ifndef __OAIORAN_RU_H__
#define __OAIORAN_RU_H__
#include <stdint.h>
#include "common/utils/threadPool/notified_fifo.h"

extern notifiedFIFO_t ru_dl_sync_fifo;

// Installs a callback that triggers callbacks_per_slot times in a slot when when all packets for
// corresponding symbols are received. e.g. if callbacks_per_slot == 2, the callback will trigger
// at reception window ends for symbols 7 and 14. This in turn unblocks xran_oru_tx_read_slot and
// can be used for timing purposes
void install_symbol_callback(void* handle, int callbacks_per_slot, int mu);

// Read samples DL IQ samples for frame slot symbol for all antennas
int xran_oru_tx_read_slot(uint32_t **txdataF, int nb_tx, int *frame, int *slot, int *symbol, int *num_symbols, struct timespec *ts);
void xran_oru_send_prach(uint32_t *prachF, int aarx, int frame, int slot, int symbol);

#endif

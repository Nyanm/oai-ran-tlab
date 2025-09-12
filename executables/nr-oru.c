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
#define _GNU_SOURCE
#include "nr-oru.h"
#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/INIT/nr_phy_init.h"
#include "openair1/SCHED_NR/sched_nr.h"

#include <sched.h>

void oru_downlink_processing(RU_t *ru, c16_t* txDataF_ptr[ru->nb_tx], int frame, int slot, int start_symbol, int num_symbols) {
  start_meas(&ru->tx_fhaul);
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    apply_nr_rotation_TX(fp,
                         txDataF_ptr[aatx],
                         fp->symbol_rotation[0],
                         slot,
                         fp->N_RB_DL,
                         start_symbol,
                         num_symbols);
    nr_feptx0(ru, slot, start_symbol, num_symbols, aatx);
  }
  // Assume this function called in order
  static int frame_tx_unwrap = 0;
  static int last_frame = 0;
  if (frame < last_frame) {
    frame_tx_unwrap += 1024;
  }
  last_frame = frame;

  uint64_t timestamp_tx =
      (uint64_t)(frame + frame_tx_unwrap) * fp->samples_per_subframe * 10 + fp->get_samples_slot_timestamp(slot, fp, 0);

  // TODO: prepare tx here
  // tx_rf(ru, frame, slot, timestamp_tx);
  (void)timestamp_tx;
  stop_meas(&ru->tx_fhaul);
}

void *oru_north_read_thread(void *arg)
{
  ORU_t *oru = (ORU_t *)arg;

  RU_t *ru = (RU_t *)oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  char threadname[40];
  sprintf(threadname, "oru_thread %u", ru->idx);

  AssertFatal(ru->ifdevice.xran_api.north_in_func != NULL, "No fronthaul interface at north port");
  __attribute__((aligned(32))) c16_t txDataF[ru->nb_tx][ceil_mod(fp->ofdm_symbol_size * 14, 32)];
  c16_t *txDataF_ptr[ru->nb_tx];
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    txDataF_ptr[aatx] = txDataF[aatx];
  }
  while (!oai_exit) {
    int num_symbols = 0;
    sense_of_time_t sense_of_time;
    ru->ifdevice.xran_api.north_in_func((uint32_t **)txDataF_ptr, ru->nb_tx, &sense_of_time, &num_symbols);
    LOG_D(PHY,
          "[RU_thread] read data: frame %d, slot %d, start_symbol %d, num_symbols %d\n",
          sense_of_time.frame,
          sense_of_time.slot,
          sense_of_time.symbol,
          num_symbols);
    oru_downlink_processing(ru, txDataF_ptr, sense_of_time.frame, sense_of_time.slot, sense_of_time.symbol, num_symbols);
  }
  return NULL;
}

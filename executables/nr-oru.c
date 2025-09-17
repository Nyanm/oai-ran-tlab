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

typedef struct {
  openair0_timestamp sample;
  int slot;
  int frame;
  int symbol;
} initial_sync_t;

typedef struct {
  int frame_unwrap;
  int last_frame;
  int64_t sync_offset;
  initial_sync_t initial_sync;
} sync_params_t;

extern void tx_rf(RU_t *ru, int frame, int slot, uint64_t timestamp);

void perform_initial_sync(ORU_t *oru, sense_of_time_t *sense_of_time, sync_params_t *sync_params)
{
  initial_sync_t *initial_sync = &sync_params->initial_sync;
  initial_sync->frame = sense_of_time->frame;
  initial_sync->slot = sense_of_time->slot;
  initial_sync->symbol = sense_of_time->symbol;
  initial_sync->sample = oru->ru->rfdevice.get_timestamp(&oru->ru->rfdevice, &sense_of_time->ts);
  NR_DL_FRAME_PARMS *fp = oru->ru->nr_frame_parms;
  sync_params->frame_unwrap = 0;
  sync_params->last_frame = initial_sync->frame;
  sync_params->sync_offset = initial_sync->sample;
  sync_params->sync_offset -= (uint64_t)(sync_params->initial_sync.frame) * fp->samples_per_subframe * 10
                              + fp->get_samples_slot_timestamp(sync_params->initial_sync.slot, fp, 0);
  LOG_I(PHY,
        "RU synchronized: frame, slot %d.%d, symbol %d,  offset: %ld\n",
        initial_sync->frame,
        initial_sync->slot,
        initial_sync->symbol,
        sync_params->sync_offset);
}

static inline int get_samples_symbol(int slot, int symbol, NR_DL_FRAME_PARMS *fp)
{
  if (symbol == 0) {
    return 0;
  }

  if (fp->numerology_index == 0) {
    int num_samples = 0;
    int num_symbols_to_add = symbol;

    // Add symbol 7
    if (symbol > 7) {
      num_samples += fp->nb_prefix_samples0 + fp->ofdm_symbol_size;
      num_symbols_to_add--;
    }
    // add symbol 0
    num_samples += fp->nb_prefix_samples0 + fp->ofdm_symbol_size;
    num_symbols_to_add--;

    num_samples += num_symbols_to_add * (fp->nb_prefix_samples + fp->ofdm_symbol_size);
    return num_samples;
  } else {
    int num_samples = 0;
    int num_symbols_to_add = symbol;

    // Add first symbol
    num_samples += (slot % (fp->slots_per_subframe / 2)) ? fp->nb_prefix_samples : fp->nb_prefix_samples0;
    num_samples += fp->ofdm_symbol_size;
    num_symbols_to_add--;

    num_samples += (fp->ofdm_symbol_size + fp->nb_prefix_samples) * num_symbols_to_add;
    return num_samples;
  }
}

openair0_timestamp get_timestamp(ORU_t *oru, sense_of_time_t *sense_of_time, sync_params_t *sync_params)
{
  if (sync_params->last_frame > sense_of_time->frame) {
    sync_params->frame_unwrap++;
  }
  sync_params->last_frame = sense_of_time->frame;
  NR_DL_FRAME_PARMS *fp = oru->ru->nr_frame_parms;
  int num_frames = sense_of_time->frame + sync_params->frame_unwrap * 1024;

  uint64_t timestamp = (uint64_t)(num_frames)*fp->samples_per_subframe * 10
                       + fp->get_samples_slot_timestamp(sense_of_time->slot, fp, 0)
                       + get_samples_symbol(sense_of_time->slot, sense_of_time->symbol, fp);

  timestamp += sync_params->sync_offset;
  return timestamp;
}

void oru_downlink_processing(RU_t *ru,
                             c16_t *txDataF_ptr[ru->nb_tx],
                             int frame,
                             int slot,
                             int start_symbol,
                             int num_symbols,
                             openair0_timestamp timestamp_tx)
{
  start_meas(&ru->tx_fhaul);
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    apply_nr_rotation_TX(fp, txDataF_ptr[aatx], fp->symbol_rotation[0], slot, fp->N_RB_DL, start_symbol, num_symbols);
    nr_feptx0(ru, slot, start_symbol, num_symbols, aatx);
  }
  tx_rf(ru, frame, slot, timestamp_tx);
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
  sync_params_t sync_params;

  while (!oai_exit) {
    int num_symbols = 0;
    sense_of_time_t sense_of_time;
    ru->ifdevice.xran_api.north_in_func((uint32_t **)txDataF_ptr, ru->nb_tx, &sense_of_time, &num_symbols);
    if (sense_of_time.symbol == 0) {
      perform_initial_sync(oru, &sense_of_time, &sync_params);
      break;
    }
  }

  while (!oai_exit) {
    int num_symbols = 0;
    sense_of_time_t sense_of_time;
    ru->ifdevice.xran_api.north_in_func((uint32_t **)txDataF_ptr, ru->nb_tx, &sense_of_time, &num_symbols);
    openair0_timestamp timestamp_tx = get_timestamp(oru, &sense_of_time, &sync_params);
    if ((sense_of_time.frame & 0xff) == 0 && sense_of_time.slot == 0) {
      LOG_I(PHY,
            "[RU_thread] read data: frame %d, slot %d, start_symbol %d, num_symbols %d\n",
            sense_of_time.frame,
            sense_of_time.slot,
            sense_of_time.symbol,
            num_symbols);
    }
    oru_downlink_processing(ru,
                            txDataF_ptr,
                            sense_of_time.frame,
                            sense_of_time.slot,
                            sense_of_time.symbol,
                            num_symbols,
                            timestamp_tx);
  }
  return NULL;
}

void *oru_south_read_thread(void *arg)
{
  ORU_t *oru = arg;
  RU_t *ru = oru->ru;

  const int num_samples = 3000;
  c16_t throwaway_samples[ru->nb_rx][num_samples];
  void *rxp[ru->nb_rx];
  for (int i = 0; i < ru->nb_rx; i++)
    rxp[i] = throwaway_samples[i];

  openair0_timestamp timestamp;
  while (!oai_exit) {
    ru->rfdevice.trx_read_func(&ru->rfdevice, &timestamp, rxp, num_samples, ru->nb_rx);
  }

  // Perform RX processing
  return NULL;
}

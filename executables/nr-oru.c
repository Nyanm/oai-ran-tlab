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
#include "PHY/TOOLS/tools_defs.h"
#define _GNU_SOURCE
#include "nr-oru.h"
#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/INIT/nr_phy_init.h"
#include "openair1/SCHED_NR/sched_nr.h"
#include "notified_fifo.h"
#include "openair1/PHY/NR_TRANSPORT/nr_transport_proto.h"

#include <sched.h>

typedef struct {
  int frame_unwrap;
  int last_frame;
  int64_t sync_offset;
} sync_params_t;

extern void tx_rf_symbols(RU_t *ru, int frame, int slot, uint64_t timestamp, int start_symbol, int num_symbols);

void perform_initial_sync(ORU_t *oru, sense_of_time_t *sense_of_time, initial_sync_t *initial_sync)
{
  initial_sync->frame = sense_of_time->frame;
  initial_sync->slot = sense_of_time->slot;
  initial_sync->symbol = sense_of_time->symbol;
  initial_sync->sample = oru->ru->rfdevice.get_timestamp(&oru->ru->rfdevice, &sense_of_time->ts);
  LOG_I(PHY,
        "RU synchronized: frame, slot %d.%d, symbol %d, sample: %ld\n",
        initial_sync->frame,
        initial_sync->slot,
        initial_sync->symbol,
        initial_sync->sample);
}

void initialize_sync_params(NR_DL_FRAME_PARMS *fp, sync_params_t *sync_params, initial_sync_t *initial_sync)
{
  sync_params->frame_unwrap = 0;
  sync_params->last_frame = initial_sync->frame;
  sync_params->sync_offset = initial_sync->sample;
  sync_params->sync_offset -=
      (uint64_t)(initial_sync->frame) * fp->samples_per_subframe * 10 + fp->get_samples_slot_timestamp(initial_sync->slot, fp, 0);
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
                       + get_samples_symbol_timestamp(fp, sense_of_time->slot, sense_of_time->symbol);

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
  for (int symbol = start_symbol; symbol < start_symbol + num_symbols; symbol++) {
    LOG_D(PHY,
          "Ant 0 Signal energy %d.%d.%d %.3f\n",
          frame,
          slot,
          symbol,
          10 * log10(signal_energy_nodc(&txDataF_ptr[0][fp->ofdm_symbol_size * symbol], fp->ofdm_symbol_size)));
  }
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    apply_nr_rotation_TX(fp, txDataF_ptr[aatx], fp->symbol_rotation[0], slot, fp->N_RB_DL, start_symbol, num_symbols);
    nr_feptx0(ru, slot, start_symbol, num_symbols, aatx);
  }
  tx_rf_symbols(ru, frame, slot, timestamp_tx, start_symbol, num_symbols);
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
  ru->common.txdataF_BF = (int32_t **)txDataF_ptr;
  sync_params_t sync_params;

  notifiedFIFO_elt_t *sync_msg = newNotifiedFIFO_elt(sizeof(initial_sync_t), 0, NULL, NULL);
  initial_sync_t *initial_sync = NotifiedFifoData(sync_msg);
  while (!oai_exit) {
    int num_symbols = 0;
    sense_of_time_t sense_of_time;
    ru->ifdevice.xran_api.north_in_func((uint32_t **)txDataF_ptr, ru->nb_tx, &sense_of_time, &num_symbols);
    if (sense_of_time.symbol == 0) {
      perform_initial_sync(oru, &sense_of_time, initial_sync);
      initialize_sync_params(oru->ru->nr_frame_parms, &sync_params, initial_sync);
      break;
    }
  }
  pushNotifiedFIFO(&oru->sync_fifo, sync_msg);

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
    nfapi_nr_config_request_scf_t *cfg = &ru->config;
    int slot_type = nr_slot_select(cfg, sense_of_time.frame, sense_of_time.slot % fp->slots_per_frame);
    if (slot_type != NR_UPLINK_SLOT)
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

void rx_initial_sync(ORU_t *oru, int *slot, int *frame)
{
  RU_t *ru = oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  const int num_samples = 3000;
  c16_t throwaway_samples[ru->nb_rx][num_samples];
  void *rxp[ru->nb_rx];
  for (int i = 0; i < ru->nb_rx; i++)
    rxp[i] = throwaway_samples[i];

  openair0_timestamp timestamp;
  initial_sync_t initial_sync;
  while (!oai_exit) {
    int samples_read = ru->rfdevice.trx_read_func(&ru->rfdevice, &timestamp, rxp, num_samples, ru->nb_rx);
    AssertFatal(samples_read == num_samples, "Unexpected number of samples received\n");
    notifiedFIFO_elt_t *elt = pollNotifiedFIFO(&oru->sync_fifo);
    if (elt) {
      memcpy(&initial_sync, NotifiedFifoData(elt), sizeof(initial_sync));
      break;
    }
  }

  // Synchornize to ORAN timing
  int next_slot = initial_sync.slot;
  int next_frame = initial_sync.frame;
  openair0_timestamp next_sample = timestamp + num_samples;
  int64_t diff = next_sample - initial_sync.sample;
  LOG_I(PHY,
        "Sychronizing to frame slot %d.%d, sample %ld next_sample %ld diff %ld\n",
        next_frame,
        next_slot,
        initial_sync.sample,
        next_sample,
        diff);

  uint64_t samples_to_sync_by = 0;
  if (diff < 0) {
    samples_to_sync_by = -diff;
  } else {
    while (diff > 0) {
      uint32_t samples_per_slot = fp->get_samples_per_slot(next_slot, fp);
      samples_to_sync_by += samples_per_slot;
      diff -= samples_per_slot;
      next_slot++;
      if (next_slot == fp->slots_per_frame) {
        next_slot = 0;
        next_frame++;
        if (next_frame == 1024) {
          next_frame = 0;
        }
      }
    }
    samples_to_sync_by += diff;
  }

  LOG_I(PHY, "Thrashing %lu samples to sync to slot %d, frame %d\n", samples_to_sync_by, next_slot, next_frame);
  while (!oai_exit && samples_to_sync_by > 0) {
    int samples_to_read = min(num_samples, samples_to_sync_by);
    int samples_read = ru->rfdevice.trx_read_func(&ru->rfdevice, &timestamp, rxp, samples_to_read, ru->nb_rx);
    AssertFatal(samples_to_read == samples_read, "Unexpected number of samples received\n");
    samples_to_sync_by -= samples_to_read;
  }
  *slot = next_slot;
  *frame = next_frame;
}

void *oru_south_read_thread(void *arg)
{
  ORU_t *oru = arg;
  RU_t *ru = oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  int current_slot = 0;
  int current_frame = 0;
  rx_initial_sync(oru, &current_slot, &current_frame);
  const int symbols_per_iteration = 7;

  while (!oai_exit) {
    int rx_slot_type = nr_slot_select(&ru->config, current_frame, current_slot);
    for (int symbol = 0; symbol < 14; symbol += symbols_per_iteration) {
      int samples_to_read = get_samples_symbol_duration(fp, current_slot, symbol, symbols_per_iteration);
      size_t offset = fp->get_samples_slot_timestamp(current_slot, fp, 0) + get_samples_symbol_timestamp(fp, current_slot, symbol);
      c16_t *rxp[fp->nb_antennas_rx];
      for (int aarx = 0; aarx < fp->nb_antennas_rx; aarx++) {
        rxp[aarx] = (c16_t *)&ru->common.rxdata[aarx][offset];
      }

      openair0_timestamp timestamp;
      int num_samples_read = ru->rfdevice.trx_read_func(&ru->rfdevice, &timestamp, (void **)rxp, samples_to_read, ru->nb_rx);
      AssertFatal(num_samples_read == samples_to_read, "Unexpected number of samples received\n");
      if (rx_slot_type == NR_UPLINK_SLOT || rx_slot_type == NR_MIXED_SLOT) {
        if (current_slot == 19 && symbol + symbols_per_iteration > 13) {
          int prach_fmt = 8; // TODO: get this from RU config
          int numRA = 0; // TODO: get this from RU config
          int beam = 0; // TODO: Set to 0 for now
          int prachStartSymbol = 0; // TODO: get this from RU config
          int prachStartSlot = current_slot; // TODO: get this from RU config
          int prachOccasion = 0; // TODO: get this from RU config
          rx_nr_prach_ru(ru, prach_fmt, numRA, beam, prachStartSymbol, prachStartSlot, prachOccasion, current_frame, current_slot);
          ru->ifdevice.xran_api.north_write_prach_func((uint32_t **)ru->prach_rxsigF[0], current_slot, current_frame);   
        }
      }
      ru->ifdevice.xran_api.north_out_func(current_slot, 0, ru->nb_rx, ((1 << symbols_per_iteration) - 1) << symbol);
    }
    current_slot++;
    if (current_slot == fp->slots_per_frame) {
      current_slot = 0;
      current_frame++;
      if (current_frame == 1024) {
        current_frame = 0;
      }
    }
  }

  // Perform RX processing
  return NULL;
}

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
#include "PHY/defs_RU.h"
#include "PHY/impl_defs_nr.h"
#include "log.h"
#include "nfapi_nr_interface_scf.h"
#include "platform_types.h"
#include "task_ans.h"
#include "thread-pool.h"
#include "time_meas.h"
#include <bits/pthreadtypes.h>
#include <time.h>
#include <unistd.h>
#define _GNU_SOURCE
#include "nr-oru.h"
#include "openair1/PHY/defs_nr_common.h"
#include "openair1/PHY/INIT/nr_phy_init.h"
#include "openair1/SCHED_NR/sched_nr.h"
#include "notified_fifo.h"
#include "openair1/PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "openair2/LAYER2/NR_MAC_COMMON/nr_mac_common.h"

#include <sched.h>

typedef struct {
  int x;
  int y;
  int size_x;
  int size_y;
} thread_grid2d_t;

typedef struct {
  int frame_unwrap;
  int last_frame;
  int64_t sync_offset;
} sync_params_t;

typedef struct {
  ORU_t *oru;
  int frame;
  int slot;
} prach_task_args_t;

typedef struct {
  ORU_t *oru;
  int frame;
  int slot;
  int num_symbols;
  int start_symbol;
  thread_grid2d_t thread_grid;
} pusch_task_args_t;

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

typedef struct {
  RU_t *ru;
  NR_DL_FRAME_PARMS *fp;
  int slot;
  int start_symbol;
  int num_symbols;
  int aatx;
  c16_t *txdataF;
  task_ans_t *task_ans;
} dl_symbol_process_t;

void dl_symbol_process(void *arg)
{
  dl_symbol_process_t *args = (dl_symbol_process_t *)arg;
  apply_nr_rotation_TX(args->fp,
                       args->txdataF,
                       args->fp->symbol_rotation[0],
                       args->slot,
                       args->fp->N_RB_DL,
                       args->start_symbol,
                       args->num_symbols);
  nr_feptx0(args->ru, args->slot, args->start_symbol, args->num_symbols, args->aatx);
  completed_task_ans(args->task_ans);
}

void oru_downlink_processing(ORU_t *oru,
                             c16_t *txDataF_ptr[oru->ru->nb_tx],
                             int frame,
                             int slot,
                             int start_symbol,
                             int num_symbols,
                             openair0_timestamp timestamp_tx)
{
  RU_t *ru = oru->ru;
  start_meas(&ru->tx_fhaul);
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  int num_paralell_workers_per_antenna = num_symbols > 4 ? 2 : 1; // Ensure at least quarter slot parallelization
  task_t tasks[ru->nb_tx][num_paralell_workers_per_antenna];
  dl_symbol_process_t dl_process_args[ru->nb_tx][num_paralell_workers_per_antenna];
  task_ans_t task_ans;
  init_task_ans(&task_ans, num_paralell_workers_per_antenna * ru->nb_tx);
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    for (int i = 0; i < num_paralell_workers_per_antenna; i++) {
      tasks[aatx][i].func = dl_symbol_process;
      tasks[aatx][i].args = (void *)&dl_process_args[aatx][i];
      dl_process_args[aatx][i].ru = ru;
      dl_process_args[aatx][i].fp = fp;
      dl_process_args[aatx][i].slot = slot;
      dl_process_args[aatx][i].start_symbol = start_symbol + num_symbols / num_paralell_workers_per_antenna * i;
      dl_process_args[aatx][i].num_symbols =
          min(num_symbols / num_paralell_workers_per_antenna, num_symbols - (num_symbols / num_paralell_workers_per_antenna) * i);
      dl_process_args[aatx][i].aatx = aatx;
      dl_process_args[aatx][i].txdataF = txDataF_ptr[aatx];
      dl_process_args[aatx][i].task_ans = &task_ans;
      pushTpool(&oru->tpool, tasks[aatx][i]);
    }
  }
  LOG_D(PHY,
        "[RU_thread] transmit data: frame %d, slot %d, start_symbol %d, num_symbols %d, timestamp %ld\n",
        frame,
        slot,
        start_symbol,
        num_symbols,
        timestamp_tx);
  join_task_ans(&task_ans);
  tx_rf_symbols(ru, frame, slot, timestamp_tx, start_symbol, num_symbols);
  stop_meas(&ru->tx_fhaul);
}

void *oru_sync_thread(void *arg)
{
  ORU_t *oru = (ORU_t *)arg;

  RU_t *ru = (RU_t *)oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  AssertFatal(ru->ifdevice.xran_api.north_in_func != NULL, "No fronthaul interface at north port");
  __attribute__((aligned(32))) c16_t txDataF[ru->nb_tx][ceil_mod(fp->ofdm_symbol_size * 14, 32)];
  c16_t *txDataF_ptr[ru->nb_tx];
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    txDataF_ptr[aatx] = txDataF[aatx];
  }

  initial_sync_t initial_sync;
  while (!oai_exit) {
    int num_symbols = 0;
    sense_of_time_t sense_of_time;
    ru->ifdevice.xran_api.north_in_func((uint32_t **)txDataF_ptr, ru->nb_tx, &sense_of_time, &num_symbols);
    if (sense_of_time.symbol == 0) {
      perform_initial_sync(oru, &sense_of_time, &initial_sync);
      break;
    }
  }

  for (int i = 0; i < oru->num_sync_messages_needed; i++) {
    notifiedFIFO_elt_t *sync_msg = newNotifiedFIFO_elt(sizeof(initial_sync_t), 0, NULL, NULL);
    initial_sync_t *initial_sync_p = NotifiedFifoData(sync_msg);
    *initial_sync_p = initial_sync;
    pushNotifiedFIFO(&oru->sync_fifo, sync_msg);
  }

  return NULL;
}

void *oru_north_read_thread(void *arg)
{
  ORU_t *oru = (ORU_t *)arg;

  RU_t *ru = (RU_t *)oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  AssertFatal(ru->ifdevice.xran_api.north_in_func != NULL, "No fronthaul interface at north port");
  __attribute__((aligned(32))) c16_t txDataF[ru->nb_tx][ceil_mod(fp->ofdm_symbol_size * 14, 32)];
  c16_t *txDataF_ptr[ru->nb_tx];
  for (int aatx = 0; aatx < ru->nb_tx; aatx++) {
    txDataF_ptr[aatx] = txDataF[aatx];
  }
  ru->common.txdataF_BF = (int32_t **)txDataF_ptr;

  notifiedFIFO_elt_t * elt = pullNotifiedFIFO(&oru->sync_fifo);
  initial_sync_t *initial_sync = NotifiedFifoData(elt);
  delNotifiedFIFO_elt(elt);
  sync_params_t sync_params;
  initialize_sync_params(fp, &sync_params, initial_sync);
  LOG_A(PHY,
        "ORU North read thread started at frame %d, slot %d, symbol %d\n",
        initial_sync->frame,
        initial_sync->slot,
        initial_sync->symbol);

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
      oru_downlink_processing(oru,
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

void receive_prach(ORU_t *oru, int frame, int slot)
{
  RU_t *ru = oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  uint16_t RA_sfn_index = -1;
  if (get_nr_prach_sched_from_info(oru->prach_info, ru->prach_config_index, frame, slot, ru->numerology, FR1, &RA_sfn_index, TDD)) {
    // Fill PRACH item
    prach_item_t prach_id;
    prach_id.frame = frame;
    prach_id.slot = slot;
    prach_id.num_slots = oru->prach_info.format < 4 ? get_long_prach_dur(oru->prach_info.format, fp->numerology_index) : 1;
    prach_id.msg1_frequencystart = ru->prach_msg1_freq;
    prach_id.mu = 1;
    nfapi_nr_config_request_scf_t *cfg = &ru->config;
    prach_id.prach_sequence_length = cfg->prach_config.prach_sequence_length.value;
    prach_id.restricted_set = 0;
    prach_id.numerology_index = fp->numerology_index;
    prach_id.nb_rx = ru->nb_rx;
    prach_id.rx_prach = &oru->rx_prach;
    prach_id.beams[0] = 0; // TODO: Beamforming not supported yet

    // Fill PRACH PDU
    nfapi_nr_prach_pdu_t *prach_pdu = &prach_id.pdu;
    prach_pdu->prach_start_symbol = oru->prach_info.start_symbol;
    prach_pdu->num_prach_ocas = 1; // TODO: Hardcoded.

    uint16_t format0 = oru->prach_info.format & 0xff;
    uint16_t format1 = (oru->prach_info.format >> 8) & 0xff;
    if (format1 != 0xff) {
      switch (format0) {
        case 0xa1:
          prach_pdu->prach_format = 11;
          break;
        case 0xa2:
          prach_pdu->prach_format = 12;
          break;
        case 0xa3:
          prach_pdu->prach_format = 13;
          break;
        default:
          AssertFatal(1 == 0, "Only formats A1/B1 A2/B2 A3/B3 are valid for dual format");
      }
    } else {
      switch (format0) {
        case 0:
          prach_pdu->prach_format = 0;
          break;
        case 1:
          prach_pdu->prach_format = 1;
          break;
        case 2:
          prach_pdu->prach_format = 2;
          break;
        case 3:
          prach_pdu->prach_format = 3;
          break;
        case 0xa1:
          prach_pdu->prach_format = 4;
          break;
        case 0xa2:
          prach_pdu->prach_format = 5;
          break;
        case 0xa3:
          prach_pdu->prach_format = 6;
          break;
        case 0xb1:
          prach_pdu->prach_format = 7;
          break;
        case 0xb4:
          prach_pdu->prach_format = 8;
          break;
        case 0xc0:
          prach_pdu->prach_format = 9;
          break;
        case 0xc2:
          prach_pdu->prach_format = 10;
          break;
        default:
          AssertFatal(1 == 0, "Invalid PRACH format");
      }
    }
    rx_nr_prach_ru(&prach_id, ru->common.rxdata, fp, ru->N_TA_offset);
    uint32_t *prach_sig[fp->nb_antennas_rx];
    for (int i = 0; i < fp->nb_antennas_rx; i++) {
        prach_sig[i] = (uint32_t *)prach_id.rxsigF[0][i];
    }
    ru->ifdevice.xran_api.north_write_prach_func(prach_sig, prach_id.slot, prach_id.frame);
  }
}

void receive_pusch(ORU_t *oru, int frame, int slot, int start_symbol, int num_symbols, thread_grid2d_t *thread_grid)
{
  int aarx_start = thread_grid->y;
  int aarx_stride = thread_grid->size_y;
  int symbol_start = start_symbol + thread_grid->x;
  int symbol_stride = thread_grid->size_x;

  RU_t *ru = oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  nfapi_nr_config_request_scf_t *config = &ru->config;
  nfapi_nr_tdd_table_t *tdd_table = &config->tdd_table;
  AssertFatal(tdd_table->tdd_period.tl.tag == NFAPI_NR_CONFIG_TDD_PERIOD_TAG, "");
  int nb_periods_per_frame = get_nb_periods_per_frame(tdd_table->tdd_period.value);
  int n_tdd_period = fp->slots_per_frame / nb_periods_per_frame;
  AssertFatal(n_tdd_period > 0, "n_tdd_period is zero\n");
  nfapi_nr_max_num_of_symbol_per_slot_t *max_num_of_symbol_per_slot_list =
      config->tdd_table.max_tdd_periodicity_list[slot % n_tdd_period].max_num_of_symbol_per_slot_list;

  c16_t rxdataF[fp->symbols_per_slot * fp->ofdm_symbol_size] __attribute__((aligned(32)));
  for (int aarx = aarx_start; aarx < fp->nb_antennas_rx; aarx += aarx_stride) {
    for (int symbol = symbol_start; symbol < start_symbol + num_symbols; symbol += symbol_stride) {
      if (max_num_of_symbol_per_slot_list[symbol].slot_config.value != 1)
        continue;
      nr_slot_fep_ul(fp,
                     ru->common.rxdata[aarx],
                     (int32_t *)rxdataF,
                     symbol,
                     slot,
                     ru->N_TA_offset);
      apply_nr_rotation_symbol_RX(fp,
                                  &rxdataF[symbol * fp->ofdm_symbol_size],
                                  fp->symbol_rotation[link_type_ul],
                                  fp->N_RB_UL,
                                  slot,
                                  symbol);
      ru->ifdevice.xran_api.north_write_pusch_func((uint32_t *)&rxdataF[symbol * fp->ofdm_symbol_size],
                                                    frame,
                                                    slot,
                                                    symbol,
                                                    aarx);
    }
  }
}

void pusch_job(void *args) {
  pusch_task_args_t *job = (pusch_task_args_t *)args;
  receive_pusch(job->oru, job->frame, job->slot, job->start_symbol, job->num_symbols, &job->thread_grid);
}

void prach_job(void *args) {
  prach_task_args_t *job = (prach_task_args_t *)args;
  receive_prach(job->oru, job->frame, job->slot);
}

void *oru_south_read_thread(void *arg)
{
  ORU_t *oru = arg;
  RU_t *ru = oru->ru;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;

  int current_slot = 0;
  int current_frame = 0;
  rx_initial_sync(oru, &current_slot, &current_frame);
  const int symbols_per_iteration = 4;
  notifiedFIFO_t response_fifo;
  initNotifiedFIFO(&response_fifo);

  while (!oai_exit) {
    int rx_slot_type = nr_slot_select(&ru->config, current_frame, current_slot);
    for (int symbol = 0; symbol < 14; symbol += symbols_per_iteration) {
      int num_symbols = min(symbols_per_iteration, 14 - symbol);
      int samples_to_read = get_samples_symbol_duration(fp, current_slot, symbol, num_symbols);
      size_t offset = fp->get_samples_slot_timestamp(current_slot, fp, 0) + get_samples_symbol_timestamp(fp, current_slot, symbol);
      c16_t *rxp[fp->nb_antennas_rx];
      for (int aarx = 0; aarx < fp->nb_antennas_rx; aarx++) {
        rxp[aarx] = (c16_t *)&ru->common.rxdata[aarx][offset];
      }

      openair0_timestamp timestamp;
      int num_samples_read = ru->rfdevice.trx_read_func(&ru->rfdevice, &timestamp, (void **)rxp, samples_to_read, ru->nb_rx);
      AssertFatal(num_samples_read == samples_to_read, "Unexpected number of samples received\n");
      LOG_D(PHY,
            "[ORU south] read data: frame %d, slot %d, symbol %d, timestamp %ld num_symbols %d, samples %d\n",
            current_frame,
            current_slot,
            symbol,
            timestamp,
            num_symbols,
            num_samples_read);

      if (rx_slot_type == NR_UPLINK_SLOT || rx_slot_type == NR_MIXED_SLOT) {
        int num_jobs = 0;
        start_meas(&oru->rx);
        if (symbol == 0) {
          num_jobs++;
          notifiedFIFO_elt_t *prach_task = newNotifiedFIFO_elt(sizeof(prach_task_args_t), 0, &response_fifo, prach_job);
          prach_task_args_t *job = NotifiedFifoData(prach_task);
          job->oru = oru;
          job->frame = current_frame;
          job->slot = current_slot;
          pushNotifiedFIFO(&oru->prach_actor.fifo, prach_task);
        }

        for (int i = 0; i < NUM_PUSCH_ACTORS; i++) {
          num_jobs++;
          notifiedFIFO_elt_t *pusch_task = newNotifiedFIFO_elt(sizeof(pusch_task_args_t), 0, &response_fifo, pusch_job);
          pusch_task_args_t *job = NotifiedFifoData(pusch_task);
          job->oru = oru;
          job->frame = current_frame;
          job->slot = current_slot;
          job->start_symbol = symbol;
          job->num_symbols = num_symbols;
          job->thread_grid.x = 0;
          job->thread_grid.size_x = 1;
          job->thread_grid.y = i;
          job->thread_grid.size_y = NUM_PUSCH_ACTORS;
          pushNotifiedFIFO(&oru->pusch_actors[i].fifo, pusch_task);
        }

        while (num_jobs > 0) {
          notifiedFIFO_elt_t *elt = pullNotifiedFIFO(&response_fifo);
          delNotifiedFIFO_elt(elt);
          num_jobs--;
        }
        ru->ifdevice.xran_api.north_out_func(current_slot, 0, ru->nb_rx, ((1 << num_symbols) - 1) << symbol);
        stop_meas(&oru->rx);
      }
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

  return NULL;
}

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

#include "platform_types.h"
#include <stdio.h>
#include <string.h>
#include "common_lib.h"
#include "radio/ETHERNET/ethernet_lib.h"

#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "openair1/PHY/defs_gNB.h"
#include "../fhi_72/oran-params.h"
#include <stdatomic.h>
#include "common/utils/threadPool/notified_fifo.h"
#include "PHY/MODULATION/modulation_common.h"
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "openair1/PHY/MODULATION/nr_modulation.h"
#include "PHY/INIT/nr_phy_init.h"
#include "SCHED_NR/sched_nr.h"
#include "common/utils/threadPool/pthread_utils.h"
#include "openair1/PHY/NR_TRANSPORT/nr_transport_proto.h"

typedef struct {
  openair0_timestamp timestamp;
  int slot;
  int frame;
} fd_rfsim_timestamp;

typedef struct {
  bool is_started;
  rru_config_msg_type_t last_msg;
  bool capabilities_sent;
  openair0_config_t openair0_cfg;
  notifiedFIFO_t sync_fifo;
  RU_t ru;
  pthread_mutex_t mutex;
} fd_rfsim_state_t;

extern void nr_feptx(void *arg);
static int trx_start(openair0_device *device)
{
  printf("Starting fd_rfsim\n");
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;
  if (s->ru.rfdevice.trx_start_func) {
    if (s->ru.rfdevice.trx_start_func(&s->ru.rfdevice) < 0) {
      LOG_E(HW, "Failed to start subdevice\n");
      return -1;
    }
  }
  s->is_started = true;
  return 0;
}

static void trx_end(openair0_device *device)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;
  if (s->ru.rfdevice.trx_end_func) {
    s->ru.rfdevice.trx_end_func(&s->ru.rfdevice);
  }
  s->is_started = false;
}

static int trx_stop(openair0_device *device)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;
  if (s->ru.rfdevice.trx_stop_func) {
    if (s->ru.rfdevice.trx_stop_func(&s->ru.rfdevice) < 0) {
      LOG_E(HW, "Failed to stop subdevice\n");
      return -1;
    }
  }
  s->is_started = false;
  return 0;
}

static int trx_set_freq(openair0_device *device, openair0_config_t *openair0_cfg)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;

  if (!s->is_started) {
    LOG_E(HW, "Device not started, cannot set frequency\n");
    return -1;
  }

  if (s->ru.rfdevice.trx_set_freq_func) {
    if (s->ru.rfdevice.trx_set_freq_func(&s->ru.rfdevice, openair0_cfg) < 0) {
      LOG_E(HW, "Failed to set frequency on subdevice\n");
      return -1;
    }
  }
  return 0;
}

static int trx_set_gains(openair0_device *device, openair0_config_t *openair0_cfg)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;

  if (!s->is_started) {
    LOG_E(HW, "Device not started, cannot set gains\n");
    return -1;
  }

  if (s->ru.rfdevice.trx_set_gains_func) {
    if (s->ru.rfdevice.trx_set_gains_func(&s->ru.rfdevice, openair0_cfg) < 0) {
      LOG_E(HW, "Failed to set gains on subdevice\n");
      return -1;
    }
  }
  return 0;
}


static int trx_reset_stats(openair0_device *device)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;

  if (!s->is_started) {
    LOG_E(HW, "Device not started, cannot reset stats\n");
    return -1;
  }

  if (s->ru.rfdevice.trx_reset_stats_func) {
    return s->ru.rfdevice.trx_reset_stats_func(&s->ru.rfdevice);
  }
  return 0;
}

int ethernet_tune(openair0_device *device, unsigned int option, int value)
{
  return 0;
}

static int trx_write_raw(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  return 0;
}

static int trx_read_raw(openair0_device *device, openair0_timestamp *timestamp, void **buff, int nsamps, int cc)
{
  return 0;
}

char *msg_type(int t)
{
  static char *s[12] = {
      "RAU_tick",
      "RRU_capabilities",
      "RRU_config",
      "RRU_config_ok",
      "RRU_start",
      "RRU_stop",
      "RRU_sync_ok",
      "RRU_frame_resynch",
      "RRU_MSG_max_num",
      "RRU_check_sync",
      "RRU_config_update",
      "RRU_config_update_ok",
  };

  if (t < 0 || t > 11)
    return "UNKNOWN";
  return s[t];
}

static int trx_ctlsend(openair0_device *device, void *msg, ssize_t msg_len)
{
  RRU_CONFIG_msg_t *rru_config_msg = msg;
  printf("rru_config_msg->type %d [%s]\n", rru_config_msg->type, msg_type(rru_config_msg->type));
  return msg_len;
}

static int trx_ctlrecv(openair0_device *device, void *msg, ssize_t msg_len)
{
  RRU_CONFIG_msg_t *rru_config_msg = msg;
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)device->priv;

  printf("ORAN: %s\n", __FUNCTION__);

  if (s->last_msg == RAU_tick && s->capabilities_sent == 0) {
    printf("ORAN ctrlrcv RRU_tick received and send capabilities hard coded\n");
    RRU_capabilities_t *cap;
    rru_config_msg->type = RRU_capabilities;
    rru_config_msg->len = sizeof(RRU_CONFIG_msg_t) - MAX_RRU_CONFIG_SIZE + sizeof(RRU_capabilities_t);
    // Fill RRU capabilities (see openair1/PHY/defs_RU.h)
    // For now they are hard coded - try to retreive the params from openari device

    cap = (RRU_capabilities_t *)&rru_config_msg->msg[0];
    cap->FH_fmt = OAI_IF4p5_only;
    cap->num_bands = 1;
    cap->band_list[0] = 78;
    // cap->num_concurrent_bands             = 1; component carriers
    cap->nb_rx[0] = 1; // device->openair0_cfg->rx_num_channels;
    cap->nb_tx[0] = 1; // device->openair0_cfg->tx_num_channels;
    cap->max_pdschReferenceSignalPower[0] = -27;
    cap->max_rxgain[0] = 90;
    cap->N_RB_DL[0] = 106;
    cap->N_RB_UL[0] = 106;

    s->capabilities_sent = 1;

    return rru_config_msg->len;
  }
  if (s->last_msg == RRU_config) {
    printf("Oran RRU_config\n");
    rru_config_msg->type = RRU_config_ok;
  }
  return 0;
}

extern void rx_rf(RU_t *ru, int *frame, int *slot);
void nr_fep_tp(RU_t *ru, int slot);
static void fh_if4p5_south_in(RU_t *ru, int *frame, int *slot)
{
  fd_rfsim_state_t *s = (fd_rfsim_state_t *)ru->ifdevice.priv;
  LOG_D(HW, "fd_rfsim: fh_if4p5_south_in: frame %d, slot %d\n", *frame, *slot);
  // O-DU expects
  // rxDataF here: ru_info.rxdataF = ru->common.rxdataF;
  /// PRACH data here: ru_info.prach_buf = ru->prach_rxsigF[0]; // index: [prach_oca][ant_id]
  s->ru.common.rxdataF = ru->common.rxdataF;

  NR_DL_FRAME_PARMS *fp = s->ru.nr_frame_parms;
  start_meas(&s->ru.rx_fhaul);
  rx_rf(&s->ru, frame, slot);
  int rx_slot_type = nr_slot_select(&s->ru.config, *frame, *slot);
  if (rx_slot_type == NR_UPLINK_SLOT || rx_slot_type == NR_MIXED_SLOT) {
    prach_item_t *prach_id = find_nr_prach(&ru->gNB_list[0]->prach_list, *frame, *slot, SEARCH_EXIST);
    if (prach_id) {
      rx_nr_prach_ru(prach_id, s->ru.common.rxdata, fp, s->ru.N_TA_offset);
    }
    nr_fep_tp(&s->ru, *slot);
    if (!ru->gNB_list[0]->phase_comp) {
      int soffset = (*slot & 3) * fp->symbols_per_slot * fp->ofdm_symbol_size;
      for (int aa = 0; aa < fp->nb_antennas_rx; aa++) {
        const uint max_symb = fp->Ncp == EXTENDED ? 12 : 14;
        for (int sym = 0; sym < max_symb; sym++)
          apply_nr_rotation_symbol_RX(fp,
                                      (c16_t *)s->ru.common.rxdataF[aa] + soffset + sym * fp->ofdm_symbol_size,
                                      fp->symbol_rotation[1],
                                      fp->N_RB_UL,
                                      *slot,
                                      sym);
      }
    }
  }

  RU_proc_t *proc = &ru->proc;
  int slots_per_frame = fp->slots_per_frame;
  proc->tti_rx = *slot;
  proc->frame_rx = *frame;
  proc->tti_tx = (*slot + ru->sl_ahead) % slots_per_frame;
  proc->frame_tx = (*slot > (slots_per_frame - 1 - ru->sl_ahead)) ? (*frame + 1) & 1023 : *frame;
  proc->first_rx = 0;

  stop_meas(&s->ru.rx_fhaul);
}

extern void tx_rf(RU_t *ru, int frame, int slot, uint64_t timestamp);
extern void nr_feptx0(RU_t *ru, int tti_tx, int first_symbol, int num_symbols, int aa);
static void fh_if4p5_south_out(RU_t *ru, int frame, int slot, uint64_t timestamp)
{
  fd_rfsim_state_t *state = (fd_rfsim_state_t *)ru->ifdevice.priv;
  RU_t *ru_lower = &state->ru;
  ru_lower->common.txdataF_BF = ru->common.txdataF_BF;
  start_meas(&ru_lower->tx_fhaul);
  // FD data is available in ru->common.txdataF_BF.
  NR_DL_FRAME_PARMS *fp = ru_lower->nr_frame_parms;
  for (int i = 0; i < ru->nb_tx; i++) {
    if (!ru->gNB_list[0]->phase_comp) {
      apply_nr_rotation_TX(fp,
                           (c16_t *)ru->common.txdataF_BF[i],
                           fp->symbol_rotation[0],
                           slot,
                           fp->N_RB_DL,
                           0,
                           fp->Ncp == EXTENDED ? 12 : 14);
    }
    nr_feptx0(ru_lower, slot, 0, fp->Ncp == EXTENDED ? 12 : 14, i);
  }
  LOG_D(HW, "fd_rfsim: fh_if4p5_south_out: frame %d, slot %d, timestamp %lu\n", frame, slot, timestamp);
  // Assume this function called in order
  static int frame_tx_unwrap = 0;
  static int last_frame = 0;
  if (frame < last_frame) {
    frame_tx_unwrap += 1024;
  }
  last_frame = frame;

  uint64_t timestamp_tx = (frame + frame_tx_unwrap) * fp->samples_per_subframe * 10 + fp->get_samples_slot_timestamp(slot, fp, 0);

  tx_rf(ru_lower, frame, slot, timestamp_tx);
  // ru->proc = ru_lower->proc; // Copy the RU proc structure back
  stop_meas(&ru_lower->tx_fhaul);
}

static void *get_internal_parameter(char *name)
{
  printf("ORAN: %s\n", __FUNCTION__);

  if (!strcmp(name, "fh_if4p5_south_in"))
    return (void *)fh_if4p5_south_in;
  if (!strcmp(name, "fh_if4p5_south_out"))
    return (void *)fh_if4p5_south_out;

  return NULL;
}

int get_tdd_period(uint8_t nb_periods_per_frame)
{
  int tdd_period = 0;
  switch (nb_periods_per_frame) {
    case 20:
      tdd_period = 0; // 10ms/0p5ms
      break;

    case 16:
      tdd_period = 1; // 10ms/0p625ms
      break;

    case 10:
      tdd_period = 2; // 10ms/1ms
      break;

    case 8:
      tdd_period = 3; // 10ms/1p25ms
      break;

    case 5:
      tdd_period = 4; // 10ms/2ms
      break;

    case 4:
      tdd_period = 5; // 10ms/2p5ms
      break;

    case 2:
      tdd_period = 6; // 10ms/5ms
      break;

    case 1:
      tdd_period = 7; // 10ms/10ms
      break;

    default:
      AssertFatal(1 == 0, "Undefined nb_periods_per_frame %d\n", nb_periods_per_frame);
  }
  return tdd_period;
}

typedef struct {
  int scs;
  int freq_range;
  int nr_band;
  int ofdm_offset_divisor;
  uint64_t dl_CarrierFreq;
  uint64_t ul_CarrierFreq;
  int num_rb_dl;
  int num_rb_ul;
  int num_rx_ant;
  int num_tx_ant;
} oru_config;

static void configure_ru_t_for_oru(RU_t *ru, oru_config *oru_cfg)
{
  memset(ru, 0, sizeof(*ru));
  ru->num_gNB = 1;
  ru->nr_frame_parms = calloc_or_fail(1, sizeof(*ru->nr_frame_parms));
  ru->proc.first_rx = 1;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  fp->freq_range = oru_cfg->freq_range;
  fp->nr_band = oru_cfg->nr_band;
  fp->ofdm_offset_divisor = oru_cfg->ofdm_offset_divisor;
  fp->dl_CarrierFreq = oru_cfg->dl_CarrierFreq;
  fp->ul_CarrierFreq = oru_cfg->ul_CarrierFreq;

  nfapi_nr_config_request_scf_t *cfg = &ru->config;
  memset(cfg, 0, sizeof(*cfg));
  cfg->cell_config.frame_duplex_type.value = TDD;
  cfg->ssb_config.scs_common.value = oru_cfg->scs;
  cfg->carrier_config.dl_grid_size[cfg->ssb_config.scs_common.value].value = oru_cfg->num_rb_dl;
  cfg->carrier_config.ul_grid_size[cfg->ssb_config.scs_common.value].value = oru_cfg->num_rb_ul;
  cfg->carrier_config.num_rx_ant.value = oru_cfg->num_rx_ant;
  cfg->carrier_config.num_tx_ant.value = oru_cfg->num_tx_ant;
  cfg->prach_config.prach_sub_c_spacing.value = oru_cfg->scs;

  nr_init_frame_parms(&ru->config, fp);
  init_symbol_rotation(fp);
  init_timeshift_rotation(fp);
  nr_dump_frame_parms(fp);

  ru->if_south = LOCAL_RF;
  ru->function = NGFI_RRU_IF5;
  ru->nb_tx = oru_cfg->num_tx_ant;
  ru->nb_rx = oru_cfg->num_rx_ant;
  nr_phy_init_RU(ru);

  ru->N_TA_offset = set_default_nta_offset(fp->freq_range, fp->samples_per_subframe);
}

static bool configure_fd_rfsim(openair0_device *device, openair0_config_t *openair0_config)
{
  fd_rfsim_state_t *state = calloc_or_fail(1, sizeof(*state));
  mutexinit(state->mutex);
  state->is_started = false;
  state->last_msg = (rru_config_msg_type_t)-1;
  state->capabilities_sent = false;
  device->priv = state;

  RU_t *ru_lower = &state->ru;

  oru_config oru_cfg = {
      .scs = 1,
      .freq_range = FR1,
      .nr_band = openair0_config->nr_band,
      .ofdm_offset_divisor = 8, // Default value
      .dl_CarrierFreq = openair0_config->tx_freq[0],
      .ul_CarrierFreq = openair0_config->rx_freq[0],
      .num_rb_dl = openair0_config->num_rb_dl,
      .num_rb_ul = openair0_config->num_rb_dl,
      .num_rx_ant = openair0_config->rx_num_channels,
      .num_tx_ant = openair0_config->tx_num_channels,
  };

  configure_ru_t_for_oru(ru_lower, &oru_cfg);

  // Setup TDD table
  ru_lower->config.tdd_table.tdd_period.value = get_tdd_period(openair0_config->split7.n_tdd_period);
  NR_DL_FRAME_PARMS *fp = ru_lower->nr_frame_parms;
  ru_lower->config.tdd_table.max_tdd_periodicity_list = calloc_or_fail(fp->slots_per_frame, sizeof(nfapi_nr_max_tdd_periodicity_t));
  for (int i = 0; i < fp->slots_per_frame; i++) {
    ru_lower->config.tdd_table.max_tdd_periodicity_list[i].max_num_of_symbol_per_slot_list =
        calloc_or_fail(14, sizeof(nfapi_nr_max_num_of_symbol_per_slot_t));
    for (int j = 0; j < 14; j++) {
      ru_lower->config.tdd_table.max_tdd_periodicity_list[i].max_num_of_symbol_per_slot_list[j].slot_config.value =
          openair0_config->split7.slot_dirs[i % openair0_config->split7.n_tdd_period].sym_dir[j];
    }
  }

  // Allocate some threads to the threadpool
  ru_lower->threadPool = calloc_or_fail(1, sizeof(*ru_lower->threadPool));
  initFloatingCoresTpool(10, ru_lower->threadPool, false, "RU_lower");

  // Force load RFSimulator as the RF device
  memcpy(&ru_lower->openair0_cfg, openair0_config, sizeof(*openair0_config));
  int ret = openair0_load(&ru_lower->rfdevice, "rfsimulator", &ru_lower->openair0_cfg, NULL);
  AssertFatal(ret == 0, "Failed to load openair0 device\n");

  // These are fhi_72 specific parameters, currently unused.
  // verify oran section is present: we don't have a list but the below returns
  // numelt > 0 if the block is there
  paramlist_def_t pl = {0};
  strncpy(pl.listname, CONFIG_STRING_ORAN, sizeof(pl.listname) - 1);
  config_getlist(config_get_if(), &pl, NULL, 0, /* prefix */ NULL);
  if (pl.numelt == 0) {
    printf("Configuration section \"%s\" not present: cannot initialize fd_rfsim!\n", CONFIG_STRING_ORAN);
    return false;
  }

  paramdef_t fhip[] = ORAN_GLOBALPARAMS_DESC;
  checkedparam_t fhip_CheckParams[] = ORAN_GLOBALPARAMS_CHECK_DESC;
  static_assert(sizeofArray(fhip) == sizeofArray(fhip_CheckParams), "fhip and fhip_CheckParams should have the same size");
  int nump = sizeofArray(fhip);
  config_set_checkfunctions(fhip, fhip_CheckParams, nump);
  ret = config_get(config_get_if(), fhip, nump, CONFIG_STRING_ORAN);
  if (ret <= 0) {
    printf("problem reading section \"%s\"\n", CONFIG_STRING_ORAN);
    return false;
  }

  paramdef_t FHconfigs[] = ORAN_FH_DESC;
  paramlist_def_t FH_ConfigList = {CONFIG_STRING_ORAN_FH};
  char aprefix[MAX_OPTNAME_SIZE] = {0};
  sprintf(aprefix, "%s", CONFIG_STRING_ORAN);
  const int nfh = sizeofArray(FHconfigs);
  config_getlist(config_get_if(), &FH_ConfigList, FHconfigs, nfh, aprefix);

  return true;
}

__attribute__((__visibility__("default"))) int transport_init(openair0_device *device,
                                                              openair0_config_t *openair0_cfg,
                                                              eth_params_t *eth_params)
{
  bool ret = configure_fd_rfsim(device, openair0_cfg);
  AssertFatal(ret, "Failed to configure fd_rfsim");

  device->Mod_id = 0;
  device->transp_type = ETHERNET_TP;
  device->trx_start_func = trx_start;
  device->trx_reset_stats_func = trx_reset_stats;
  device->trx_end_func = trx_end;
  device->trx_stop_func = trx_stop;
  device->trx_set_freq_func = trx_set_freq;
  device->trx_set_gains_func = trx_set_gains;
  device->trx_write_func = trx_write_raw;
  device->trx_read_func = trx_read_raw;
  device->trx_ctlsend_func = trx_ctlsend;
  device->trx_ctlrecv_func = trx_ctlrecv;
  device->get_internal_parameter = get_internal_parameter;
  device->openair0_cfg = openair0_cfg;

  return 0;
}

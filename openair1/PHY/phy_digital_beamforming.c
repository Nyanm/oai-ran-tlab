/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "phy_digital_beamforming.h"
#include "defs_gNB.h"
#include "pthread.h"

/// @brief Holds grid info for future slots for local beamforming.
static struct grid_slots_head rx_grid_list = SLIST_HEAD_INITIALIZER(rx_grid_list);
static pthread_mutex_t list_lock = PTHREAD_MUTEX_INITIALIZER;

/* Get current Rx slot's BF info. */
struct nr_grid_slot *get_grid_slot(const uint32_t frame, const uint32_t slot)
{
  pthread_mutex_lock(&list_lock);
  struct grid_slot_entry *g = NULL;
  struct grid_slots_head *head = &rx_grid_list;
  SLIST_FOREACH(g, head, next)
  {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      // Found existing entry. Return
      break;
    }
  }
  pthread_mutex_unlock(&list_lock);
  return &g->grid;
}

/* Remove grid entry from slot list. */
void remove_grid_slot(const uint32_t frame, const uint32_t slot)
{
  pthread_mutex_lock(&list_lock);
  struct grid_slot_entry *g = NULL;
  struct grid_slots_head *head = &rx_grid_list;
  SLIST_FOREACH(g, head, next)
  {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      SLIST_REMOVE(head, g, grid_slot_entry, next);
      free(g);
      break;
    }
  }
  pthread_mutex_unlock(&list_lock);
}

/* Add new Rx BF info to list. */
struct nr_grid_slot *add_grid_slot_entry(const uint32_t frame, const uint32_t slot)
{
  pthread_mutex_lock(&list_lock);
  struct grid_slot_entry *g = NULL;
  struct grid_slots_head *head = &rx_grid_list;
  struct grid_slot_entry *prev = head->slh_first;
  SLIST_FOREACH(g, head, next)
  {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      // Found existing entry. Exit with error.
      DevAssert(0);
    }
  }

  // No entry exists. Create one and add to it.
  // Create one.
  struct grid_slot_entry *new = calloc(1, sizeof(*new));
  new->grid.frame = frame;
  new->grid.slot = slot;

  // List empty. Add to head
  if (!prev)
    SLIST_INSERT_HEAD(head, new, next);
  // Add to end of list.
  else
    SLIST_INSERT_AFTER(prev, new, next);

  pthread_mutex_unlock(&list_lock);
  return &new->grid;
}

// #define DBF_DEBUG

/* Check for the first port's beam ID for MSB value. We trust MAC to send same MSB in all PDUs. */
static bool check_low_phy_bf(const struct nr_grid *nrg)
{
  const uint16_t beam_id = nrg[0].grid_info[0].beam_id;
  return (IS_BIT_SET(beam_id, 15));
}

/* Add PUSCH sched info for BF. */
static int fill_pusch_grid_info(const uint32_t frame, const uint32_t slot, const nfapi_nr_pusch_pdu_t *pdu, struct nr_grid_slot *nrg)
{
  int ret = 0;
  uint16_t stream_idx = 0;
  /* According to FAPI, when dig_bf_interface = 0 the BF is straight wire which means baseband and logical ports
  are mapped one to one. So, there could be baseband ports not mapped to any logical ports. */
  bool no_bf = pdu->beamforming.dig_bf_interface == 0;
  uint16_t beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->rb_start + pdu->bwp_start,
                             .num_prb = pdu->rb_size,
                             .start_symbol = pdu->start_symbol_index,
                             .num_symbols = pdu->nr_of_symbols,
                             .is_straightwire_bf = no_bf};

    uint16_t port_id = pdu->param_v4.spatialStreamIndices[stream_idx];
    struct nr_grid *nrg_p = nrg->grid + port_id;
    nrg_p->grid_info[nrg_p->num_sections++] = info;

    LOG_D(PHY, "%d.%d Adding PUSCH grid info for port %d\n", frame, slot, port_id);
    stream_idx++;
    beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  } while (stream_idx < pdu->param_v4.numSpatialStreamIndices);
  return ret;
}

/* Add PUSCH sched info for BF. */
static int fill_pucch_grid_info(const uint32_t frame, const uint32_t slot, const nfapi_nr_pucch_pdu_t *pdu, struct nr_grid_slot *nrg)
{
  int ret = 0;
  uint32_t stream_idx = 0;
  bool no_bf = pdu->beamforming.dig_bf_interface == 0;
  uint16_t beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->prb_start + pdu->bwp_start,
                             .num_prb = pdu->prb_size,
                             .start_symbol = pdu->start_symbol_index,
                             .num_symbols = pdu->nr_of_symbols,
                             .is_straightwire_bf = no_bf};

    uint16_t port_id = pdu->param_v4.spatialStreamIndices[stream_idx];
    struct nr_grid *nrg_p = nrg->grid + port_id;
    nrg_p->grid_info[nrg_p->num_sections++] = info;

    LOG_D(PHY, "%d.%d Adding PUCCH grid info for port %d\n", frame, slot, port_id);
    stream_idx++;
    beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  } while (stream_idx < pdu->param_v4.numSpatialStreamIndices);
  return ret;
}

/* Add PRACH sched info for BF. */
static int fill_prach_grid_info(const uint32_t frame, const uint32_t slot, const nfapi_nr_prach_pdu_t *pdu, const RU_t *ru, struct nr_grid_slot *nrg)
{
  int ret = 0;
  uint32_t stream_idx = 0;
  bool no_bf = pdu->beamforming.dig_bf_interface == 0;
  uint16_t beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  do {
    const int fdm_idx = pdu->num_ra;
    const int start_re = ru->config.prach_config.num_prach_fd_occasions_list[fdm_idx].k1.value;
    const int start_rb = start_re / NR_NB_SC_PER_RB;
    // TODO: configure for long format;
    if (ru->config.prach_config.prach_sequence_length.value != 1) {
      LOG_E(PHY, "Beamforming not implemented for long format\n");
      return -1;
    }
    const int num_rb = NR_PRACH_SEQ_LEN_S / NR_NB_SC_PER_RB;
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = start_rb,
                             .num_prb = num_rb,
                             .start_symbol = pdu->prach_start_symbol,
                             .num_symbols = NR_SYMBOLS_PER_SLOT - pdu->prach_start_symbol,
                             .is_straightwire_bf = no_bf};
    // For now lets beamform till the end of PRACH slot.

    uint16_t port_id = pdu->param_v4.spatialStreamIndices[stream_idx];
    struct nr_grid *nrg_p = nrg->grid + port_id;
    nrg_p->grid_info[nrg_p->num_sections++] = info;

    LOG_D(PHY, "%d.%d Adding PRACH grid info for port %d\n", frame, slot, port_id);
    stream_idx++;
    beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  } while (stream_idx < pdu->param_v4.numSpatialStreamIndices);
  return ret;
}

/* Add SRS sched info for BF. */
static int fill_srs_grid_info(const uint32_t frame, const uint32_t slot, const nfapi_nr_srs_pdu_t *pdu, struct nr_grid_slot *nrg)
{
  /*
    Beamforming of SRS signals is very unlikely to happen. If the gNB decides to BF SRS, then the gird info
    should be updated with reMask to exactly specify REs and its beams ID.
  */
  int ret = 0;
  uint32_t stream_idx = 0;
  bool no_bf = pdu->beamforming.dig_bf_interface == 0;
  uint16_t beam_id = no_bf ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->bwp_start,
                             .num_prb = pdu->bwp_size,
                             .start_symbol = pdu->time_start_position,
                             .num_symbols = 1 << pdu->num_symbols,
                             .is_straightwire_bf = no_bf};

    uint16_t port_id = pdu->srs_parameters_v4.Ul_spatial_stream_ports[stream_idx];
    struct nr_grid *nrg_p = nrg->grid + port_id;
    nrg_p->grid_info[nrg_p->num_sections++] = info;

    LOG_D(PHY, "%d.%d Adding SRS grid info for port %d\n", frame, slot, port_id);
    stream_idx++;
    beam_id = pdu->beamforming.prgs_list[0].dig_bf_interface_list[stream_idx].beam_idx;
  } while (stream_idx < pdu->srs_parameters_v4.num_ul_spatial_streams_ports);
  return ret;
}

/* Populate grid info for current slot. */
void fill_rx_grid_info(const RU_t *ru,
                       const uint32_t frame,
                       const uint32_t slot,
                       const nfapi_nr_ul_tti_request_number_of_pdus_t *ul_pdu,
                       struct nr_grid_slot *nrg)
{
  int ret = 0;
  switch (ul_pdu->pdu_type) {
    case NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE:
      ret = fill_pusch_grid_info(frame, slot, &ul_pdu->pusch_pdu, nrg);
      break;

    case NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE:
      ret = fill_pucch_grid_info(frame, slot, &ul_pdu->pucch_pdu, nrg);
      break;

    case NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE:
      ret = fill_prach_grid_info(frame, slot, &ul_pdu->prach_pdu, ru, nrg);
      break;

    case NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE:
      ret = fill_srs_grid_info(frame, slot, &ul_pdu->srs_pdu, nrg);
      break;
  }
  if (ret < 0) {
    LOG_E(PHY, "Error configuring UL slots for beamforming\n");
    return;
  }
}

/*
   Based on FAPI's beam id MSB, decide if BF is done locally or to pass beam id to remote RU.
   If BF is done on remote RU (7.2 split), then fill the xran buffer with BF info immediately.
   TODO:
   If BF is to be performed locally, then prepare the weights to be applied for each baseband
   antenna port and PRB group. When the time comes, these weights are applied to the samples.
*/
void send_rx_grid_info(RU_t *ru, struct nr_grid_slot *nrg)
{
  const bool low_phy_bf = check_low_phy_bf(nrg->grid);
  if (low_phy_bf) {
    // Call xran function to fill cplane buffer with beam and PRB info for this UL slot.
    if (ru->fh_south_out_ctrl)
      ru->fh_south_out_ctrl(ru, nrg->frame, nrg->slot, 0, nrg->grid);
  } else {
    // Local Rx beamforming is done when the slot is received. Store the grid info.
    struct nr_grid_slot *new = add_grid_slot_entry(nrg->frame, nrg->slot);
    *new = *nrg;
  }
}

#define MADD_SHIFT 15

/* Apply BF weights. */
void apply_beamforming(const nfapi_nr_dbt_pdu_t *dbt,
                       const NR_DL_FRAME_PARMS *fp,
                       const uint16_t num_bb,
                       c16_t **bb,
                       struct nr_grid *nrg,
                       bool is_tx)
{
  DevAssert(nrg);
  const uint8_t num_logical_ports = fp->nb_antennas_tx;
  // Loop over logical ports
  for (uint_fast8_t l = 0; l < num_logical_ports; l++) {
    // Loop over section
    for (uint_fast16_t sec = 0; sec < nrg[l].num_sections; sec++) {
      const struct grid_info *g = nrg[l].grid_info + sec;

      const uint16_t beam_id = g->beam_id & 0x7fff;
      const nfapi_nr_txru_t *cur_wt_vec;
      if (!g->is_straightwire_bf) {
        DevAssert(dbt);
        AssertFatal(beam_id <= dbt->num_dig_beams, "Beam id %d exceeds DBT size\n", beam_id);
        AssertFatal(beam_id == dbt->dig_beam_list[beam_id].beam_idx, "Beam id not consistent with DBT\n");
        DevAssert(num_bb == dbt->num_txrus);
        cur_wt_vec = dbt->dig_beam_list[beam_id].txru_list;
      }

#ifdef DBF_DEBUG
      LOG_I(PHY,
            "sec %lu, start sym %u, num sym %u, start rb %u, num rb %u, port %u\n",
            sec,
            g->start_symbol,
            g->num_symbols,
            g->start_prb,
            g->num_prb,
            l);
#endif

      for (uint_fast16_t b = 0; b < num_bb; b++) {
        // Straightwire BF. Logical and baseband ports are mapped one to one
        if (g->is_straightwire_bf && (l != b))
          break;
        for (uint_fast8_t sym = g->start_symbol; sym < g->start_symbol + g->num_symbols; sym++) {
          uint32_t txdataF_offset = sym * fp->ofdm_symbol_size + g->start_prb * NR_NB_SC_PER_RB;
          c16_t *cur_bb = bb[b] + txdataF_offset;
          c16_t *cur_log = nrg[l].dataF + txdataF_offset;

          const int num_re = g->num_prb * NR_NB_SC_PER_RB;
          const c16_t wt = (g->is_straightwire_bf) ? (c16_t){.r = INT16_MAX, .i = INT16_MAX} : *(c16_t *)(cur_wt_vec + b);
          if (is_tx)
            multadd_cpx_vector_cpx_scalar(cur_log, wt, cur_bb, num_re, MADD_SHIFT);
          else
            multadd_cpx_vector_cpx_scalar(cur_bb, wt, cur_log, num_re, MADD_SHIFT);
        }
      }
    }
  }
}

void tx_beamforming_if(RU_t *ru)
{
  struct nr_grid *nrg = ru->common.ru_tx_grid;

  const bool low_phy_bf = check_low_phy_bf(nrg);

  const NR_DL_FRAME_PARMS *fp = &ru->gNB_list[0]->frame_parms;

  const int num_logical_ports = fp->nb_antennas_rx;
  // No allocations. Set memory to 0 and exit.
  if (!nrg->num_sections) {
    for (int l = 0; l < num_logical_ports; l++)
      memset(ru->common.txdataF_BF[l], 0, sizeof(c16_t) * fp->samples_per_slot_wCP);
    return;
  }

  if (low_phy_bf) {
    /* LoPHY beamforming. Copy all logical port buffer to RU. */
    for (int l = 0; l < num_logical_ports; l++)
      memcpy(ru->common.txdataF_BF[l],
             ru->gNB_list[0]->common_vars.tx_grid_info[l].dataF,
             sizeof(c16_t) * fp->samples_per_slot_wCP);
  } else {
    PHY_VARS_gNB *gnb = ru->gNB_list[0];
    apply_beamforming(&gnb->gNB_config.dbt_config,
                      &gnb->frame_parms,
                      ru->nb_rx,
                      (c16_t **)ru->common.txdataF_BF,
                      ru->common.ru_tx_grid,
                      true);
  }
}

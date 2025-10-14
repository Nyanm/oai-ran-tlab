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

#include "phy_digital_beamforming.h"
#include "defs_gNB.h"

/* Get current Rx slot's BF info. */
struct grid_slot_entry *get_grid_slot(struct grid_slots_head *head, const uint32_t frame, const uint32_t slot)
{
  struct grid_slot_entry *g = NULL;
  SLIST_FOREACH(g, head, next) {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      // Found existing entry. Return
      break;
    }
  }
  return g;
}

/* Remove grid entry from slot list. */
void remove_grid_slot(struct grid_slots_head *head, const uint32_t frame, const uint32_t slot)
{
  struct grid_slot_entry *g = NULL;
  SLIST_FOREACH(g, head, next) {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      SLIST_REMOVE(head, g, grid_slot_entry, next);
      free(g);
      break;
    }
  }
}

/* Add new Rx BF info to list. */
static int add_grid_slot_entry(struct grid_slots_head *head, const uint32_t frame, const uint32_t slot, const uint32_t port_id, const struct grid_info *info)
{
  int ret = -1;
  struct grid_slot_entry *g = NULL;
  struct grid_slot_entry *prev = head->slh_first;
  SLIST_FOREACH(g, head, next) {
    if (g->grid.frame == frame && g->grid.slot == slot) {
      // Found existing entry. Add to entry.
      struct nr_grid *nrg = &g->grid.grid[port_id];
      nrg->grid_info[nrg->num_sections++] = *info;
      ret = 0;
      break;
    }
    prev = g;
  }

  // No entry exists. Create one and add to it.
  if (!g) {
    // Create one.
    struct grid_slot_entry *new = calloc(1, sizeof(*new));
    ret = (new == NULL) ? -1 : 0;
    new->grid.frame = frame;
    new->grid.slot = slot;
    struct nr_grid *nrg = &new->grid.grid[port_id];
    nrg->grid_info[nrg->num_sections++] = *info;

    // List empty. Add to head
    if (!prev)
      SLIST_INSERT_HEAD(head, new, next);
    // Add to end of list.
    else
      SLIST_INSERT_AFTER(prev, new, next);
  }

  return ret;
}

/* Add PUSCH sched info for BF. */
static int fill_pusch_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_pusch_pdu_t *pdu)
{
  /*
    According to FAPI specifications, MAC should decide on the number of spatial streams (dig_bf_interface)
    from the DU and RU configuration.
  */
  int ret = 0;
  uint32_t port_id = 0;
  /* According to FAPI, when dig_bf_interface = 0 the BF is straight wire which means baseband and logical ports
  are mapped one to one. So, there could be baseband ports not mapped to any logical ports. */
  uint16_t beam_id =
      (pdu->beamforming.dig_bf_interface == 0) ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->rb_start + pdu->bwp_start,
                             .num_prb = pdu->rb_size,
                             .start_symbol = pdu->start_symbol_index,
                             .num_symbols = pdu->nr_of_symbols};

    /* It is unclear how the logical port id is signaled in case of MU-MIMO. When scheduling MU-MIMO, only a
    subset of logical antenna ports are used for a UE. But FAPI says the logical port index start from 0.
    This is a problem only when unitary precoding (no precoding) is used. */
    int cur_ret = add_grid_slot_entry(&ru->common.rx_grid, frame, slot, port_id, &info);
    if (cur_ret != 0)
      LOG_E(PHY, " %d.%d Error copying PUSCH grid info to RU\n", frame, slot);

    LOG_D(PHY, "%d.%d Adding PUSCH grid info for port %d\n", frame, slot, port_id);
    ret |= cur_ret;
    port_id++;
    beam_id = pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  } while (port_id < pdu->beamforming.dig_bf_interface);
  return ret;
}

/* Add PUSCH sched info for BF. */
static int fill_pucch_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_pucch_pdu_t *pdu)
{
  int ret = 0;
  uint32_t port_id = 0;
  uint16_t beam_id =
      (pdu->beamforming.dig_bf_interface == 0) ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->prb_start + pdu->bwp_start,
                             .num_prb = pdu->prb_size,
                             .start_symbol = pdu->start_symbol_index,
                             .num_symbols = pdu->nr_of_symbols};

    int cur_ret = add_grid_slot_entry(&ru->common.rx_grid, frame, slot, port_id, &info);
    if (cur_ret != 0)
      LOG_E(PHY, " %d.%d Error copying PUSCH grid info to RU\n", frame, slot);

    LOG_D(PHY, "%d.%d Adding PUCCH grid info for port %d\n", frame, slot, port_id);
    ret |= cur_ret;
    port_id++;
    beam_id = pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  } while (port_id < pdu->beamforming.dig_bf_interface);
  return ret;
}

/* Add PRACH sched info for BF. */
static int fill_prach_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_prach_pdu_t *pdu)
{
  int ret = 0;
  /* Check if PRACH FFT is done in DU (split 8). Then skip storing because beamforming is done separately for PRACH. */
  if (ru->feprx)
    return ret;

  uint32_t port_id = 0;
  uint16_t beam_id =
      (pdu->beamforming.dig_bf_interface == 0) ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  do {
    const int fdm_idx = pdu->num_ra;
    const int start_re = ru->config.prach_config.num_prach_fd_occasions_list[fdm_idx].k1.value;
    const int offset = start_re - ru->gNB_list[0]->frame_parms.first_carrier_offset;
    const int start_rb = (offset / NR_NB_SC_PER_RB) + (offset % NR_NB_SC_PER_RB != 0);
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
                             .num_symbols = NR_NUMBER_OF_SYMBOLS_PER_SLOT - pdu->prach_start_symbol};
    // For now lets beamform till the end of PRACH slot.

    int cur_ret = add_grid_slot_entry(&ru->common.rx_grid, frame, slot, port_id, &info);
    if (cur_ret != 0)
      LOG_E(PHY, " %d.%d Error copying PUSCH grid info to RU\n", frame, slot);

    LOG_D(PHY, "%d.%d Adding PRACH grid info for port %d\n", frame, slot, port_id);
    ret |= cur_ret;
    port_id++;
    beam_id = pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  } while (port_id < pdu->beamforming.dig_bf_interface);
  return ret;
}

/* Add SRS sched info for BF. */
static int fill_srs_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_srs_pdu_t *pdu)
{
  /*
    Beamforming of SRS signals is very unlikely to happen. If the gNB decides to BF SRS, then the gird info
    should be updated with reMask to exactly specify which REs and its beams ID.
  */
  int ret = 0;
  uint32_t port_id = 0;
  uint16_t beam_id = (pdu->beamforming.dig_bf_interface == 0) ? 0 : pdu->beamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx;
  do {
    struct grid_info info = {.beam_id = beam_id,
                             .start_prb = pdu->bwp_start,
                             .num_prb = pdu->bwp_size,
                             .start_symbol = pdu->time_start_position,
                             .num_symbols = 1 << pdu->num_symbols};

    int cur_ret = add_grid_slot_entry(&ru->common.rx_grid, frame, slot, port_id, &info);
    if (cur_ret != 0)
      LOG_E(PHY, " %d.%d Error copying PUSCH grid info to RU\n", frame, slot);

    LOG_D(PHY, "%d.%d Adding SRS grid info for port %d\n", frame, slot, port_id);
    ret |= cur_ret;
    port_id++;
    beam_id = pdu->beamforming.prgs_list[0].dig_bf_interface_list[port_id].beam_idx;
  } while (port_id < pdu->beamforming.dig_bf_interface);
  return ret;
}

/* Add current slot's UL sched info to BF info list. */
void fill_rx_grid_info(RU_t *ru, const uint32_t frame, const uint32_t slot, const nfapi_nr_ul_tti_request_number_of_pdus_t *ul_pdu)
{
  int ret = 0;
  switch (ul_pdu->pdu_type) {
    case NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE:
      ret = fill_pusch_grid_info(ru, frame, slot, &ul_pdu->pusch_pdu);
      break;

    case NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE:
      ret = fill_pucch_grid_info(ru, frame, slot, &ul_pdu->pucch_pdu);
      break;

    case NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE:
      ret = fill_prach_grid_info(ru, frame, slot, &ul_pdu->prach_pdu);
      break;

    case NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE:
      ret = fill_srs_grid_info(ru, frame, slot, &ul_pdu->srs_pdu);
      break;
  }
  if (ret < 0)
    LOG_E(PHY, "Error configuring UL slots for beamforming\n");
}

/* Check for the first port's beam ID for MSB value. We trust MAC to send same MSB in all PDUs. */
bool check_low_phy_bf(struct nr_grid *nrg)
{
  const uint16_t beam_id = nrg[0].grid_info[0].beam_id;
  return (IS_BIT_SET(beam_id, 15));
}

/* Process current slot's BF info.
   Based on FAPI's beam id MSB, decide if BF is done locally or to pass beam id to remote RU.
   If BF is to be performed locally, then prepare the weights to be applied for each baseband
   antenna port and PRB group. When the time comes, these weights are applied to the samples.
   If BF is done on remote RU (7.2 split), then fill the xran buffer with BF info immediately.*/
void process_rx_grid_info_bf(RU_t *ru, const uint32_t frame, const uint32_t slot)
{
  struct grid_slot_entry *g = get_grid_slot(&ru->common.rx_grid, frame, slot);

  if (!g)
    return;

  struct nr_grid *nrg = g->grid.grid;
  const bool low_phy_bf = check_low_phy_bf(nrg);

  if (low_phy_bf) {
    // Call xran function to fill cplane buffer with beam and PRB info for this UL slot.
    if (ru->fh_south_out_ctrl)
      ru->fh_south_out_ctrl(ru, frame, slot, nrg);
    return;
  }

  // Local BF. Do nothing here. Beam weights should be applied when samples are received.
  /* TODO: Allocate memory for holding weights matrix of size MxP for M baseband ports
      and P logical ports. The MAC has to pre-configure PHY with beam table via TLV 0x1043. */
  return;
}

/* The following implementation should also work for DAS. The beam wieghts has to be generated
   accordingly. For eg., a DAS with total 8 baseband ports and 2 sectors (4 each), the weight
   for beam id 1 can be [x, x, x, x, 0, 0, 0, 0]. The MAC has to have knowledge of the DAS to
   assign the right beams. */

/* Apply BF weights if local BF is configured by MAC. */
void apply_rx_beamforming(RU_t *ru, const uint32_t frame, const uint32_t slot)
{
  /* The gNB uses RBs that are not allocated to estimate noise power. So here the radio antenna port's
  freq domain samples are entirely copied to logical port buffer so the noise can be estimated. Later,
  the implementation can be changed to estimate the noise power from PUSCH DMRS which will eleminate the
  need to copy over unused PRBs in the slot.

  This is equivalent to no beamforming. */
  const NR_DL_FRAME_PARMS *fp = &ru->gNB_list[0]->frame_parms;
  const int num_logical_ports = ru->gNB_list[0]->frame_parms.nb_antennas_rx;
  const int slot_offset = (slot % RU_RX_SLOT_DEPTH) * fp->samples_per_slot_wCP;
  for (int l = 0; l < num_logical_ports; l++) {
    memcpy(ru->gNB_list[0]->common_vars.rxdataF[l] + slot_offset,
           ru->common.rxdataF[l] + slot_offset,
           sizeof(c16_t) * fp->samples_per_slot_wCP);
  }

  struct grid_slot_entry *g = get_grid_slot(&ru->common.rx_grid, frame, slot);

  if (!g)
    return;

  struct nr_grid *nrg = g->grid.grid;
  const bool low_phy_bf = check_low_phy_bf(nrg);

  // BF already applied at RU. Exit.
  if (low_phy_bf)
    return;

  // Loop over logical ports
  for (int l = 0; l < num_logical_ports; l++) {

    // Loop over section
    for (int sec = 0; sec < nrg->num_sections; sec++){
      struct grid_info *g = nrg->grid_info + sec;
      const int beam_id = g->beam_id & 0x7fff;

      if (beam_id == 0)
        continue;

      AssertFatal(beam_id == 0, "Rx beamforming not implemented yet\n");

      // Unity weights for each baseband port
      for (int b = 0; b < ru->nb_rx; b++) {
        // TODO: use SIMD. Combine with phase compensation.

        for (int sym = g->start_symbol; sym < g->start_symbol + g->num_symbols; sym++) {
          const uint32_t rxdataF_offset = slot_offset + sym * fp->ofdm_symbol_size;
          c16_t *rxdataF_BF_re = ru->gNB_list[0]->common_vars.rxdataF[l] + rxdataF_offset;
          c16_t *rxdataF_re = (c16_t *)ru->common.rxdataF[b] + rxdataF_offset;

          for (int r = g->start_prb; r < g->start_prb + g->num_prb; r++) {
            uint32_t k_offset = fp->first_carrier_offset + r * NR_NB_SC_PER_RB;

            // Send data from this logical port to all baseband ports.
            for (int re = 0; re < NR_NB_SC_PER_RB; re++) {
              k_offset += re;
              k_offset %= fp->ofdm_symbol_size;
              rxdataF_BF_re[k_offset] = c16add(rxdataF_BF_re[k_offset], rxdataF_re[k_offset]);
            }
          }
        }
      }
    }
    nrg++;
  }

  return;
}

/* Apply BF weights to DL slots. */
void apply_tx_beamforming(RU_t *ru, const uint32_t frame, const uint32_t slot)
{
  struct nr_grid *nrg = ru->common.ru_tx_grid;
  const bool low_phy_bf = check_low_phy_bf(nrg);

  const NR_DL_FRAME_PARMS *fp = &ru->gNB_list[0]->frame_parms;
  memset(ru->common.txdataF_BF[0], 0, sizeof(c16_t) * ALNARS_64_16(fp->N_RB_DL * NR_NB_SC_PER_RB) * 14);

  const int num_logical_ports = ru->gNB_list[0]->frame_parms.nb_antennas_tx;

  if (low_phy_bf) {
    /* LoPHY beamforming. Copy all logical port buffer to RU. */
    for (int l = 0; l < num_logical_ports; l++)
      memcpy(ru->common.txdataF_BF[l],
            ru->gNB_list[0]->common_vars.tx_grid_info[l].dataF,
            sizeof(c16_t) * ALNARS_64_16(fp->N_RB_DL * NR_NB_SC_PER_RB) * 14);
  } else {
    // Loop over logical ports
    for (int l = 0; l < num_logical_ports; l++) {

      // Loop over section
      for (int sec = 0; sec < nrg->num_sections; sec++){
        struct grid_info *g = nrg->grid_info + sec;
        const int beam_id = g->beam_id & 0x7fff;
        LOG_D(PHY,
              "sec %d, start sym %d, num sym %d, start rb %d, num rb %d, port %d\n",
              sec,
              g->start_symbol,
              g->num_symbols,
              g->start_prb,
              g->num_prb,
              l);

        // Unity weights for each baseband port
        for (int b = 0; b < ru->nb_tx; b++) {
          // TODO: use SIMD
          const int symb_size = fp->N_RB_DL * NR_NB_SC_PER_RB;
          const int symb_buf_size = ALNARS_64_16(symb_size);

          for (int sym = g->start_symbol; sym < g->start_symbol + g->num_symbols; sym++) {
            uint32_t txdataF_offset = sym * symb_buf_size;
            c16_t *txdataF_BF_re = (c16_t *)ru->common.txdataF_BF[b] + txdataF_offset;
            c16_t *txdataF_re = ru->gNB_list[0]->common_vars.tx_grid_info[l].dataF + txdataF_offset;

            // No beamforming or boresight beam
            if (beam_id == 0) {
              if (b != l)
                break;
              const uint16_t start_re = g->start_prb * NR_NB_SC_PER_RB;
              const uint16_t num_re = g->num_prb * NR_NB_SC_PER_RB;
              memcpy(txdataF_BF_re + start_re, txdataF_re + start_re, num_re * sizeof(c16_t));
            } else {
              // Beamforming
              AssertFatal(0, "Tx beamforming not implemented yet\n");
              for (int r = g->start_prb; r < g->start_prb + g->num_prb; r++) {
                uint32_t k_offset = r * NR_NB_SC_PER_RB;

                // Send data from this logical port to all baseband ports.
                for (int re = 0; re < NR_NB_SC_PER_RB; re++) {
                  // TODO: rotate by beam weights
                  txdataF_BF_re[k_offset] = c16add(txdataF_BF_re[k_offset], txdataF_re[k_offset]);
                  k_offset++;
                }
              }
            }
          }
        }
      }
      nrg++;
    }
  }
  return;
}
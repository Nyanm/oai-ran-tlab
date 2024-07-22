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

#ifndef __NR_ESTIMATION_DEFS__H__
#define __NR_ESTIMATION_DEFS__H__


#include "PHY/defs_nr_UE.h"

/** @addtogroup _PHY_PARAMETER_ESTIMATION_BLOCKS_
 * @{
 */

/* A function to perform the channel estimation of DL PRS signal */
int nr_prs_channel_estimation(uint8_t gNB_id,
                              uint8_t rsc_id,
                              uint8_t rep_num,
                              PHY_VARS_NR_UE *ue,
                              const UE_nr_rxtx_proc_t *proc,
                              NR_DL_FRAME_PARMS *frame_params,
                              c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP]);

/* Generic function to find the peak of channel estimation buffer */
void peak_estimator(c16_t *buffer, int32_t buf_len, int32_t *peak_idx, int32_t *peak_val, int32_t mean_val);

/*!
\brief This function performs channel estimation including frequency and temporal interpolation
*/
void nr_pdcch_channel_estimation(const PHY_VARS_NR_UE *ue,
                                 int nb_rb,
                                 int rb_offset,
                                 int dmrs_ref,
                                 uint16_t first_carrier_offset,
                                 uint16_t BWPStart,
                                 int32_t pdcch_est_size,
                                 c16_t pdcch_dl_ch_estimates[][pdcch_est_size],
                                 c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ue->frame_parms.ofdm_symbol_size],
                                 c16_t *pilot);

c32_t nr_pbch_dmrs_correlation(const NR_DL_FRAME_PARMS *fp,
                               const UE_nr_rxtx_proc_t *proc,
                               const int symbol,
                               const int dmrss,
                               const int Nid_cell,
                               const int ssb_start_subcarrier,
                               const uint32_t nr_gold_pbch[NR_PBCH_DMRS_LENGTH_DWORD],
                               const c16_t rxdataF[fp->nb_antennas_rx][fp->ofdm_symbol_size]);

int nr_pbch_channel_estimation(const NR_DL_FRAME_PARMS *fp,
                               const sl_nr_ue_phy_params_t *sl_phy_params,
                               c16_t dl_ch_estimates[fp->ofdm_symbol_size],
                               const UE_nr_rxtx_proc_t *proc,
                               int dmrss,
                               uint ssb_index,
                               uint n_hf,
                               int ssb_start_subcarrier,
                               const c16_t rxdataF[fp->ofdm_symbol_size],
                               bool sidelink,
                               uint Nid);

void nr_pdsch_channel_estimation(PHY_VARS_NR_UE *ue,
                                 const UE_nr_rxtx_proc_t *proc,
                                 const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch,
                                 const freq_alloc_bitmap_t *freq_alloc,
                                 unsigned short p,
                                 unsigned char symbol,
                                 c16_t dl_ch_estimates[ue->frame_parms.ofdm_symbol_size],
                                 const c16_t rxdataF[ue->frame_parms.ofdm_symbol_size],
                                 uint32_t *nvar);

int nr_adjust_synch_ue(const NR_DL_FRAME_PARMS *frame_parms,
                       PHY_VARS_NR_UE *ue,
                       const c16_t dl_ch_estimates_time[][frame_parms->ofdm_symbol_size],
                       uint8_t frame,
                       uint8_t slot,
                       short coef);

void nr_ue_measurements(PHY_VARS_NR_UE *ue,
                        const UE_nr_rxtx_proc_t *proc,
                        int number_rbs,
                        uint32_t pdsch_est_size,
                        int32_t dl_ch_estimates[][pdsch_est_size]);

uint32_t nr_ue_calculate_ssb_rsrp(const NR_DL_FRAME_PARMS *fp,
                                  const UE_nr_rxtx_proc_t *proc,
                                  const c16_t rxdataF[][fp->samples_per_slot_wCP],
                                  int ssb_start_subcarrier);

void nr_ue_ssb_rsrp_measurements(PHY_VARS_NR_UE *ue,
                                 const int ssb_index,
                                 const UE_nr_rxtx_proc_t *proc,
                                 const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ue->frame_parms.ofdm_symbol_size]);

void nr_ue_rrc_measurements(PHY_VARS_NR_UE *ue,
                            const UE_nr_rxtx_proc_t *proc,
                            const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ue->frame_parms.ofdm_symbol_size]);

void phy_adjust_gain_nr(PHY_VARS_NR_UE *ue,
                        uint32_t rx_power_fil_dB,
                        uint8_t gNB_id);

int nr_pdsch_ptrs_tdinterpol(const NR_UE_DLSCH_t *dlsch, c16_t phase_per_symbol[NR_SYMBOLS_PER_SLOT]);

void nr_pdsch_ptrs_compensate(const c16_t phase_per_symbol,
                              const int symbol,
                              const NR_UE_DLSCH_t *dlsch,
                              c16_t rxdataF_comp[dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB]);

void nr_pdsch_ptrs_processing_core(const PHY_VARS_NR_UE *ue,
                                   const int gNB_id,
                                   const int nr_slot_rx,
                                   const int symbol,
                                   const int rnti,
                                   const NR_UE_DLSCH_t *dlsch,
                                   c16_t rxdataF_comp[dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
                                   c16_t *phase_per_symbol,
                                   int32_t *ptrs_re_symbol);

int nr_sl_psbch_rsrp_measurements(sl_nr_ue_phy_params_t *sl_phy_params,
                                  const NR_DL_FRAME_PARMS *fp,
                                  const int symbol,
                                  const c16_t rxdataF[][fp->ofdm_symbol_size],
                                  bool use_SSS,
                                  openair0_config_t *openair0_cfg);
/** @}*/
#endif

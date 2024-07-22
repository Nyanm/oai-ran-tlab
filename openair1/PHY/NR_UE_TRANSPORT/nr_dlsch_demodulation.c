/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*! \file nr_dlsch_demodulation.c
 * \brief Top-level routines for demodulating the PDSCH physical channel from 38-211, V15.2 2018-06
 * \author H.Wang
 * \date 2018
 * \version 0.1
 * \company Eurecom
 * \note
 * \warning
 */

#include "common/platform_constants.h"
#include "nr_phy_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto_ue.h"
#include "PHY/sse_intrin.h"
#include "T.h"
#include "openair1/PHY/NR_UE_ESTIMATION/nr_estimation.h"
#include "openair1/PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "common/utils/nr/nr_common.h"
#include <complex.h>
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"

// #define DEBUG_HARQ(a...) printf(a)
#define DEBUG_HARQ(...)
//#define DEBUG_DLSCH_DEMOD
//#define DEBUG_PDSCH_RX

// [MCS][i_mod (0,1,2) = (2,4,6)]
//unsigned char offset_mumimo_llr_drange_fix=0;
//inferference-free case
/*unsigned char interf_unaw_shift_tm4_mcs[29]={5, 3, 4, 3, 3, 2, 1, 1, 2, 0, 1, 1, 1, 1, 0, 0,
                                             1, 1, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 0} ;*/

//unsigned char interf_unaw_shift_tm1_mcs[29]={5, 5, 4, 3, 3, 3, 2, 2, 4, 4, 2, 3, 3, 3, 1, 1,
//                                          0, 1, 1, 2, 5, 4, 4, 6, 5, 1, 0, 5, 6} ; // mcs 21, 26, 28 seem to be errorneous

/*
unsigned char offset_mumimo_llr_drange[29][3]={{8,8,8},{7,7,7},{7,7,7},{7,7,7},{6,6,6},{6,6,6},{6,6,6},{5,5,5},{4,4,4},{1,2,4}, // QPSK
{5,5,4},{5,5,5},{5,5,5},{3,3,3},{2,2,2},{2,2,2},{2,2,2}, // 16-QAM
{2,2,1},{3,3,3},{3,3,3},{3,3,1},{2,2,2},{2,2,2},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}}; //64-QAM
*/
 /*
 //first optimization try
 unsigned char offset_mumimo_llr_drange[29][3]={{7, 8, 7},{6, 6, 7},{6, 6, 7},{6, 6, 6},{5, 6, 6},{5, 5, 6},{5, 5, 6},{4, 5, 4},{4, 3, 4},{3, 2, 2},{6, 5, 5},{5, 4, 4},{5, 5, 4},{3, 3, 2},{2, 2, 1},{2, 1, 1},{2, 2, 2},{3, 3, 3},{3, 3, 2},{3, 3, 2},{3, 2, 1},{2, 2, 2},{2, 2, 2},{0, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0}};
 */
 //second optimization try
 /*
   unsigned char offset_mumimo_llr_drange[29][3]={{5, 8, 7},{4, 6, 8},{3, 6, 7},{7, 7, 6},{4, 7, 8},{4, 7, 4},{6, 6, 6},{3, 6, 6},{3, 6, 6},{1, 3, 4},{1, 1, 0},{3, 3, 2},{3, 4, 1},{4, 0, 1},{4, 2, 2},{3, 1, 2},{2, 1, 0},{2, 1, 1},{1, 0, 1},{1, 0, 1},{0, 0, 0},{1, 0, 0},{0, 0, 0},{0, 1, 0},{1, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0}};  w
 */
//unsigned char offset_mumimo_llr_drange[29][3]= {{0, 6, 5},{0, 4, 5},{0, 4, 5},{0, 5, 4},{0, 5, 6},{0, 5, 3},{0, 4, 4},{0, 4, 4},{0, 3, 3},{0, 1, 2},{1, 1, 0},{1, 3, 2},{3, 4, 1},{2, 0, 0},{2, 2, 2},{1, 1, 1},{2, 1, 0},{2, 1, 1},{1, 0, 1},{1, 0, 1},{0, 0, 0},{1, 0, 0},{0, 0, 0},{0, 1, 0},{1, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0},{0, 0, 0}};

#define print_ints(s,x) printf("%s = %d %d %d %d\n",s,(x)[0],(x)[1],(x)[2],(x)[3])
#define print_shorts(s,x) printf("%s = [%d+j*%d, %d+j*%d, %d+j*%d, %d+j*%d]\n",s,(x)[0],(x)[1],(x)[2],(x)[3],(x)[4],(x)[5],(x)[6],(x)[7])

static bool overlap_csi_symbol(const fapi_nr_dl_config_csirs_pdu_rel15_t *csi_pdu, int symbol)
{
  int num_l0 [18] = {1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 4, 2, 2, 4};
  for (int s = 0; s < num_l0[csi_pdu->row - 1]; s++) {
    if (symbol == csi_pdu->symb_l0 + s)
      return true;
  }
  // check also l1 if relevant
  if (csi_pdu->row == 13 || csi_pdu->row == 14 || csi_pdu->row == 16 || csi_pdu->row == 17) {
    for (int s = 0; s < 2; s++) { // two consecutive symbols including l1
      if (symbol == csi_pdu->symb_l1 + s)
        return true;
    }
  }
  return false;
}

uint32_t build_csi_overlap_bitmap(const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config, int symbol)
{
  // LS 16 bits for even RBs, MS 16 bits for odd RBs
  uint32_t csi_res_bitmap = 0;
  int num_k[18] = {1, 1, 1, 1, 1, 4, 2, 2, 6, 3, 4, 4, 3, 3, 3, 4, 4, 4};
  for (int i = 0; i < dlsch_config->numCsiRsForRateMatching; i++) {
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csi_pdu = &dlsch_config->csiRsForRateMatching[i];

    if (!overlap_csi_symbol(csi_pdu, symbol))
      continue;

    int num_kp = 1;
    int mult = 1;
    int k0_step = 0;
    int num_k0 = 1;
    switch (csi_pdu->row) {
      case 1:
        k0_step = 4;
        num_k0 = 3;
        break;
      case 2:
        break;
      case 4:
        num_kp = 2;
        mult = 4;
        k0_step = 2;
        num_k0 = 2;
        break;
      default:
        num_kp = 2;
        mult = 2;
    }
    int found = 0;
    int bit = 0;
    uint32_t temp_res_map = 0;
    while (found < num_k[csi_pdu->row - 1]) {
      if ((csi_pdu->freq_domain >> bit) & 0x01) {
        for (int k0 = 0; k0 < num_k0; k0++) {
          for (int kp = 0; kp < num_kp; kp++) {
            int re = (bit * mult) + (k0 * k0_step) + kp;
            temp_res_map |= (1 << re);
          }
        }
        found++;
      }
      bit++;
      AssertFatal(bit < 13,
                  "Couldn't find %d positive bits in bitmap %d for CSI freq. domain\n",
                  num_k[csi_pdu->row - 1],
                  csi_pdu->freq_domain);
    }
    if (csi_pdu->freq_density < 2)
      csi_res_bitmap |= (temp_res_map << (16 * csi_pdu->freq_density));
    else
      csi_res_bitmap |= (temp_res_map + (temp_res_map << 16));
  }
  return csi_res_bitmap;
}

bool get_isPilot_symbol(const int symbol, const NR_UE_DLSCH_t *dlsch)
{
  return (dlsch->dlsch_config.dlDmrsSymbPos >> symbol) & 1;
}

int get_nb_re_pdsch_symbol(const int symbol, const NR_UE_DLSCH_t *dlsch, const uint32_t csi_res_bitmap, const freq_alloc_bitmap_t *freq_alloc)
{
  const bool isPilot = get_isPilot_symbol(symbol, dlsch);
  const int config_type = dlsch->dlsch_config.dmrsConfigType;
  const int nb_rb_pdsch = freq_alloc->num_rbs;
  int nb_re_pdsch = (isPilot)
                        ? ((config_type == NFAPI_NR_DMRS_TYPE1) ? nb_rb_pdsch * (12 - 6 * dlsch->dlsch_config.n_dmrs_cdm_groups)
                                                                : nb_rb_pdsch * (12 - 4 * dlsch->dlsch_config.n_dmrs_cdm_groups))
                        : (nb_rb_pdsch * 12);
  if (csi_res_bitmap == 0)
    return nb_re_pdsch;

  uint32_t csi_re_count = 0;
  const uint32_t csi_res_even = csi_res_bitmap & 0xfff;
  const uint32_t csi_res_odd = (csi_res_bitmap >> 16) & 0xfff;
  const int start = freq_alloc->start[0] + dlsch->dlsch_config.BWPStart;
  const int end = freq_alloc->end[freq_alloc->num_blocks - 1] + 1;
  for (int rb = start; rb < end; rb++) {
    if ((freq_alloc->bitmap[rb / 8] >> (rb % 8)) & 0x01) {
      uint32_t rb_csi_pattern = (rb % 2 == 0) ? csi_res_even : csi_res_odd;
      csi_re_count += __builtin_popcount(rb_csi_pattern);
    }
  }
  nb_re_pdsch = (nb_re_pdsch > csi_re_count) ? (nb_re_pdsch - csi_re_count) : 0;
  return nb_re_pdsch;
}

int compute_dl_valid_re(const NR_UE_DLSCH_t *dlsch, const int32_t ptrs_re[][NR_SYMBOLS_PER_SLOT], int ret[NR_SYMBOLS_PER_SLOT])
{
  const int start_symb = dlsch->dlsch_config.start_symbol;
  const int numb_symb = dlsch->dlsch_config.number_symbols;

  int sum = 0;
  for (int i = start_symb; i < start_symb + numb_symb; i++) {
    ret[i] -= ptrs_re[0][i];
    sum += ret[i];
  }
  return sum;
}

static void nr_channel_compensation(const int length,
                                    const int nb_rb,
                                    const int output_shift,
                                    const int mod_order,
                                    const c16_t dl_ch_estimates_ext[nb_rb * NR_NB_SC_PER_RB],
                                    const c16_t rxdataF_ext[nb_rb * NR_NB_SC_PER_RB],
                                    c16_t dl_ch_mag[nb_rb * NR_NB_SC_PER_RB],
                                    c16_t dl_ch_magb[nb_rb * NR_NB_SC_PER_RB],
                                    c16_t dl_ch_magr[nb_rb * NR_NB_SC_PER_RB],
                                    c16_t rxdataF_comp[nb_rb * NR_NB_SC_PER_RB]);

/* Do channel compenstion on all antennas */
static void nr_pdsch_channel_compensation(
    const NR_DL_FRAME_PARMS *frame_parms,
    const NR_UE_DLSCH_t *dlsch,
    const int nb_re_pdsch,
    const int symVecSz,
    const c16_t rxdataF_ext[frame_parms->nb_antennas_rx][symVecSz],
    const c16_t dl_ch_est_ext[dlsch->Nl][frame_parms->nb_antennas_rx][symVecSz],
    c16_t dl_ch_mag[dlsch->Nl][frame_parms->nb_antennas_rx][symVecSz],
    c16_t dl_ch_magb[dlsch->Nl][frame_parms->nb_antennas_rx][symVecSz],
    c16_t dl_ch_magr[dlsch->Nl][frame_parms->nb_antennas_rx][symVecSz],
    c16_t rxdataF_comp[dlsch->Nl][frame_parms->nb_antennas_rx][symVecSz])
{
  const int log2_maxh = get_maxh_extimates(frame_parms, dlsch, dlsch->Nl, frame_parms->nb_antennas_rx, symVecSz, nb_re_pdsch, dl_ch_est_ext);
  for (int l = 0; l < dlsch->Nl; l++) {
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      nr_channel_compensation(nb_re_pdsch,
                              log2_maxh,
                              dlsch->dlsch_config.qamModOrder,
                              symVecSz,
                              dl_ch_est_ext[l][aarx],
                              rxdataF_ext[aarx],
                              dl_ch_mag[l][aarx],
                              dl_ch_magb[l][aarx],
                              dl_ch_magr[l][aarx],
                              rxdataF_comp[l][aarx]);
    }
  }
}

/* Extract channel estimates for corresponding data REs in a OFDM symbol for all antennas */
void nr_generate_pdsch_extracted_chestimates(
    const PHY_VARS_NR_UE *ue,
    const NR_UE_DLSCH_t *dlsch,
    const int symbol,
    const int csi_res_bitmap,
    const freq_alloc_bitmap_t *freq_alloc,
    const c16_t dl_ch_est[NR_SYMBOLS_PER_SLOT][dlsch->Nl][ue->frame_parms.nb_antennas_rx][ue->frame_parms.ofdm_symbol_size],
    const int vecSize,
    c16_t dl_ch_est_ext[dlsch->Nl][ue->frame_parms.nb_antennas_rx][vecSize])
{
  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  const int nbAntRx = frame_parms->nb_antennas_rx;
  const int dlDmrsSymbPos = dlsch->dlsch_config.dlDmrsSymbPos;
  /* TODO: Could be launched in Tpool for MIMO */
  const int validDmrsEst = (ue->chest_time == 0) ? get_valid_dmrs_idx_for_channel_est(dlDmrsSymbPos, symbol)
                                                 : get_next_dmrs_symbol_in_slot(dlDmrsSymbPos, 0, 14);
  for (int aarx = 0; aarx < nbAntRx; aarx++) {
    for (int l = 0; l < dlsch->Nl; l++) {
      /* Extract estimates for all symbols */
      nr_extract_pdsch_chest_res(frame_parms,
                                 &dlsch->dlsch_config,
                                 get_isPilot_symbol(symbol, dlsch),
                                 csi_res_bitmap,
                                 freq_alloc,
                                 dl_ch_est[validDmrsEst][l][aarx],
                                 vecSize,
                                 dl_ch_est_ext[l][aarx]);
    }
  }
}

/* Do PTRS estimation on all antennas */
void nr_pdsch_ptrs_processing(
    const PHY_VARS_NR_UE *ue,
    const int gNB_id,
    const int rnti,
    const int slot,
    const int symbol,
    const NR_UE_DLSCH_t *dlsch,
    c16_t rxdataF_comp[dlsch->Nl][ue->frame_parms.nb_antennas_rx][dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
    c16_t ptrs_phase[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT],
    int32_t ptrs_re[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT])
{
  for (int aarx = 0; aarx < ue->frame_parms.nb_antennas_rx; aarx++) {
    nr_pdsch_ptrs_processing_core(ue,
                                  gNB_id,
                                  slot,
                                  symbol,
                                  rnti,
                                  dlsch,
                                  rxdataF_comp[0][aarx],
                                  ptrs_phase[aarx] + symbol,
                                  ptrs_re[aarx] + symbol);
  }
}

void nr_pdsch_comp_out(void *parms)
{
  nr_ue_symb_data_t *msg = (nr_ue_symb_data_t *)parms;
  const PHY_VARS_NR_UE *ue = msg->UE;
  const UE_nr_rxtx_proc_t *proc = msg->proc;
  const NR_UE_DLSCH_t *dlsch = msg->p_dlsch;
  const int symbolVecSize = msg->symbVecSz;
  const int nl = dlsch->Nl;
  const int n_rx = ue->frame_parms.nb_antennas_rx;
  const c16_t(*dl_ch_est)[NR_SYMBOLS_PER_SLOT][nl][n_rx][ue->frame_parms.ofdm_symbol_size] =
      (const c16_t(*)[NR_SYMBOLS_PER_SLOT][nl][n_rx][ue->frame_parms.ofdm_symbol_size])msg->pdsch_dl_ch_estimates;

  const uint32_t csi_res_bitmap = build_csi_overlap_bitmap(&dlsch->dlsch_config, msg->symbol);
  c16_t(*dl_ch_est_ext)[nl][n_rx][symbolVecSize] = (c16_t(*)[nl][n_rx][symbolVecSize])msg->pdsch_dl_ch_est_ext;
  nr_generate_pdsch_extracted_chestimates(ue, dlsch, msg->symbol, csi_res_bitmap,msg->freq_alloc, *dl_ch_est,symbolVecSize, *dl_ch_est_ext);

  const int nb_re_pdsch = get_nb_re_pdsch_symbol(msg->symbol, dlsch, csi_res_bitmap, msg->freq_alloc);
  *msg->dl_valid_re = nb_re_pdsch;
  {
    // nr_scale_channel common function takes in a 2D array input
    c16_t(*dl_ch_est_ext_scale)[dlsch->Nl * ue->frame_parms.nb_antennas_rx][symbolVecSize] =
        (c16_t(*)[dlsch->Nl * ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->pdsch_dl_ch_est_ext;
    nr_scale_channel(symbolVecSize, *dl_ch_est_ext_scale, 0, nb_re_pdsch, nl, n_rx, 0);
  }

  const int maxh = get_maxh_extimates(&ue->frame_parms, dlsch, nl, n_rx, symbolVecSize, nb_re_pdsch, *dl_ch_est_ext);

  const c16_t(*rxdataF_ext)[ue->frame_parms.nb_antennas_rx][symbolVecSize] =
      (const c16_t(*)[ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->rxdataF_ext;

  c16_t(*dl_ch_mag)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize] =
      (c16_t(*)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->dl_ch_mag;

  c16_t(*dl_ch_magb)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize] =
      (c16_t(*)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->dl_ch_magb;

  c16_t(*dl_ch_magr)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize] =
      (c16_t(*)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->dl_ch_magr;

  c16_t(*rxdataF_comp)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize] =
      (c16_t(*)[dlsch->Nl][ue->frame_parms.nb_antennas_rx][symbolVecSize])msg->rxdataF_comp;

  nr_pdsch_channel_compensation(&ue->frame_parms,
                                dlsch,
                                nb_re_pdsch,
                                symbolVecSize,
                                *rxdataF_ext,
                                *dl_ch_est_ext,
                                *dl_ch_mag,
                                *dl_ch_magb,
                                *dl_ch_magr,
                                *rxdataF_comp);

  if (ue->frame_parms.nb_antennas_rx > 1) {
    nr_dlsch_detection_mrc(dlsch->Nl,
                           ue->frame_parms.nb_antennas_rx,
                           dlsch->dlsch_config.number_rbs,
                           nb_re_pdsch,
                           *rxdataF_comp,
                           *dl_ch_mag,
                           *dl_ch_magb,
                           *dl_ch_magr);
    if (dlsch->Nl >= 2) {
      nr_dlsch_mmse(dlsch->Nl,
                    ue->frame_parms.nb_antennas_rx,
                    dlsch->dlsch_config.number_rbs,
                    nb_re_pdsch,
                    dlsch->dlsch_config.qamModOrder,
                    maxh,
                    dlsch->nvar,
                    symbolVecSize,
                    *dl_ch_est_ext,
                    *rxdataF_comp,
                    *dl_ch_mag,
                    *dl_ch_magb,
                    *dl_ch_magr);
    }
  }

  c16_t(*ptrs_phase)[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT] =
      (c16_t(*)[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT])msg->ptrs_phase_per_slot;

  int32_t(*ptrs_re)[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT] =
      (int32_t(*)[ue->frame_parms.nb_antennas_rx][NR_SYMBOLS_PER_SLOT])msg->ptrs_re_per_slot;

  if ((dlsch->dlsch_config.pduBitmap & 0x1) && (dlsch->rnti_type == TYPE_C_RNTI_)) {
    nr_pdsch_ptrs_processing(ue,
                             proc->gNB_id,
                             dlsch->rnti,
                             proc->nr_slot_rx,
                             msg->symbol,
                             dlsch,
                             *rxdataF_comp,
                             *ptrs_phase,
                             *ptrs_re);
  }

  // Task completed
  completed_task_ans(msg->ans);
}

void nr_compute_channel_correlation(const int n_layers,
                                    const int length,
                                    const int nb_antennas_rx,
                                    const int antIdx,
                                    const int output_shift,
                                    const int vecSz,
                                    const c16_t dl_ch_estimates_ext[n_layers][nb_antennas_rx][vecSz],
                                    int32_t rho[n_layers][n_layers][vecSz])
{
  // we compute the Tx correlation matrix for each Rx antenna
  // As an example the 2x2 MIMO case requires
  // rho[aarx][nl*nl] = [cov(H_aarx_0,H_aarx_0) cov(H_aarx_0,H_aarx_1)
  //                               cov(H_aarx_1,H_aarx_0) cov(H_aarx_1,H_aarx_1)], aarx=0,...,nb_antennas_rx-1

  // int avg_rho_re[frame_parms->nb_antennas_rx][nl*nl];
  // int avg_rho_im[frame_parms->nb_antennas_rx][nl*nl];

  const int nb_rb_0 = length / 12 + ((length % 12) ? 1 : 0);
  for (int l = 0; l < n_layers; l++) {
    for (int atx = 0; atx < n_layers; atx++) {
      // avg_rho_re[aarx][l*n_layers+atx] = 0;
      // avg_rho_im[aarx][l*n_layers+atx] = 0;
      simde__m128i *rho128 = (simde__m128i *)rho[l][atx];
      simde__m128i *dl_ch128 = (simde__m128i *)dl_ch_estimates_ext[l][antIdx];
      simde__m128i *dl_ch128_2 = (simde__m128i *)dl_ch_estimates_ext[atx][antIdx];

      // multiply by conjugated channel
      mult_cpx_conj_vector((c16_t *)dl_ch128, (c16_t *)dl_ch128_2, (c16_t *)rho128, 12 * nb_rb_0, output_shift);
    }
  }
}

//==============================================================================================
// Pre-processing for LLR computation
//==============================================================================================

static void nr_channel_compensation(const int length,
                                    const int output_shift,
                                    const int mod_order,
                                    const int vecSz,
                                    const c16_t dl_ch_estimates_ext[vecSz],
                                    const c16_t rxdataF_ext[vecSz],
                                    c16_t dl_ch_mag[vecSz],
                                    c16_t dl_ch_magb[vecSz],
                                    c16_t dl_ch_magr[vecSz],
                                    c16_t rxdataF_comp[vecSz])
{
  simde__m128i QAM_amp128 = {0};
  simde__m128i QAM_amp128b = {0};
  simde__m128i QAM_amp128r = {0};

  if (mod_order == 4) {
    QAM_amp128 = simde_mm_set1_epi16(QAM16_n1); // 2/sqrt(10)
    QAM_amp128b = simde_mm_setzero_si128();
    QAM_amp128r = simde_mm_setzero_si128();
  } else if (mod_order == 6) {
    QAM_amp128 = simde_mm_set1_epi16(QAM64_n1); //
    QAM_amp128b = simde_mm_set1_epi16(QAM64_n2);
    QAM_amp128r = simde_mm_setzero_si128();
  } else if (mod_order == 8) {
    QAM_amp128 = simde_mm_set1_epi16(QAM256_n1);
    QAM_amp128b = simde_mm_set1_epi16(QAM256_n2);
    QAM_amp128r = simde_mm_set1_epi16(QAM256_n3);
  }

  const int nb_rb_0 = length / 12 + ((length % 12) ? 1 : 0);
  const simde__m128i *dl_ch128 = (simde__m128i *)dl_ch_estimates_ext;
  const simde__m128i *rxdataF128 = (simde__m128i *)rxdataF_ext;
  simde__m128i *dl_ch_mag128 = (simde__m128i *)dl_ch_mag;
  simde__m128i *dl_ch_mag128b = (simde__m128i *)dl_ch_magb;
  simde__m128i *dl_ch_mag128r = (simde__m128i *)dl_ch_magr;
  simde__m128i *rxdataF_comp128 = (simde__m128i *)rxdataF_comp;
  for (int rb = 0; rb < nb_rb_0; rb++) {
    simde__m128i mmtmpD0;
    simde__m128i mmtmpD1;

    if (mod_order > 2) {
      // get channel amplitude if not QPSK

      mmtmpD0 = simde_mm_madd_epi16(dl_ch128[0], dl_ch128[0]);
      mmtmpD0 = simde_mm_srai_epi32(mmtmpD0, output_shift);

      mmtmpD1 = simde_mm_madd_epi16(dl_ch128[1], dl_ch128[1]);
      mmtmpD1 = simde_mm_srai_epi32(mmtmpD1, output_shift);

      mmtmpD0 = simde_mm_packs_epi32(mmtmpD0, mmtmpD1); //|H[0]|^2 |H[1]|^2 |H[2]|^2 |H[3]|^2 |H[4]|^2 |H[5]|^2 |H[6]|^2 |H[7]|^2

      // store channel magnitude here in a new field of dlsch

      dl_ch_mag128[0] = simde_mm_unpacklo_epi16(mmtmpD0, mmtmpD0);
      dl_ch_mag128b[0] = dl_ch_mag128[0];
      dl_ch_mag128r[0] = dl_ch_mag128[0];
      dl_ch_mag128[0] = simde_mm_mulhrs_epi16(dl_ch_mag128[0], QAM_amp128);
      dl_ch_mag128b[0] = simde_mm_mulhrs_epi16(dl_ch_mag128b[0], QAM_amp128b);
      dl_ch_mag128r[0] = simde_mm_mulhrs_epi16(dl_ch_mag128r[0], QAM_amp128r);

      dl_ch_mag128[1] = simde_mm_unpackhi_epi16(mmtmpD0, mmtmpD0);
      dl_ch_mag128b[1] = dl_ch_mag128[1];
      dl_ch_mag128r[1] = dl_ch_mag128[1];
      dl_ch_mag128[1] = simde_mm_mulhrs_epi16(dl_ch_mag128[1], QAM_amp128);
      dl_ch_mag128b[1] = simde_mm_mulhrs_epi16(dl_ch_mag128b[1], QAM_amp128b);
      dl_ch_mag128r[1] = simde_mm_mulhrs_epi16(dl_ch_mag128r[1], QAM_amp128r);

      mmtmpD0 = simde_mm_madd_epi16(dl_ch128[2], dl_ch128[2]);
      mmtmpD0 = simde_mm_srai_epi32(mmtmpD0, output_shift);
      mmtmpD1 = simde_mm_packs_epi32(mmtmpD0, mmtmpD0);

      dl_ch_mag128[2] = simde_mm_unpacklo_epi16(mmtmpD1, mmtmpD1);
      dl_ch_mag128b[2] = dl_ch_mag128[2];
      dl_ch_mag128r[2] = dl_ch_mag128[2];

      dl_ch_mag128[2] = simde_mm_mulhrs_epi16(dl_ch_mag128[2], QAM_amp128);
      dl_ch_mag128b[2] = simde_mm_mulhrs_epi16(dl_ch_mag128b[2], QAM_amp128b);
      dl_ch_mag128r[2] = simde_mm_mulhrs_epi16(dl_ch_mag128r[2], QAM_amp128r);
    }
    // Multiply received data by conjugated channel
    rxdataF_comp128[0] = oai_mm_cpx_mult_conj(dl_ch128[0], rxdataF128[0], output_shift);
    rxdataF_comp128[1] = oai_mm_cpx_mult_conj(dl_ch128[1], rxdataF128[1], output_shift);
    rxdataF_comp128[2] = oai_mm_cpx_mult_conj(dl_ch128[2], rxdataF128[2], output_shift);

    dl_ch128 += 3;
    dl_ch_mag128 += 3;
    dl_ch_mag128b += 3;
    dl_ch_mag128r += 3;
    rxdataF128 += 3;
    rxdataF_comp128 += 3;
  }
}

int get_nr_channel_level_median(const int avg, const int length, const int extSize, const c16_t dl_ch_estimates_ext[extSize])
{
  int max = avg; // initialize the med point for max
  int min = avg; // initialize the med point for min
  const int length2 = length >> 2; // length = number of REs, hence length2=nb_REs*(32/128) in SIMD loop

  const simde__m128i *dl_ch128 = (simde__m128i *)dl_ch_estimates_ext;

  for (int i = 0; i < length2; i++) {
    const simde__m128i norm128D =
        simde_mm_srai_epi32(simde_mm_madd_epi16(dl_ch128[0], dl_ch128[0]), 2); //[|H_0|²/4 |H_1|²/4 |H_2|²/4 |H_3|²/4]

    int32_t *tmp = (int32_t *)&norm128D;
    const int64_t norm_pack = (int64_t)tmp[0] + tmp[1] + tmp[2] + tmp[3];

    if (norm_pack > max)
      max = norm_pack; // store values more than max
    if (norm_pack < min)
      min = norm_pack; // store values less than min
    dl_ch128 += 1;
  }

  const int median = (max + min) >> 1;
  return median;
}

/* Get average channel level from all antennas */
int32_t get_maxh_extimates(
    const NR_DL_FRAME_PARMS *frame_parms,
    const NR_UE_DLSCH_t *dlsch,
    const int nl,
    const int nbAntRx,
    const int symbVectSz,
    const int nb_re_pdsch,
    const c16_t dl_ch_est_ext[nl][nbAntRx][symbVectSz])
{
  int32_t avg[nl * nbAntRx];
  if (nb_re_pdsch) {
    const c16_t(*ch)[nl * nbAntRx][symbVectSz] = (const c16_t (*)[nl * nbAntRx][symbVectSz])dl_ch_est_ext;
    nr_channel_level(0, symbVectSz, *ch, nbAntRx, nl, avg, nb_re_pdsch);
  } else
    LOG_E(NR_PHY, "Average channel level is 0: nb_re_pdsch = %d\n", nb_re_pdsch);
  int median;
  int avgs = 0;
  for (int l = 0; l < nl; l++) {
    for (int aarx = 0; aarx < nbAntRx; aarx++) {
      avgs = cmax(avgs, avg[l * nbAntRx + aarx]);
      if (nl > 1) {
        median = get_nr_channel_level_median(avg[l * nbAntRx + aarx],
                                             nb_re_pdsch,
                                             symbVectSz,
                                             dl_ch_est_ext[l][aarx]);
        avgs = cmax(avgs, median);
      }
    }
  }
  return ((log2_approx(avgs) / 2) + 1);
}

//==============================================================================================
// Extraction functions
//==============================================================================================

static void set_dmrs_csi_overlap_bitmap(uint8_t config_type,
                                        uint8_t n_dmrs_cdm_groups,
                                        uint32_t csi_res_bitmap,
                                        bool isPilot,
                                        uint32_t *dmrs_csi_overlap_even,
                                        uint32_t *dmrs_csi_overlap_odd)
{
  uint32_t dmrs_rb_bitmap = 0;
  if (isPilot) {
    dmrs_rb_bitmap = 0xfff; // all REs taken by dmrs
    if (config_type == NFAPI_NR_DMRS_TYPE1 && n_dmrs_cdm_groups == 1)
      dmrs_rb_bitmap = 0x555; // alternating REs starting from 0
    if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 1)
      dmrs_rb_bitmap = 0xc3;  // REs 0,1 and 6,7
    if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 2)
      dmrs_rb_bitmap = 0x3cf;  // REs 0,1,2,3 and 6,7,8,9
  }

  // csi_res_bitmap LS 16 bits for even RBs, MS 16 bits for odd RBs
  uint32_t csi_res_even = csi_res_bitmap & 0xfff;
  uint32_t csi_res_odd = (csi_res_bitmap >> 16) & 0xfff;
  AssertFatal((dmrs_rb_bitmap & csi_res_even) == 0, "DMRS RE overlapping with CSI RE, it shouldn't happen\n");
  AssertFatal((dmrs_rb_bitmap & csi_res_odd) == 0, "DMRS RE overlapping with CSI RE, it shouldn't happen\n");
  *dmrs_csi_overlap_even = csi_res_even + dmrs_rb_bitmap;
  *dmrs_csi_overlap_odd = csi_res_odd + dmrs_rb_bitmap;
}

/* Extract data resourse elements from received OFDM symbol */
void nr_extract_data_res(const NR_DL_FRAME_PARMS *frame_parms,
                         const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config,
                        const freq_alloc_bitmap_t *freq_alloc,
                         const bool isPilot,
                         const uint32_t csiResBitMap,
                         const c16_t rxdataF[frame_parms->ofdm_symbol_size],
                         const int vecSz,
                         c16_t rxdataF_ext[vecSz])
{
  const int configType = dlsch_config->dmrsConfigType;
  const int nDmrsCdmGroups = dlsch_config->n_dmrs_cdm_groups;
  if (configType == NFAPI_NR_DMRS_TYPE1)
    AssertFatal(nDmrsCdmGroups == 1 || nDmrsCdmGroups == 2, "nDmrsCdmGroups %d is illegal\n", nDmrsCdmGroups);
  else
    AssertFatal(nDmrsCdmGroups == 1 || nDmrsCdmGroups == 2 || nDmrsCdmGroups == 3,
                "nDmrsCdmGroups %d is illegal\n",
                nDmrsCdmGroups);

  uint32_t dmrsCsiOverlapEven, dmrsCsiOverlapOdd;
  set_dmrs_csi_overlap_bitmap(configType, nDmrsCdmGroups, csiResBitMap, isPilot, &dmrsCsiOverlapEven, &dmrsCsiOverlapOdd);
  for (int b = 0; b < freq_alloc->num_blocks; b++) {
    const int pdschStartRb = freq_alloc->start[b] + dlsch_config->BWPStart;
    const int pdschNbRb = freq_alloc->end[b] - freq_alloc->start[b] + 1;
    const int startRe = (frame_parms->first_carrier_offset + pdschStartRb * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
    if (isPilot || csiResBitMap) {
      int j = 0;
      int k = startRe;
      for (int rb = pdschStartRb; rb < pdschStartRb+pdschNbRb; rb++) {
        const uint32_t overlapMap = rb % 2 ? dmrsCsiOverlapOdd : dmrsCsiOverlapEven;
        for (int re = 0; re < NR_NB_SC_PER_RB; re++) {
          if (((overlapMap >> re) & 0x01) == 0) {
            rxdataF_ext[j] = rxdataF[k];
            j++;
          }
          k++;
          if (k >= frame_parms->ofdm_symbol_size)
            k -= frame_parms->ofdm_symbol_size;
        }
      }
    } else {
      if (startRe + pdschNbRb * NR_NB_SC_PER_RB <= frame_parms->ofdm_symbol_size) {
        memcpy(rxdataF_ext, &rxdataF[startRe], pdschNbRb * NR_NB_SC_PER_RB * sizeof(c16_t));
      } else {
        const int negLength = frame_parms->ofdm_symbol_size - startRe;
        const int posLength = pdschNbRb * NR_NB_SC_PER_RB - negLength;
        memcpy(rxdataF_ext, &rxdataF[startRe], negLength * sizeof(c16_t));
        memcpy(&rxdataF_ext[negLength], rxdataF, posLength * sizeof(c16_t));
      }
    }
  }
}

/* Extract channel estimates for corresponding data REs in a OFDM symbol */
void nr_extract_pdsch_chest_res(const NR_DL_FRAME_PARMS *frame_parms,
                                const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config,
                                const bool isPilot,
                                const uint32_t csiResBitMap,
                                const freq_alloc_bitmap_t *freq_alloc,
                                const c16_t dl_ch_est[frame_parms->ofdm_symbol_size],
                                const int vecSz,
                                c16_t dl_ch_est_ext[vecSz])
{
  const int configType = dlsch_config->dmrsConfigType;
  const int nDmrsCdmGroups = dlsch_config->n_dmrs_cdm_groups;
  if (configType == NFAPI_NR_DMRS_TYPE1)
    AssertFatal(nDmrsCdmGroups == 1 || nDmrsCdmGroups == 2, "nDmrsCdmGroups %d is illegal\n", nDmrsCdmGroups);
  else
    AssertFatal(nDmrsCdmGroups == 1 || nDmrsCdmGroups == 2 || nDmrsCdmGroups == 3,
                "nDmrsCdmGroups %d is illegal\n",
                nDmrsCdmGroups);

  uint32_t dmrsCsiOverlapEven, dmrsCsiOverlapOdd;
  set_dmrs_csi_overlap_bitmap(configType, nDmrsCdmGroups, csiResBitMap,isPilot, &dmrsCsiOverlapEven, &dmrsCsiOverlapOdd);
  for (int b = 0; b < freq_alloc->num_blocks; b++) {
    const int startRb = freq_alloc->start[b] + dlsch_config->BWPStart;
    const int pdschNbRb = freq_alloc->end[b]-freq_alloc->start[b]+1;
    if (isPilot || csiResBitMap) {
      int j = 0;
      for (int rb = startRb; rb < startRb+pdschNbRb; rb++) {
        const uint32_t overlapMap = rb % 2 ? dmrsCsiOverlapOdd : dmrsCsiOverlapEven;
        for (int re = 0; re < NR_NB_SC_PER_RB; re++) {
          if (((overlapMap >> re) & 0x01) == 0) {
            dl_ch_est_ext[j] = dl_ch_est[re];
            j++;
          }
        }
        dl_ch_est += NR_NB_SC_PER_RB;
      }
    } else {
      memcpy(dl_ch_est_ext, dl_ch_est, pdschNbRb * NR_NB_SC_PER_RB * sizeof(c16_t));
    }
  }
}

void nr_dlsch_detection_mrc(const int nl,
                            const int n_rx,
                            const int nb_rb,
                            const int length,
                            c16_t rxdataF_comp[nl][n_rx][nb_rb * NR_NB_SC_PER_RB],
                            c16_t dl_ch_mag[nl][n_rx][nb_rb * NR_NB_SC_PER_RB],
                            c16_t dl_ch_magb[nl][n_rx][nb_rb * NR_NB_SC_PER_RB],
                            c16_t dl_ch_magr[nl][n_rx][nb_rb * NR_NB_SC_PER_RB])
{
  simde__m128i *rxdataF_comp128_0,*rxdataF_comp128_1,*dl_ch_mag128_0,*dl_ch_mag128_1,*dl_ch_mag128_0b,*dl_ch_mag128_1b,*dl_ch_mag128_0r,*dl_ch_mag128_1r;
  uint32_t nb_rb_0 = length / 12 + ((length % 12) ? 1 : 0);

  if (n_rx > 1) {
    for (int l = 0; l < nl; l++) {
      rxdataF_comp128_0 = (simde__m128i *)(rxdataF_comp[l][0]);
      dl_ch_mag128_0 = (simde__m128i *)dl_ch_mag[l][0];
      dl_ch_mag128_0b = (simde__m128i *)dl_ch_magb[l][0];
      dl_ch_mag128_0r = (simde__m128i *)dl_ch_magr[l][0];
      for (int aarx = 1; aarx < n_rx; aarx++) {
        rxdataF_comp128_1 = (simde__m128i *)(rxdataF_comp[l][aarx]);
        dl_ch_mag128_1 = (simde__m128i *)dl_ch_mag[l][aarx];
        dl_ch_mag128_1b = (simde__m128i *)dl_ch_magb[l][aarx];
        dl_ch_mag128_1r = (simde__m128i *)dl_ch_magr[l][aarx];

        // MRC on each re of rb, both on MF output and magnitude (for 16QAM/64QAM/256 llr computation)
        for (int i = 0; i < nb_rb_0 * 3; i++) {
          rxdataF_comp128_0[i] = simde_mm_adds_epi16(rxdataF_comp128_0[i],rxdataF_comp128_1[i]);
          dl_ch_mag128_0[i]    = simde_mm_adds_epi16(dl_ch_mag128_0[i],dl_ch_mag128_1[i]);
          dl_ch_mag128_0b[i]   = simde_mm_adds_epi16(dl_ch_mag128_0b[i],dl_ch_mag128_1b[i]);
          dl_ch_mag128_0r[i]   = simde_mm_adds_epi16(dl_ch_mag128_0r[i],dl_ch_mag128_1r[i]);
        }
      }
    }
#ifdef DEBUG_DLSCH_DEMOD
    for (int i = 0; i < nb_rb_0 * 3; i++) {
      printf("RB %d\n", i / 3);
      rxdataF_comp128_0 = (simde__m128i *)(rxdataF_comp[0][0]);
      rxdataF_comp128_1 = (simde__m128i *)(rxdataF_comp[0][n_rx]);
      print_shorts("tx 1 mrc_re/mrc_Im:",(int16_t*)&rxdataF_comp128_0[i]);
      print_shorts("tx 2 mrc_re/mrc_Im:",(int16_t*)&rxdataF_comp128_1[i]);
      // printf("mrc mag0 = %d = %d \n",((int16_t*)&dl_ch_mag128_0[0])[0],((int16_t*)&dl_ch_mag128_0[0])[1]);
      // printf("mrc mag0b = %d = %d \n",((int16_t*)&dl_ch_mag128_0b[0])[0],((int16_t*)&dl_ch_mag128_0b[0])[1]);
    }
#endif
  }
}

/* Zero Forcing Rx function: nr_a_sum_b()
 * Compute the complex addition x=x+y
 *
 * */
void nr_a_sum_b(c16_t *input_x, c16_t *input_y, unsigned short nb_rb)
{
  unsigned short rb;
  simde__m128i *x = (simde__m128i *)input_x;
  simde__m128i *y = (simde__m128i *)input_y;

  for (rb=0; rb<nb_rb; rb++) {
    x[0] = simde_mm_adds_epi16(x[0], y[0]);
    x[1] = simde_mm_adds_epi16(x[1], y[1]);
    x[2] = simde_mm_adds_epi16(x[2], y[2]);
    x += 3;
    y += 3;
  }
}

/* Zero Forcing Rx function: nr_element_sign()
 * Compute b=sign*a */
static inline void nr_element_sign(c16_t *a, c16_t *b, unsigned short nb_rb, int32_t sign)
{
  const int16_t nr_sign[8] __attribute__((aligned(16))) = {-1, -1, -1, -1, -1, -1, -1, -1};
  simde__m128i *a_128,*b_128;

  a_128 = (simde__m128i *)a;
  b_128 = (simde__m128i *)b;

  for (int rb = 0; rb < 3 * nb_rb; rb++) {
    if (sign < 0)
      b_128[rb] = simde_mm_sign_epi16(a_128[rb], ((simde__m128i *)nr_sign)[0]);
    else
      b_128[rb] = a_128[rb];

#ifdef DEBUG_DLSCH_DEMOD
    print_shorts("b:", (int16_t *)b_128);
#endif
  }
}

/* Zero Forcing Rx function: nr_det_4x4()
 * Compute the matrix determinant for 4x4 Matrix
 *
 * */
static void nr_determin(int size,
                        c16_t *a44[][size], //
                        c16_t *ad_bc, // ad-bc
                        unsigned short nb_rb,
                        int32_t sign,
                        int32_t shift0)
{
  AssertFatal(size > 0, "");

  if(size==1) {
    nr_element_sign(a44[0][0], // a
                    ad_bc, // b
                    nb_rb,
                    sign);
  } else {
    int16_t k, rr[size - 1], cc[size - 1];
    c16_t outtemp[12 * nb_rb] __attribute__((aligned(32)));
    c16_t outtemp1[12 * nb_rb] __attribute__((aligned(32)));
    c16_t *sub_matrix[size - 1][size - 1];
    for (int rtx=0;rtx<size;rtx++) {//row calculation for determin
      int ctx=0;
      //find the submatrix row and column indices
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      k=0;
      for(int cctx=0;cctx<size;cctx++)
        if(cctx != ctx) cc[k++] = cctx;
      // fill out the sub matrix corresponds to this element

      for (int ridx = 0; ridx < (size - 1); ridx++)
        for (int cidx = 0; cidx < (size - 1); cidx++)
          sub_matrix[cidx][ridx] = a44[cc[cidx]][rr[ridx]];

      nr_determin(size - 1,
                  sub_matrix, // a33
                  outtemp,
                  nb_rb,
                  ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1) * sign,
                  shift0);
      mult_complex_vectors(a44[ctx][rtx], outtemp, rtx == 0 ? ad_bc : outtemp1, sizeofArray(outtemp1), shift0);

      if (rtx != 0)
        nr_a_sum_b(ad_bc, outtemp1, nb_rb);
    }
  }
}

static double complex nr_determin_cpx(int32_t size, // size
                                      double complex a44_cpx[][size], //
                                      int32_t sign)
{
  double complex outtemp, outtemp1;
  //Allocate the submatrix elements
  DevAssert(size > 0);
  if(size==1) {
    return (a44_cpx[0][0] * sign);
  }else {
    double complex sub_matrix[size - 1][size - 1];
    int16_t k, rr[size - 1], cc[size - 1];
    outtemp1 = 0;
    for (int rtx=0;rtx<size;rtx++) {//row calculation for determin
      int ctx=0;
      //find the submatrix row and column indices
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      k=0;
      for(int cctx=0;cctx<size;cctx++)
        if(cctx != ctx) cc[k++] = cctx;
      //fill out the sub matrix corresponds to this element
       for (int ridx=0;ridx<(size-1);ridx++)
         for (int cidx=0;cidx<(size-1);cidx++)
           sub_matrix[cidx][ridx] = a44_cpx[cc[cidx]][rr[ridx]];

       outtemp = nr_determin_cpx(size - 1,
                                 sub_matrix, // a33
                                 ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1) * sign);
       outtemp1 += a44_cpx[ctx][rtx] * outtemp;
    }

    return((double complex)outtemp1);
  }
}

/* Zero Forcing Rx function: nr_matrix_inverse()
 * Compute the matrix inverse and determinant up to 4x4 Matrix
 *
 * */
uint8_t nr_matrix_inverse(int32_t size,
                          c16_t *a44[][size], // Input matrix//conjH_H_elements[0]
                          c16_t *inv_H_h_H[][size], // Inverse
                          c16_t *ad_bc, // determin
                          unsigned short nb_rb,
                          int32_t flag, // fixed point or floating flag
                          int32_t shift0)
{
  DevAssert(size > 1);
  int16_t k,rr[size-1],cc[size-1];

  if(flag) {//fixed point SIMD calc.
    //Allocate the submatrix elements
    c16_t *sub_matrix[size - 1][size - 1];

    //Compute Matrix determinant
    nr_determin(size,
                a44, //
                ad_bc, // determinant
                nb_rb,
                +1,
                shift0);
    //print_shorts("nr_det_",(int16_t*)&ad_bc[0]);

    //Compute Inversion of the H^*H matrix
    /* For 2x2 MIMO matrix, we compute
     * *        |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
     * * H_h_H= |                                                                 |
     * *        |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
     * *
     * *inv(H_h_H) =(1/det)*[d  -b
     * *                     -c  a]
     * **************************************************************************/
    for (int rtx=0;rtx<size;rtx++) {//row
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      for (int ctx=0;ctx<size;ctx++) {//column
        k=0;
        for(int cctx=0;cctx<size;cctx++)
          if(cctx != ctx) cc[k++] = cctx;

        //fill out the sub matrix corresponds to this element
        for (int ridx=0;ridx<(size-1);ridx++)
          for (int cidx=0;cidx<(size-1);cidx++)
            // To verify
            sub_matrix[cidx][ridx]=a44[cc[cidx]][rr[ridx]];

        nr_determin(size - 1, // size
                    sub_matrix,
                    inv_H_h_H[rtx][ctx], // out transpose
                    nb_rb,
                    ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1),
                    shift0);
      }
    }
  }
  else {//floating point calc.
    //Allocate the submatrix elements
    double complex sub_matrix_cpx[size - 1][size - 1];
    //Convert the IQ samples (in Q15 format) to float complex
    double complex a44_cpx[size][size];
    double complex inv_H_h_H_cpx[size][size];
    double complex determin_cpx;
    for (int i=0; i<12*nb_rb; i++) {

      //Convert Q15 to floating point
      for (int rtx=0;rtx<size;rtx++) {//row
        for (int ctx=0;ctx<size;ctx++) {//column
          a44_cpx[ctx][rtx] =
              ((double)(a44[ctx][rtx])[i].r) / (1 << (shift0 - 1)) + I * ((double)(a44[ctx][rtx])[i].i) / (1 << (shift0 - 1));
        }
      }
      //Compute Matrix determinant (copy real value only)
      determin_cpx = nr_determin_cpx(size,
                                     a44_cpx, //
                                     +1);
      //if (i<4) printf("order %d nr_det_cpx = %lf+j%lf \n",log2_approx(creal(determin_cpx)),creal(determin_cpx),cimag(determin_cpx));

      //Round and convert to Q15 (Out in the same format as Fixed point).
      if (creal(determin_cpx)>0) {//determin of the symmetric matrix is real part only
        ((short *)ad_bc)[i << 1] = (short)((creal(determin_cpx) * (1 << (shift0))) + 0.5); //
      } else {
        ((short *)ad_bc)[i << 1] = (short)((creal(determin_cpx) * (1 << (shift0))) - 0.5); //
      }
      //Compute Inversion of the H^*H matrix (normalized output divide by determinant)
      for (int rtx=0;rtx<size;rtx++) {//row
        k=0;
        for(int rrtx=0;rrtx<size;rrtx++)
          if(rrtx != rtx) rr[k++] = rrtx;
        for (int ctx=0;ctx<size;ctx++) {//column
          k=0;
          for(int cctx=0;cctx<size;cctx++)
            if(cctx != ctx) cc[k++] = cctx;

          //fill out the sub matrix corresponds to this element
          for (int ridx=0;ridx<(size-1);ridx++)
            for (int cidx=0;cidx<(size-1);cidx++)
              sub_matrix_cpx[cidx][ridx] = a44_cpx[cc[cidx]][rr[ridx]];

          inv_H_h_H_cpx[rtx][ctx] = nr_determin_cpx(size - 1, // size,
                                                    sub_matrix_cpx, //
                                                    ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1));
          //if (i==0) printf("H_h_H(r%d,c%d)=%lf+j%lf --> inv_H_h_H(%d,%d) = %lf+j%lf \n",rtx,ctx,creal(a44_cpx[ctx*size+rtx]),cimag(a44_cpx[ctx*size+rtx]),ctx,rtx,creal(inv_H_h_H_cpx[rtx*size+ctx]),cimag(inv_H_h_H_cpx[rtx*size+ctx]));

          if (creal(inv_H_h_H_cpx[rtx][ctx]) > 0)
            inv_H_h_H[rtx][ctx][i].r = (short)((creal(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) + 0.5); // Convert to Q 18
          else
            inv_H_h_H[rtx][ctx][i].r = (short)((creal(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) - 0.5); //

          if (cimag(inv_H_h_H_cpx[rtx][ctx]) > 0)
            inv_H_h_H[rtx][ctx][i].i = (short)((cimag(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) + 0.5); //
          else
            inv_H_h_H[rtx][ctx][i].i = (short)((cimag(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) - 0.5); //

          //if (i<4) printf("inv_H_h_H_FP(%d,%d)= %d+j%d \n",ctx,rtx, ((short *) inv_H_h_H[rtx*size+ctx])[i<<1],((short *) inv_H_h_H[rtx*size+ctx])[(i<<1)+1]);
        }
      }
    }
  }
  return(0);
}

/* Zero Forcing Rx function: nr_conjch0_mult_ch1()
 *
 *
 * */
// TODO: This function is just a wrapper, can be removed.
void nr_conjch0_mult_ch1(c16_t *ch0, c16_t *ch1, c16_t *ch0conj_ch1, unsigned short nb_rb, unsigned char output_shift0)
{
  //This function is used to compute multiplications in H_hermitian * H matrix
  mult_cpx_conj_vector(ch0, ch1, ch0conj_ch1, 12 * nb_rb, output_shift0);
}

/*
 * MMSE Rx function: up to 4 layers
 */
void nr_dlsch_mmse(const int nl,
                   const int n_rx,
                   const int nb_rb,
                   const int length,
                   const int mod_order,
                   const int shift,
                   const uint32_t noise_var,
                   const int vecSz,
                   const c16_t dl_ch_estimates_ext[nl][n_rx][vecSz],
                   c16_t rxdataF_comp[nl][n_rx][vecSz],
                   c16_t dl_ch_mag[nl][n_rx][vecSz],
                   c16_t dl_ch_magb[nl][n_rx][vecSz],
                   c16_t dl_ch_magr[nl][n_rx][vecSz])
{
  uint32_t nb_rb_0 = (length + 11) / 12;
  c16_t determ_fin[12 * nb_rb_0] __attribute__((aligned(32)));

  ///Allocate H^*H matrix elements and sub elements
  c16_t conjH_H_elements_data[n_rx][nl][nl][12 * nb_rb_0];
  memset(conjH_H_elements_data, 0, sizeof(conjH_H_elements_data));
  c16_t *conjH_H_elements[n_rx][nl][nl];
  for (int aarx = 0; aarx < n_rx; aarx++)
    for (int rtx = 0; rtx < nl; rtx++)
      for (int ctx = 0; ctx < nl; ctx++)
        conjH_H_elements[aarx][rtx][ctx] = conjH_H_elements_data[aarx][rtx][ctx];

  //Compute H^*H matrix elements and sub elements:(1/2^log2_maxh)*conjH_H_elements
  for (int rtx = 0; rtx < nl; rtx++) {//row
    for (int ctx = 0; ctx < nl; ctx++) {//column
      for (int aarx = 0; aarx < n_rx; aarx++)  {
        c16_t *ch0r = (c16_t *)dl_ch_estimates_ext[rtx][aarx];
        c16_t *ch0c = (c16_t *)dl_ch_estimates_ext[ctx][aarx];
        nr_conjch0_mult_ch1(ch0r,
                            ch0c,
                            conjH_H_elements[aarx][ctx][rtx], // sic
                            nb_rb_0,
                            shift);
        if (aarx != 0)
          nr_a_sum_b(conjH_H_elements[0][ctx][rtx], conjH_H_elements[aarx][ctx][rtx], nb_rb_0);
      }
    }
  }

  // Add noise_var such that: H^h * H + noise_var * I
  if (noise_var != 0) {
    simde__m128i nvar_128i = simde_mm_set1_epi32(noise_var >> 3);
    for (int p = 0; p < nl; p++) {
      simde__m128i *conjH_H_128i = (simde__m128i *)conjH_H_elements[0][p][p];
      for (int k = 0; k < 3 * nb_rb_0; k++) {
        conjH_H_128i[0] = simde_mm_add_epi32(conjH_H_128i[0], nvar_128i);
        conjH_H_128i++;
      }
    }
  }

  //Compute the inverse and determinant of the H^*H matrix
  //Allocate the inverse matrix
  c16_t *inv_H_h_H[nl][nl];
  c16_t inv_H_h_H_data[nl][nl][12 * nb_rb_0];
  memset(inv_H_h_H_data, 0, sizeof(inv_H_h_H_data));
  for (int rtx = 0; rtx < nl; rtx++)
    for (int ctx = 0; ctx < nl; ctx++)
      inv_H_h_H[ctx][rtx] = inv_H_h_H_data[ctx][rtx];

  int fp_flag = 1;//0: float point calc 1: Fixed point calc
  nr_matrix_inverse(nl,
                    conjH_H_elements[0], // Input matrix
                    inv_H_h_H, // Inverse
                    determ_fin, // determin
                    nb_rb_0,
                    fp_flag, // fixed point flag
                    shift - (fp_flag == 1 ? 1 : 0)); // the out put is Q15

  // multiply Matrix inversion pf H_h_H by the rx signal vector
  c16_t outtemp[12 * nb_rb_0] __attribute__((aligned(32)));
  //Allocate rxdataF for zforcing out
  c16_t rxdataF_zforcing[nl][12 * nb_rb_0];
  memset(rxdataF_zforcing, 0, sizeof(rxdataF_zforcing));

  for (int rtx = 0; rtx < nl; rtx++) {//Output Layers row
    // loop over Layers rtx=0,...,N_Layers-1
    for (int ctx = 0; ctx < nl; ctx++) { // column multi
      // printf("Computing r_%d c_%d\n",rtx,ctx);
      // print_shorts(" H_h_H=",(int16_t*)&conjH_H_elements[ctx*nl+rtx][0][0]);
      // print_shorts(" Inv_H_h_H=",(int16_t*)&inv_H_h_H[ctx*nl+rtx][0]);
      mult_complex_vectors(inv_H_h_H[ctx][rtx],
                           (c16_t *)(rxdataF_comp[ctx][0]),
                           outtemp,
                           sizeofArray(outtemp),
                           shift - (fp_flag == 1 ? 1 : 0));
      nr_a_sum_b(rxdataF_zforcing[rtx], outtemp, nb_rb_0); // a = a + b
    }
#ifdef DEBUG_DLSCH_DEMOD
    printf("Computing layer_%d \n", rtx);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][0]);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][4]);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][8]);
#endif
  }

  //Copy zero_forcing out to output array
  for (int rtx = 0; rtx < nl; rtx++)
    nr_element_sign(rxdataF_zforcing[rtx], (c16_t *)(rxdataF_comp[rtx][0]), nb_rb_0, + 1);

  //Update LLR thresholds with the Matrix determinant
  simde__m128i *dl_ch_mag128_0=NULL,*dl_ch_mag128b_0=NULL,*dl_ch_mag128r_0=NULL,*determ_fin_128;
  simde__m128i mmtmpD2,mmtmpD3;
  simde__m128i QAM_amp128={0},QAM_amp128b={0},QAM_amp128r={0};
  short nr_realpart[8]__attribute__((aligned(16))) = {1,0,1,0,1,0,1,0};
  determ_fin_128      = (simde__m128i *)&determ_fin[0];

  if (mod_order > 2) {
    if (mod_order == 4) {
      QAM_amp128 = simde_mm_set1_epi16(QAM16_n1);  //2/sqrt(10)
      QAM_amp128b = simde_mm_setzero_si128();
      QAM_amp128r = simde_mm_setzero_si128();
    } else if (mod_order == 6) {
      QAM_amp128  = simde_mm_set1_epi16(QAM64_n1); //4/sqrt{42}
      QAM_amp128b = simde_mm_set1_epi16(QAM64_n2); //2/sqrt{42}
      QAM_amp128r = simde_mm_setzero_si128();
    } else if (mod_order == 8) {
      QAM_amp128 = simde_mm_set1_epi16(QAM256_n1); //8/sqrt{170}
      QAM_amp128b = simde_mm_set1_epi16(QAM256_n2);//4/sqrt{170}
      QAM_amp128r = simde_mm_set1_epi16(QAM256_n3);//2/sqrt{170}
    }
    dl_ch_mag128_0 = (simde__m128i *)dl_ch_mag[0][0];
    dl_ch_mag128b_0 = (simde__m128i *)dl_ch_magb[0][0];
    dl_ch_mag128r_0 = (simde__m128i *)dl_ch_magr[0][0];

    for (int rb = 0; rb < 3 * nb_rb_0; rb++) {
      //for symmetric H_h_H matrix, the determinant is only real values
      mmtmpD2 = simde_mm_sign_epi16(determ_fin_128[0],*(simde__m128i*)&nr_realpart[0]);//set imag part to 0
      mmtmpD3 = simde_mm_shufflelo_epi16(mmtmpD2,SIMDE_MM_SHUFFLE(2,3,0,1));
      mmtmpD3 = simde_mm_shufflehi_epi16(mmtmpD3,SIMDE_MM_SHUFFLE(2,3,0,1));
      mmtmpD2 = simde_mm_add_epi16(mmtmpD2,mmtmpD3);

      dl_ch_mag128_0[0] = mmtmpD2;
      dl_ch_mag128b_0[0] = mmtmpD2;
      dl_ch_mag128r_0[0] = mmtmpD2;

      dl_ch_mag128_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128_0[0], QAM_amp128);
      dl_ch_mag128b_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128b_0[0],QAM_amp128b);
      dl_ch_mag128r_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128r_0[0],QAM_amp128r);

      determ_fin_128 += 1;
      dl_ch_mag128_0 += 1;
      dl_ch_mag128b_0 += 1;
      dl_ch_mag128r_0 += 1;
    }
  }
}

void nr_dlsch_layer_demapping(const uint8_t Nl,
                              const uint8_t mod_order,
                              const int llrLayerSize,
                              const int16_t llr_layers[NR_SYMBOLS_PER_SLOT][NR_MAX_NB_LAYERS][llrLayerSize],
                              const NR_UE_DLSCH_t *dlsch,
                              const int32_t re_len[NR_SYMBOLS_PER_SLOT],
                              int16_t *llr)
{
  const int s0 = dlsch->dlsch_config.start_symbol;
  const int s1 = dlsch->dlsch_config.number_symbols;

  int k = 0;
  switch (Nl) {
    case 1:
      for (int i = s0; i < (s0 + s1); i++) {
        memcpy(llr + k, llr_layers[i][0], re_len[i] * mod_order * sizeof(int16_t));
        k += re_len[i] * mod_order;
      }
      break;

    case 2:
    case 3:
    case 4:
      for (int i = s0; i < (s0 + s1); i++) {
        int m = 0;
        for (int j = 0; j < re_len[i]; j++) {
          for (int l = 0; l < Nl; l++) {
            memcpy(llr + k, llr_layers[i][l] + m * mod_order, sizeof(int16_t) * mod_order);
            k += mod_order;
            // if (i<4) printf("length%d: llr_layers[l%d][m%d]=%d: \n",length,l,m,llr_layers[l][i*mod_order+m]);
          }
          m++;
        }
      }
      break;

    default:
      AssertFatal(0, "Not supported number of layers %d\n", Nl);
  }
}

/* Computes LLRs from compensated PDSCH signal per OFDM symbol for all layers */
int nr_dlsch_llr(const NR_UE_DLSCH_t *dlsch,
                 const int len,
                 const c16_t dl_ch_mag[dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
                 const c16_t dl_ch_magb[dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
                 const c16_t dl_ch_magr[dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
                 const int nb_antennas_rx,
                 const c16_t rxdataF_comp[dlsch->Nl][nb_antennas_rx][dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB],
                 const int llrSize,
                 int16_t layer_llr[dlsch->Nl][llrSize])
{
  switch (dlsch->dlsch_config.qamModOrder) {
    case 2 :
      for(int l=0; l < dlsch[0].Nl; l++)
        nr_qpsk_llr((int32_t *)rxdataF_comp[l][0], layer_llr[l], len);
      break;

    case 4 :
      for(int l=0; l < dlsch[0].Nl; l++)
        nr_16qam_llr((int32_t *)rxdataF_comp[l][0], (int32_t *)dl_ch_mag, layer_llr[l], len);
      break;

    case 6 :
      for(int l=0; l < dlsch[0].Nl; l++)
        nr_64qam_llr((int32_t *)rxdataF_comp[l][0], (int32_t *)dl_ch_mag, (int32_t *)dl_ch_magb, layer_llr[l], len);
      break;

    case 8:
      for(int l=0; l < dlsch[0].Nl; l++)
        nr_256qam_llr((int32_t *)rxdataF_comp[l][0], (int32_t *)dl_ch_mag, (int32_t *)dl_ch_magb, (int32_t *)dl_ch_magr, layer_llr[l], len);
      break;

    default:
      AssertFatal(false, "Unknown mod_order!!!!\n");
      break;
  }

  return 0;
}
//==============================================================================================

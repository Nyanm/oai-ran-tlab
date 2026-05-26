/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/***********************************************************************
*
* FILENAME    :  csi_rx.c
*
* MODULE      :
*
* DESCRIPTION :  function to receive the channel state information
*
************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "executables/nr-softmodem-common.h"
#include "nr_transport_proto_ue.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "common/utils/nr/nr_common.h"
#include "PHY/NR_UE_ESTIMATION/filt16a_32.h"

//#define NR_CSIRS_DEBUG
//#define NR_CSIIM_DEBUG

extern openair0_config_t openair0_cfg[MAX_CARDS];

static void nr_det_A_2x2(c16_t *a_mf_00,
                         c16_t *a_mf_01,
                         c16_t *a_mf_10,
                         c16_t *a_mf_11,
                         int32_t *det_fin,
                         const unsigned short nb_rb)
{
  simde__m128i *a_mf_00_128 = (simde__m128i *)a_mf_00;
  simde__m128i *a_mf_01_128 = (simde__m128i *)a_mf_01;
  simde__m128i *a_mf_10_128 = (simde__m128i *)a_mf_10;
  simde__m128i *a_mf_11_128 = (simde__m128i *)a_mf_11;
  simde__m128i *det_fin_128 = (simde__m128i *)det_fin;

  for (int rb = 0; rb < 3 * nb_rb; rb++) {
    // complex multiplication (I_a+jQ_a)(I_d+jQ_d) = (I_aI_d - Q_aQ_d) + j(Q_aI_d + I_aQ_d)
    // The imag part is often zero, we compute only the real part
    simde__m128i ad_re_128 = simde_mm_madd_epi16(oai_mm_conj(a_mf_00_128[0]), a_mf_11_128[0]); // Re: I_a0*I_d0 - Q_a1*Q_d1

    // complex multiplication (I_b+jQ_b)(I_c+jQ_c) = (I_bI_c - Q_bQ_c) + j(Q_bI_c + I_bQ_c)
    // The imag part is often zero, we compute only the real part
    simde__m128i bc_re_128 = simde_mm_madd_epi16(oai_mm_conj(a_mf_01_128[0]), a_mf_10_128[0]); // Re: I_b0*I_c0 - Q_b1*Q_c1

    simde__m128i det_re_128 = simde_mm_sub_epi32(ad_re_128, bc_re_128);

    // det in Q30 format
    det_fin_128[0] = simde_mm_abs_epi32(det_re_128);

    det_fin_128 += 1;
    a_mf_00_128 += 1;
    a_mf_01_128 += 1;
    a_mf_10_128 += 1;
    a_mf_11_128 += 1;
  }
}

/*
 * nr_sq_matrix_elem(): Calculate the squares of the elements of a matrix
 */
static void nr_sq_matrix_elem(c16_t *a, int32_t *a_sq, const unsigned short nb_rb)
{
  simde__m128i *a_128 = (simde__m128i *)a;
  simde__m128i *a_sq_128 = (simde__m128i *)a_sq;
  for (int rb = 0; rb < 3 * nb_rb; rb++) {
    a_sq_128[0] = simde_mm_madd_epi16(a_128[0], a_128[0]);
    a_sq_128 += 1;
    a_128 += 1;
  }
}

/*
 * Frobenius norm^2 of A
 */
static void nr_frob_norm_2x2(int32_t *a_00_sq,
                             int32_t *a_01_sq,
                             int32_t *a_10_sq,
                             int32_t *a_11_sq,
                             int32_t *num_fin,
                             const unsigned short nb_rb)
{
  simde__m128i *a_00_sq_128 = (simde__m128i *)a_00_sq;
  simde__m128i *a_01_sq_128 = (simde__m128i *)a_01_sq;
  simde__m128i *a_10_sq_128 = (simde__m128i *)a_10_sq;
  simde__m128i *a_11_sq_128 = (simde__m128i *)a_11_sq;
  simde__m128i *num_fin_128 = (simde__m128i *)num_fin;
  for (int rb = 0; rb < 3 * nb_rb; rb++) {
    simde__m128i sq_a_plus_sq_d_128 = simde_mm_add_epi32(a_00_sq_128[0], a_11_sq_128[0]);
    simde__m128i sq_b_plus_sq_c_128 = simde_mm_add_epi32(a_01_sq_128[0], a_10_sq_128[0]);
    num_fin_128[0] = simde_mm_add_epi32(sq_a_plus_sq_d_128, sq_b_plus_sq_c_128);
    num_fin_128 += 1;
    a_00_sq_128 += 1;
    a_01_sq_128 += 1;
    a_10_sq_128 += 1;
    a_11_sq_128 += 1;
  }
}

bool is_csi_rs_in_symbol(const fapi_nr_dl_config_csirs_pdu_rel15_t csirs_config_pdu, const int symbol) {

  bool ret = false;

  // 38.211-Table 7.4.1.5.3-1: CSI-RS locations within a slot
  switch(csirs_config_pdu.row){
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 9:
      if(symbol == csirs_config_pdu.symb_l0) {
        ret = true;
      }
      break;
    case 5:
    case 7:
    case 8:
    case 10:
    case 11:
    case 12:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) ) {
        ret = true;
      }
      break;
    case 13:
    case 14:
    case 16:
    case 17:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) ||
          symbol == csirs_config_pdu.symb_l1 || symbol == (csirs_config_pdu.symb_l1+1)) {
        ret = true;
      }
      break;
    case 15:
    case 18:
      if(symbol == csirs_config_pdu.symb_l0 || symbol == (csirs_config_pdu.symb_l0+1) || symbol == (csirs_config_pdu.symb_l0+2) ) {
        ret = true;
      }
      break;
    default:
      AssertFatal(0==1, "Row %d is not valid for CSI Table 7.4.1.5.3-1\n", csirs_config_pdu.row);
  }

  return ret;
}

static int nr_get_csi_rs_signal(const PHY_VARS_NR_UE *ue,
                                const UE_nr_rxtx_proc_t *proc,
                                const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                                const nr_csi_info_t *nr_csi_info,
                                const csi_mapping_parms_t *csi_mapping,
                                const int CDM_group_size,
                                c16_t csi_rs_received_signal[][ue->frame_parms.samples_per_slot_wCP],
                                uint32_t *rsrp,
                                int *rsrp_dBm,
                                const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP])
{
  const NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  uint16_t meas_count = 0;
  uint32_t rsrp_sum = 0;

  for (int ant_rx = 0; ant_rx < fp->nb_antennas_rx; ant_rx++) {

    for (int rb = csirs_config_pdu->start_rb; rb < (csirs_config_pdu->start_rb+csirs_config_pdu->nr_of_rbs); rb++) {

      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      for (int cdm_id = 0; cdm_id < csi_mapping->size; cdm_id++) {
        for (int s = 0; s < CDM_group_size; s++)  {

          // loop over frequency resource elements within a group
          for (int kp = 0; kp <= csi_mapping->kprime; kp++) {

            uint16_t k = (fp->first_carrier_offset + (rb * NR_NB_SC_PER_RB) + csi_mapping->koverline[cdm_id] + kp) % fp->ofdm_symbol_size;

            // loop over time resource elements within a group
            for (int lp = 0; lp <= csi_mapping->lprime; lp++) {
              uint16_t symb = lp + csi_mapping->loverline[cdm_id];
              uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
              const c16_t *rx_signal = &rxdataF[ant_rx][symbol_offset];
              c16_t *rx_csi_rs_signal = &csi_rs_received_signal[ant_rx][symbol_offset];
              rx_csi_rs_signal[k].r = rx_signal[k].r;
              rx_csi_rs_signal[k].i = rx_signal[k].i;

              rsrp_sum += (((int32_t)(rx_csi_rs_signal[k].r)*rx_csi_rs_signal[k].r) +
                           ((int32_t)(rx_csi_rs_signal[k].i)*rx_csi_rs_signal[k].i));

              meas_count++;

#ifdef NR_CSIRS_DEBUG
              int dataF_offset = proc->nr_slot_rx * fp->samples_per_slot_wCP;
              uint16_t port_tx = s + csi_mapping->j[cdm_id] * CDM_group_size;
              c16_t *tx_csi_rs_signal = &nr_csi_info->csi_rs_generated_signal[port_tx][symbol_offset + dataF_offset];
              LOG_I(NR_PHY,
                    "l,k (%2d,%4d) |\tport_tx %d (%4d,%4d)\tant_rx %d (%4d,%4d)\n",
                    symb,
                    k,
                    port_tx+3000,
                    tx_csi_rs_signal[k].r,
                    tx_csi_rs_signal[k].i,
                    ant_rx,
                    rx_csi_rs_signal[k].r,
                    rx_csi_rs_signal[k].i);
#else
              UNUSED(proc);
              UNUSED(nr_csi_info);
#endif
            }
          }
        }
      }
    }
  }


  *rsrp = rsrp_sum/meas_count;
  *rsrp_dBm = dB_fixed(*rsrp) + 30 - SQ15_SQUARED_NORM_FACTOR_DB
              - ((int)openair0_cfg[ue->rf_map.card].rx_gain[0] - (int)openair0_cfg[ue->rf_map.card].rx_gain_offset[0])
              - dB_fixed(ue->frame_parms.ofdm_symbol_size);

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "RSRP = %i (%i dBm)\n", *rsrp, *rsrp_dBm);
#endif

  return 0;
}

uint32_t calc_power_csirs(const uint16_t *x, const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu)
{
  uint64_t sum_x = 0;
  uint64_t sum_x2 = 0;
  uint16_t size = 0;
  for (int rb = 0; rb < csirs_config_pdu->nr_of_rbs; rb++) {
    if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != ((rb + csirs_config_pdu->start_rb) % 2)) {
      continue;
    }
    sum_x = sum_x + x[rb];
    sum_x2 = sum_x2 + x[rb] * x[rb];
    size++;
  }
  return sum_x2 / size - (sum_x / size) * (sum_x / size);
}

static int nr_csi_rs_channel_estimation(
    const NR_DL_FRAME_PARMS *fp,
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
    const nr_csi_info_t *nr_csi_info,
    const c16_t **csi_rs_generated_signal,
    const c16_t csi_rs_received_signal[][fp->samples_per_slot_wCP],
    const csi_mapping_parms_t *csi_mapping,
    const int CDM_group_size,
    c16_t csi_rs_ls_estimated_channel[][csi_mapping->ports][fp->ofdm_symbol_size],
    c16_t csi_rs_estimated_channel_freq[][csi_mapping->ports][fp->ofdm_symbol_size],
    int16_t *log2_re,
    int16_t *log2_maxh,
    uint32_t *noise_power)
{
  *noise_power = 0;
  int maxh = 0;
  int count = 0;

  for (int ant_rx = 0; ant_rx < fp->nb_antennas_rx; ant_rx++) {

    /// LS channel estimation

    const uint16_t stop_rb = csirs_config_pdu->start_rb + csirs_config_pdu->nr_of_rbs;
    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      memset(csi_rs_ls_estimated_channel[ant_rx][port_tx], 0, fp->ofdm_symbol_size * sizeof(c16_t));
    }

    for (int rb = csirs_config_pdu->start_rb; rb < stop_rb; rb++) {
      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      for (int cdm_id = 0; cdm_id < csi_mapping->size; cdm_id++) {
        for (int s = 0; s < CDM_group_size; s++)  {

          uint16_t port_tx = s + csi_mapping->j[cdm_id] * CDM_group_size;

          // loop over frequency resource elements within a group
          for (int kp = 0; kp <= csi_mapping->kprime; kp++) {
            uint16_t kinit_rx = (fp->first_carrier_offset + rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
            uint16_t k_rx = kinit_rx + csi_mapping->koverline[cdm_id] + kp;
            uint16_t kinit_tx = rb * NR_NB_SC_PER_RB;
            uint16_t k_tx = kinit_tx + csi_mapping->koverline[cdm_id] + kp;

            // loop over time resource elements within a group
            for (int lp = 0; lp <= csi_mapping->lprime; lp++) {
              uint16_t symb = lp + csi_mapping->loverline[cdm_id];
              uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
              const c16_t *tx_csi_rs_signal = &csi_rs_generated_signal[port_tx][symbol_offset];
              const c16_t *rx_csi_rs_signal = &csi_rs_received_signal[ant_rx][symbol_offset];
              c16_t tmp =
                  c16MulConjShift(tx_csi_rs_signal[k_tx], rx_csi_rs_signal[k_rx], nr_csi_info->csi_rs_generated_signal_bits);
              // This is not just the LS estimation for each (k,l), but also the sum of the different contributions
              // for the sake of optimizing the memory used.
              csi_rs_ls_estimated_channel[ant_rx][port_tx][kinit_tx].r += tmp.r;
              csi_rs_ls_estimated_channel[ant_rx][port_tx][kinit_tx].i += tmp.i;
            }
          }
        }
      }
    }

#ifdef NR_CSIRS_DEBUG
    for(int symb = 0; symb < fp->symbols_per_slot; symb++) {
      if(!is_csi_rs_in_symbol(*csirs_config_pdu,symb)) {
        continue;
      }
      for(int k = 0; k < fp->ofdm_symbol_size; k++) {
        LOG_I(NR_PHY, "l,k (%2d,%4d) | ", symb, k);
        for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
          uint64_t symbol_offset = symb * fp->ofdm_symbol_size;
          c16_t *tx_csi_rs_signal = (c16_t*)&csi_rs_generated_signal[port_tx][symbol_offset+dataF_offset];
          c16_t *rx_csi_rs_signal = (c16_t*)&csi_rs_received_signal[ant_rx][symbol_offset];
          c16_t *csi_rs_ls_estimated_channel16 = csi_rs_ls_estimated_channel[ant_rx][port_tx];
          printf("port_tx %d --> ant_rx %d, tx (%4d,%4d), rx (%4d,%4d), ls (%4d,%4d) | ",
                 port_tx+3000, ant_rx,
                 tx_csi_rs_signal[k].r, tx_csi_rs_signal[k].i,
                 rx_csi_rs_signal[k].r, rx_csi_rs_signal[k].i,
                 csi_rs_ls_estimated_channel16[k].r, csi_rs_ls_estimated_channel16[k].i);
        }
        printf("\n");
      }
    }
#endif

    /// Channel interpolation

    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      memset(csi_rs_estimated_channel_freq[ant_rx][port_tx], 0, (fp->ofdm_symbol_size) * sizeof(c16_t));
    }

    for (int rb = csirs_config_pdu->start_rb; rb < stop_rb; rb++) {
      // for freq density 0.5 checks if even or odd RB
      if(csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }

      count++;

      uint16_t k = rb * NR_NB_SC_PER_RB;
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t csi_rs_ls_estimated_channel16 = csi_rs_ls_estimated_channel[ant_rx][port_tx][k];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][k];
        if (k == 0) { // Start of OFDM symbol case or first occupied subcarrier case
          multadd_real_vector_complex_scalar(filt24_start, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16, 24);
        } else if (rb == (stop_rb - 1)) { // End of OFDM symbol case or Last occupied subcarrier case
          multadd_real_vector_complex_scalar(filt24_end, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16 - 12, 24);
        } else { // Middle case
          multadd_real_vector_complex_scalar(filt24_middle, csi_rs_ls_estimated_channel16, csi_rs_estimated_channel16 - 12, 24);
        }
      }
    }

    /// Power noise estimation
    AssertFatal(csirs_config_pdu->nr_of_rbs > 0, " nr_of_rbs needs to be greater than 0\n");
    uint16_t noise_real[fp->nb_antennas_rx][csi_mapping->ports][csirs_config_pdu->nr_of_rbs];
    uint16_t noise_imag[fp->nb_antennas_rx][csi_mapping->ports][csirs_config_pdu->nr_of_rbs];
    for (int rb = csirs_config_pdu->start_rb; rb < stop_rb; rb++) {
      if (csirs_config_pdu->freq_density <= 1 && csirs_config_pdu->freq_density != (rb % 2)) {
        continue;
      }
      uint16_t k = rb * NR_NB_SC_PER_RB;
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t *csi_rs_ls_estimated_channel16 = &csi_rs_ls_estimated_channel[ant_rx][port_tx][k];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][k];
        noise_real[ant_rx][port_tx][rb-csirs_config_pdu->start_rb] = abs(csi_rs_ls_estimated_channel16->r-csi_rs_estimated_channel16->r);
        noise_imag[ant_rx][port_tx][rb-csirs_config_pdu->start_rb] = abs(csi_rs_ls_estimated_channel16->i-csi_rs_estimated_channel16->i);
        maxh = cmax3(maxh, abs(csi_rs_estimated_channel16->r), abs(csi_rs_estimated_channel16->i));
      }
    }
    for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
      *noise_power += (calc_power_csirs(noise_real[ant_rx][port_tx], csirs_config_pdu) + calc_power_csirs(noise_imag[ant_rx][port_tx],csirs_config_pdu));
    }

#ifdef NR_CSIRS_DEBUG
    for(int k = 0; k < fp->ofdm_symbol_size; k++) {
      int rb = k / NR_NB_SC_PER_RB;
      LOG_I(NR_PHY, "(k = %4d) |\t", k);
      for(uint16_t port_tx = 0; port_tx < csi_mapping->ports; port_tx++) {
        c16_t *csi_rs_ls_estimated_channel16 = &csi_rs_ls_estimated_channel[ant_rx][port_tx][0];
        c16_t *csi_rs_estimated_channel16 = &csi_rs_estimated_channel_freq[ant_rx][port_tx][0];
        printf("Channel port_tx %d --> ant_rx %d : ls (%4d,%4d), int (%4d,%4d), noise (%4d,%4d) | ",
               port_tx + 3000,
               ant_rx,
               csi_rs_ls_estimated_channel16[k].r,
               csi_rs_ls_estimated_channel16[k].i,
               csi_rs_estimated_channel16[k].r,
               csi_rs_estimated_channel16[k].i,
               rb >= stop_rb ? 0 : noise_real[ant_rx][port_tx][rb - csirs_config_pdu->start_rb],
               rb >= stop_rb ? 0 : noise_imag[ant_rx][port_tx][rb - csirs_config_pdu->start_rb]);
      }
      printf("\n");
    }
#endif

  }

  *noise_power /= (fp->nb_antennas_rx * csi_mapping->ports);
  *log2_maxh = log2_approx(maxh - 1);
  *log2_re = log2_approx(count - 1);

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "Noise power estimation based on CSI-RS: %i\n", *noise_power);
#endif
  return 0;
}

/*
 * Rank Indicator (RI) estimation based on the condition number of the CSI-RS channel correlation matrix.
 *
 * For the estimated MIMO channel H (rows -> UE receive antenna, columns -> transmit layers), the algorithm computes either:
 *      A = H^H x H
 * or equivalently:
 *      A = H x H^H
 * depending on which dimension is smaller, such that A remains 2x2.
 *
 * In the 2x2 case:
 *      A = | a b |
 *          | c d |
 *
 * The condition metric is approximated as:
 *                           ||A||²_F
 *      cond_dB = 10log10( ------------ )
 *                            det(A)
 * where:
 *      ||A||²_F = |a|² + |b|² + |c|² + |d|²
 * and
 *      det(A) = ad - bc
 *
 * This metric is related to the matrix condition number:
 *      lambda_max / lambda_min
 * where lambda_max and lambda_min are the largest and smallest eigenvalues of A.
 *
 * Similar eigenvalues indicate low spatial correlation and good layer separability (RI = 2), while highly unbalanced eigenvalues
 * indicate an ill-conditioned channel and rank-1 preference.
 *
 * The condition metric is evaluated for each CSI-RS RE and compared against a fixed threshold (5 dB).
 */
static int nr_csi_rs_ri_estimation_2(int nb_antennas_rx,
                                     int N_ports,
                                     int ofdm_sz,
                                     const c16_t ch_freq[][N_ports][ofdm_sz],
                                     int start_rb,
                                     int nr_of_rbs,
                                     int freq_density,
                                     int16_t log2_maxh)
{
  // Choose between H^H*H (sum over RX antennas) and H*H^H (sum over TX ports)
  // so that the resulting A matrix stays 2x2 regardless of the asymmetric dimension.
  const bool is_HhxH = (N_ports <= nb_antennas_rx);
  const int A_dim = 2; // min(nb_antennas_rx, N_ports), always 2 here
  const int outer_dim = is_HhxH ? nb_antennas_rx : N_ports; // dimension being summed over

  // A 12 dB threshold was chosen to allow rank-2 operation also in moderately correlated channels.
  // A fairly conservative value for many real-world scenarios would be 5. While 18 would almost always give RI = 2.
  // Add 3 dB since ||A||^2_F / det(A) has a theoretical 3 dB floor for an ideal 2x2 channel, i.e., 12 + 3 = 15.
  const int cond_dB_threshold = 15;
  int count = 0;

  // 2x2 correlation matrix A and per-subcarrier intermediates.
  c16_t A[A_dim][A_dim][ofdm_sz] __attribute__((aligned(32)));
  memset(A, 0, sizeof(A));
  int32_t A_sq[A_dim][A_dim][ofdm_sz] __attribute__((aligned(32)));
  int32_t det_A[ofdm_sz] __attribute__((aligned(32)));
  int32_t A_frob_norm_sq[ofdm_sz] __attribute__((aligned(32)));

  // Scratch buffer for one conj-product, reused across the (i, j, outer) triple loop.
  c16_t conjch_ch[ofdm_sz] __attribute__((aligned(32)));

  for (int rb = start_rb; rb < start_rb + nr_of_rbs; rb++) {
    if (freq_density <= 1 && freq_density != (rb % 2))
      continue;
    const int k = rb * NR_NB_SC_PER_RB;

    // Accumulate A[i][j] over the outer dimension:
    //   is_HhxH : A[i][j] = sum_{m} conj(ch[m][i]) * ch[m][j]      (m = RX antenna)
    //   !is_HhxH: A[i][j] = sum_{m} conj(ch[i][m]) * ch[j][m]      (m = TX port; gives (H*H^H)^T)
    for (int idx_outer = 0; idx_outer < outer_dim; idx_outer++) {
      for (int i = 0; i < A_dim; i++) {
        for (int j = 0; j < A_dim; j++) {
          const c16_t *ch_for_conj = is_HhxH ? &ch_freq[idx_outer][i][k] : &ch_freq[i][idx_outer][k];
          const c16_t *ch_plain = is_HhxH ? &ch_freq[idx_outer][j][k] : &ch_freq[j][idx_outer][k];
          // conjch_ch = conj(ch_for_conj) * ch_plain over 1 RB (12 subcarriers)
          mult_cpx_conj_vector((c16_t *)ch_for_conj, (c16_t *)ch_plain, &conjch_ch[k], NR_NB_SC_PER_RB, log2_maxh);
          // A[i][j] += conjch_ch
          nr_a_sum_b(&A[i][j][k], &conjch_ch[k], 1);
        }
      }
    }

    // Determinant of A (denominator of the condition metric).
    nr_det_A_2x2(&A[0][0][k], &A[0][1][k], &A[1][0][k], &A[1][1][k], &det_A[k], 1);

    // Per-element |A_ij|^2 and their sum -> Frobenius norm^2 of A (numerator).
    nr_sq_matrix_elem(&A[0][0][k], &A_sq[0][0][k], 1);
    nr_sq_matrix_elem(&A[0][1][k], &A_sq[0][1][k], 1);
    nr_sq_matrix_elem(&A[1][0][k], &A_sq[1][0][k], 1);
    nr_sq_matrix_elem(&A[1][1][k], &A_sq[1][1][k], 1);
    nr_frob_norm_2x2(&A_sq[0][0][k], &A_sq[0][1][k], &A_sq[1][0][k], &A_sq[1][1][k], &A_frob_norm_sq[k], 1);

#ifdef NR_CSIRS_DEBUG
    for (int i = 0; i < A_dim; i++) {
      for (int j = 0; j < A_dim; j++) {
        c16_t *a_k = &A[i][j][k];
        int32_t *a_sq_k = &A_sq[i][j][k];
        LOG_I(NR_PHY, "A[%i][%i][%i] = (%i, %i)\n", i, j, k, a_k->r, a_k->i);
        LOG_I(NR_PHY, "A_sq[%i][%i][%i] = %i\n", i, j, k, *a_sq_k);
      }
    }
    LOG_I(NR_PHY, "(%i) det_A = %i\n", k, det_A[k]);
    LOG_I(NR_PHY, "(%i) A_frob_norm_sq = %i\n", k, A_frob_norm_sq[k]);
#endif

    // Evaluate the condition metric per RE and run a majority vote.
    for (int sc_idx = 0; sc_idx < NR_NB_SC_PER_RB; sc_idx++) {
      int8_t denum_db = dB_fixed(det_A[k + sc_idx]);
      int8_t numer_db = dB_fixed(A_frob_norm_sq[k + sc_idx]);
      int cond_db = numer_db - denum_db;

#ifdef NR_CSIRS_DEBUG
      LOG_I(NR_PHY, "denum_db = %i, numer_db = %i, cond_db = %i\n", denum_db, numer_db, cond_db);
#endif

      if (cond_db < cond_dB_threshold)
        count++;
      else
        count--;
    }
  }

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "count = %i\n", count);
#endif

  // Rank 2 if the channel is well-conditioned in the majority of REs, rank 1 otherwise.
  if (count > 0) {
#ifdef NR_CSIRS_DEBUG
    LOG_I(NR_PHY, "rank = 2\n");
#endif
    return 1;
  } else {
#ifdef NR_CSIRS_DEBUG
    LOG_I(NR_PHY, "rank = 1\n");
#endif
    return 0;
  }
}

static int nr_csi_rs_ri_estimation(const PHY_VARS_NR_UE *ue,
                                   const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                                   const uint8_t N_ports,
                                   const c16_t csi_rs_estimated_channel_freq[][N_ports][ue->frame_parms.ofdm_symbol_size],
                                   const int16_t log2_maxh)
{
  const NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  const int max_rank = min(fp->nb_antennas_rx, N_ports);
  switch (max_rank) {
    case 1:
      return 0;
    case 2:
      return nr_csi_rs_ri_estimation_2(fp->nb_antennas_rx,
                                       N_ports,
                                       fp->ofdm_symbol_size,
                                       csi_rs_estimated_channel_freq,
                                       csirs_config_pdu->start_rb,
                                       csirs_config_pdu->nr_of_rbs,
                                       csirs_config_pdu->freq_density,
                                       log2_maxh);
    default:
      LOG_W(NR_PHY, "Rank indicator computation is not implemented for %i x %i system\n", fp->nb_antennas_rx, N_ports);
      return 0;
  }
}

// Type1 Single Panel PMI indices (TS 38.214 Section 5.2.2.2.1).
typedef struct {
  uint8_t i_1_1; // first DFT beam index (4/8 ports; range depends on N1*O1)
  uint8_t i_1_2; // second DFT beam index (8 ports only; range depends on N2*O2)
  uint8_t i_1_3; // beam-pair / k1 selection (rank>=2 with 4/8 ports)
  uint8_t i_2; // co-phasing index (rank 1: 0..3; rank>=2: 0..1)
} csi_rs_pmi_t;

// DFT codebook: v_l = [1, exp(j*pi*l/4)]^T, l in {0..7}. Q8 fixed-point.
#define PMI_Q 256
static const int vl_r[8] = {256, 181, 0, -181, -256, -181, 0, 181};
static const int vl_i[8] = {0, 181, 256, 181, 0, -181, -256, -181};
// Co-phasing phi_n = exp(j*pi*n/2): pure integer entries.
static const int8_t phi_r[4] = {1, 0, -1, 0};
static const int8_t phi_i[4] = {0, 1, 0, -1};

// SISO case
static int nr_csi_rs_pmi_1port(int nb_antennas_rx,
                               int N_ports,
                               int osz,
                               const c16_t ch_freq[][N_ports][osz],
                               const fapi_nr_dl_config_csirs_pdu_rel15_t *cfg,
                               uint32_t noise,
                               int32_t *precoded_sinr_dB)
{
  int64_t signal_power = 0;
  int num_REs = 0;
  for (int rb = cfg->start_rb; rb < cfg->start_rb + cfg->nr_of_rbs; rb++) {
    if (cfg->freq_density <= 1 && cfg->freq_density != (rb % 2))
      continue;
    int k = rb * NR_NB_SC_PER_RB;
    for (int ant_rx = 0; ant_rx < nb_antennas_rx; ant_rx++) {
      c16_t h = ch_freq[ant_rx][0][k];
      signal_power += (int64_t)h.r * h.r + (int64_t)h.i * h.i;
    }
    num_REs++;
  }
  if (num_REs > 0)
    *precoded_sinr_dB = dB_fixed((signal_power / num_REs) / noise);
  return 0;
}

// Rank 1 or 2;  2 ports (TS 38.214 - Table 5.2.2.2.1-1)
static int nr_csi_rs_pmi_2ports(int nb_antennas_rx,
                                int N_ports,
                                int osz,
                                const c16_t ch_freq[][N_ports][osz],
                                const fapi_nr_dl_config_csirs_pdu_rel15_t *cfg,
                                uint32_t noise,
                                uint8_t rank_indicator,
                                csi_rs_pmi_t *pmi,
                                int32_t *precoded_sinr_dB)
{
  if (rank_indicator > 1) {
    LOG_W(NR_PHY, "PMI not implemented for 2 ports and rank %d\n", rank_indicator + 1);
    return -1;
  }
  // R = sum H^H H, 2x2 Hermitian (only R[0][0], R[0][1], R[1][1] needed)
  c64_t R[2][2] = {{{0}}};
  int N_RE_total = 0;
  for (int rb = cfg->start_rb; rb < cfg->start_rb + cfg->nr_of_rbs; rb++) {
    if (cfg->freq_density <= 1 && cfg->freq_density != (rb % 2))
      continue;
    int k = rb * NR_NB_SC_PER_RB;
    for (int ant_rx = 0; ant_rx < nb_antennas_rx; ant_rx++) {
      c16_t h0 = ch_freq[ant_rx][0][k], h1 = ch_freq[ant_rx][1][k];
      R[0][0].r += (int64_t)h0.r * h0.r + (int64_t)h0.i * h0.i; // |h0|^2 (real)
      R[1][1].r += (int64_t)h1.r * h1.r + (int64_t)h1.i * h1.i; // |h1|^2 (real)
      R[0][1].r += (int64_t)h0.r * h1.r + (int64_t)h0.i * h1.i; // Re(conj(h0)*h1)
      R[0][1].i += (int64_t)h0.r * h1.i - (int64_t)h0.i * h1.r; // Im(conj(h0)*h1)
      N_RE_total++;
    }
  }
  if (N_RE_total == 0)
    return -1;

  // total power, constant for all n
  const int64_t trace = R[0][0].r + R[1][1].r;

  if (rank_indicator == 0) {
    // Rank 1 (Table 5.2.2.2.1-1): 4 hypotheses, W_n = (1/sqrt(2))*[1; phi_n]
    // W^H R W = (1/2) * (trace + 2*Re(phi_n * R[0][1]))
    int64_t best = INT64_MIN, best_signal = 0;
    for (int n = 0; n < 4; n++) {
      int64_t cross = (int64_t)phi_r[n] * R[0][1].r - (int64_t)phi_i[n] * R[0][1].i;
      int64_t m = trace + 2 * cross;
      if (m > best) {
        best = m;
        pmi->i_2 = n;
        best_signal = m;
      }
    }
    // Average signal power per RE = best / (2 * N_RE_total)  (1/2 do W^H R W)
    int64_t sp = best_signal / (2LL * N_RE_total);
    *precoded_sinr_dB = dB_fixed(sp / noise);
  } else {
    // Rank 2 (Table 5.2.2.2.1-1): 2 hypotheses.
    // The two PMI hypotheses span the same 2D subspace, giving identical trace, determinant and eigenvalues of W^H R W.
    // They differ only in how the basis is rotated:
    //   - i_2=0  uses W = (1/2)[[1, 1],[1, -1]]   -> off-diag picks up Im(R_01)
    //   - i_2=1  uses W = (1/2)[[1, 1],[j, -j]]   -> off-diag picks up Re(R_01)
    // We will choose the rotation that minimises the off-diagonal magnitude (smaller cross-layer interference for sub-MMSE receivers).
    pmi->i_2 = (llabs(R[0][1].i) < llabs(R[0][1].r)) ? 0 : 1;
    int64_t signal_per_layer = trace / (2LL * N_RE_total);
    *precoded_sinr_dB = dB_fixed(signal_per_layer / noise);
  }
  return 0;
}

// Rank 1 or 2; 4 ports (TS 38.214 - Tables 5.2.2.2.1-5 and -6)
static int nr_csi_rs_pmi_4ports(int nb_antennas_rx,
                                int N_ports,
                                int osz,
                                const c16_t ch_freq[][N_ports][osz],
                                const fapi_nr_dl_config_csirs_pdu_rel15_t *cfg,
                                uint32_t noise,
                                uint8_t rank_indicator,
                                csi_rs_pmi_t *pmi,
                                int32_t *precoded_sinr_dB)
{
  if (rank_indicator > 1) {
    LOG_W(NR_PHY, "PMI not implemented for 4 ports and rank %d\n", rank_indicator + 1);
    return -1;
  }
  // R = sum H^H H, 4x4 Hermitian, upper triangle only.
  c64_t R[4][4] = {{{0}}};
  int N_RE_total = 0;
  for (int rb = cfg->start_rb; rb < cfg->start_rb + cfg->nr_of_rbs; rb++) {
    if (cfg->freq_density <= 1 && cfg->freq_density != (rb % 2))
      continue;
    int k = rb * NR_NB_SC_PER_RB;
    for (int ant_rx = 0; ant_rx < nb_antennas_rx; ant_rx++) {
      c16_t h[4];
      for (int p = 0; p < 4; p++)
        h[p] = ch_freq[ant_rx][p][k];
      for (int i = 0; i < 4; i++)
        for (int j = i; j < 4; j++) {
          R[i][j].r += (int64_t)h[i].r * h[j].r + (int64_t)h[i].i * h[j].i;
          R[i][j].i += (int64_t)h[i].r * h[j].i - (int64_t)h[i].i * h[j].r;
        }
      N_RE_total++;
    }
  }
  if (N_RE_total == 0)
    return -1;

  // For W = c*[v_l; phi_n*v_l], W^H R W = (1/4)[A_l + B_l + 2*Re(phi_n*D_l)]
  // A_l = v_l^H R[0:2,0:2] v_l   (Hermitian -> real)
  // B_l = v_l^H R[2:4,2:4] v_l   (Hermitian -> real)
  // D_l = v_l^H R[0:2,2:4] v_l   (general -> complex)
  int64_t A[8], B[8];
  c64_t D[8];
  for (int l = 0; l < 8; l++) {
    const int64_t vr = vl_r[l], vi = vl_i[l];
    A[l] = (int64_t)PMI_Q * PMI_Q * (R[0][0].r + R[1][1].r) + 2LL * PMI_Q * (R[0][1].r * vr - R[0][1].i * vi);
    B[l] = (int64_t)PMI_Q * PMI_Q * (R[2][2].r + R[3][3].r) + 2LL * PMI_Q * (R[2][3].r * vr - R[2][3].i * vi);
    D[l].r = (int64_t)PMI_Q * PMI_Q * (R[0][2].r + R[1][3].r)
             + (int64_t)PMI_Q * ((R[0][3].r + R[1][2].r) * vr + (R[1][2].i - R[0][3].i) * vi);
    D[l].i = (int64_t)PMI_Q * PMI_Q * (R[0][2].i + R[1][3].i)
             + (int64_t)PMI_Q * ((R[0][3].i + R[1][2].i) * vr + (R[0][3].r - R[1][2].r) * vi);
  }

  int64_t best = INT64_MIN, best_signal = 0;
  if (rank_indicator == 0) {
    // Rank 1, Table 5.2.2.2.1-5: 32 entries indexed by (l, n).
    for (int l = 0; l < 8; l++) {
      const int64_t AB = A[l] + B[l];
      for (int n = 0; n < 4; n++) {
        int64_t cross = (int64_t)phi_r[n] * D[l].r - (int64_t)phi_i[n] * D[l].i;
        int64_t m = AB + 2 * cross;
        if (m > best) {
          best = m;
          pmi->i_1_1 = l;
          pmi->i_2 = n;
          best_signal = m;
        }
      }
    }
  } else {
    // Rank 2, Table 5.2.2.2.1-6: 32 entries indexed by (l, i13, n). l' = (l + 4*i13) mod 8.
    // trace(W^H R W) = (1/8)[A_l + A_l' + B_l + B_l' + 2*Re(phi_n*(D_l - D_l'))]
    for (int l = 0; l < 8; l++)
      for (int i13 = 0; i13 < 2; i13++) {
        const int lp = (l + 4 * i13) & 7;
        const int64_t AB = (A[l] + A[lp]) + (B[l] + B[lp]);
        const int64_t dDr = D[l].r - D[lp].r, dDi = D[l].i - D[lp].i;
        for (int n = 0; n < 2; n++) {
          int64_t cross = (int64_t)phi_r[n] * dDr - (int64_t)phi_i[n] * dDi;
          int64_t m = AB + 2 * cross;
          if (m > best) {
            best = m;
            pmi->i_1_1 = l;
            pmi->i_1_3 = i13;
            pmi->i_2 = n;
            best_signal = m;
          }
        }
      }
  }
  // Average signal power per RE = best / (4 * Q^2 * N_RE_total)  [rank 1]
  // For rank 2 the (1/8) cancels in best, but we keep the (1/4) here for SINR consistency.
  int64_t sp = best_signal / (4LL * PMI_Q * PMI_Q * N_RE_total);
  *precoded_sinr_dB = dB_fixed(sp / noise);
  return 0;
}

// Rank 1; 8 ports (TS 38.214 - Table 5.2.2.2.1-9)
static int nr_csi_rs_pmi_8ports(int nb_antennas_rx,
                                int N_ports,
                                int osz,
                                const c16_t ch_freq[][N_ports][osz],
                                const fapi_nr_dl_config_csirs_pdu_rel15_t *cfg,
                                uint32_t noise,
                                uint8_t rank_indicator,
                                csi_rs_pmi_t *pmi,
                                int32_t *precoded_sinr_dB)
{
  if (rank_indicator != 0) {
    LOG_W(NR_PHY, "PMI not implemented for 8 ports and rank %d\n", rank_indicator + 1);
    return -1;
  }
  // R = sum H^H H, 8x8 Hermitian, upper triangle only.
  c64_t R[8][8] = {{{0}}};
  int N_RE_total = 0;
  for (int rb = cfg->start_rb; rb < cfg->start_rb + cfg->nr_of_rbs; rb++) {
    if (cfg->freq_density <= 1 && cfg->freq_density != (rb % 2))
      continue;
    int k = rb * NR_NB_SC_PER_RB;
    for (int ant_rx = 0; ant_rx < nb_antennas_rx; ant_rx++) {
      c16_t h[8];
      for (int p = 0; p < 8; p++)
        h[p] = ch_freq[ant_rx][p][k];
      for (int i = 0; i < 8; i++)
        for (int j = i; j < 8; j++) {
          R[i][j].r += (int64_t)h[i].r * h[j].r + (int64_t)h[i].i * h[j].i;
          R[i][j].i += (int64_t)h[i].r * h[j].i - (int64_t)h[i].i * h[j].r;
        }
      N_RE_total++;
    }
  }
  if (N_RE_total == 0)
    return -1;

  // For W = c*[v_lm; phi_n*v_lm] with v_lm = v_l1 (x) u_m2 (4x1, |v_lm[i]|=Q), decompose R into 4x4 blocks
  // R_TL = R[0:4,0:4], R_TR = R[0:4,4:8], R_BR = R[4:8,4:8],
  // and precompute A_lm, B_lm, D_lm for each (l1, m2).
  // The Kronecker structure means v_lm = [1, exp(j*pi*m2/4), exp(j*pi*l1/4), exp(j*pi*(l1+m2)/4)],
  // so the fourth entry is just a lookup at index (l1+m2)%8.
  int64_t A[8][8], B[8][8];
  c64_t D[8][8];
  for (int l1 = 0; l1 < 8; l1++) {
    for (int m2 = 0; m2 < 8; m2++) {
      const int vr[4] = {PMI_Q, vl_r[m2], vl_r[l1], vl_r[(l1 + m2) & 7]};
      const int vi[4] = {0, vl_i[m2], vl_i[l1], vl_i[(l1 + m2) & 7]};
      int64_t Aval = (int64_t)PMI_Q * PMI_Q * (R[0][0].r + R[1][1].r + R[2][2].r + R[3][3].r);
      int64_t Bval = (int64_t)PMI_Q * PMI_Q * (R[4][4].r + R[5][5].r + R[6][6].r + R[7][7].r);
      int64_t Dr = (int64_t)PMI_Q * PMI_Q * (R[0][4].r + R[1][5].r + R[2][6].r + R[3][7].r);
      int64_t Di = (int64_t)PMI_Q * PMI_Q * (R[0][4].i + R[1][5].i + R[2][6].i + R[3][7].i);
      for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) {
          // c = conj(v[i]) * v[j]
          int64_t cr = (int64_t)vr[i] * vr[j] + (int64_t)vi[i] * vi[j];
          int64_t ci = (int64_t)vr[i] * vi[j] - (int64_t)vi[i] * vr[j];
          // Hermitian blocks: 2 * Re(c * M[i][j])
          Aval += 2 * (cr * R[i][j].r - ci * R[i][j].i);
          Bval += 2 * (cr * R[i + 4][j + 4].r - ci * R[i + 4][j + 4].i);
          // General block R_TR: c * R[i][j+4] + conj(c) * R[j][i+4]
          Dr += cr * (R[i][j + 4].r + R[j][i + 4].r) + ci * (R[j][i + 4].i - R[i][j + 4].i);
          Di += cr * (R[i][j + 4].i + R[j][i + 4].i) + ci * (R[i][j + 4].r - R[j][i + 4].r);
        }
      A[l1][m2] = Aval;
      B[l1][m2] = Bval;
      D[l1][m2].r = Dr;
      D[l1][m2].i = Di;
    }
  }

  // Enumerate (l1, m2, n) - 256 hypotheses, each evaluation is O(1).
  int64_t best = INT64_MIN, best_signal = 0;
  for (int l1 = 0; l1 < 8; l1++)
    for (int m2 = 0; m2 < 8; m2++) {
      const int64_t AB = A[l1][m2] + B[l1][m2];
      for (int n = 0; n < 4; n++) {
        int64_t cross = (int64_t)phi_r[n] * D[l1][m2].r - (int64_t)phi_i[n] * D[l1][m2].i;
        int64_t m = AB + 2 * cross;
        if (m > best) {
          best = m;
          pmi->i_1_1 = l1;
          pmi->i_1_2 = m2;
          pmi->i_2 = n;
          best_signal = m;
        }
      }
    }
  // Average signal power per RE = best / (8 * Q^2 * N_RE_total)
  int64_t sp = best_signal / (8LL * PMI_Q * PMI_Q * (int64_t)N_RE_total);
  *precoded_sinr_dB = dB_fixed(sp / noise);
  return 0;
}

static int nr_csi_rs_pmi_estimation(const PHY_VARS_NR_UE *ue,
                                    const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                                    const uint8_t N_ports,
                                    const c16_t csi_rs_estimated_channel_freq[][N_ports][ue->frame_parms.ofdm_symbol_size],
                                    const uint32_t interference_plus_noise_power,
                                    const uint8_t rank_indicator,
                                    csi_rs_pmi_t *pmi,
                                    int32_t *precoded_sinr_dB)
{
  memset(pmi, 0, sizeof(*pmi));
  const NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  const int nrx = fp->nb_antennas_rx;
  const int osz = fp->ofdm_symbol_size;
  const uint32_t noise = (interference_plus_noise_power == 0) ? 1 : interference_plus_noise_power;

  switch (N_ports) {
    case 1:
      return nr_csi_rs_pmi_1port(nrx, N_ports, osz, csi_rs_estimated_channel_freq, csirs_config_pdu, noise, precoded_sinr_dB);
    case 2:
      return nr_csi_rs_pmi_2ports(nrx,
                                  N_ports,
                                  osz,
                                  csi_rs_estimated_channel_freq,
                                  csirs_config_pdu,
                                  noise,
                                  rank_indicator,
                                  pmi,
                                  precoded_sinr_dB);
    case 4:
      return nr_csi_rs_pmi_4ports(nrx,
                                  N_ports,
                                  osz,
                                  csi_rs_estimated_channel_freq,
                                  csirs_config_pdu,
                                  noise,
                                  rank_indicator,
                                  pmi,
                                  precoded_sinr_dB);
    case 8:
      return nr_csi_rs_pmi_8ports(nrx,
                                  N_ports,
                                  osz,
                                  csi_rs_estimated_channel_freq,
                                  csirs_config_pdu,
                                  noise,
                                  rank_indicator,
                                  pmi,
                                  precoded_sinr_dB);
    default:
      LOG_W(NR_PHY, "PMI not implemented for %d antenna ports\n", N_ports);
      return -1;
  }
}

int nr_csi_rs_cqi_estimation(const uint32_t precoded_sinr,
                             uint8_t *cqi) {

  *cqi = 0;

  // Default SINR table for an AWGN channel for SISO scenario, considering 0.1 BLER condition and TS 38.214 Table 5.2.2.1-2
  if(precoded_sinr>0 && precoded_sinr<=2) {
    *cqi = 4;
  } else if(precoded_sinr==3) {
    *cqi = 5;
  } else if(precoded_sinr>3 && precoded_sinr<=5) {
    *cqi = 6;
  } else if(precoded_sinr>5 && precoded_sinr<=7) {
    *cqi = 7;
  } else if(precoded_sinr>7 && precoded_sinr<=9) {
    *cqi = 8;
  } else if(precoded_sinr==10) {
    *cqi = 9;
  } else if(precoded_sinr>10 && precoded_sinr<=12) {
    *cqi = 10;
  } else if(precoded_sinr>12 && precoded_sinr<=15) {
    *cqi = 11;
  } else if(precoded_sinr==16) {
    *cqi = 12;
  } else if(precoded_sinr>16 && precoded_sinr<=18) {
    *cqi = 13;
  } else if(precoded_sinr==19) {
    *cqi = 14;
  } else if(precoded_sinr>19) {
    *cqi = 15;
  }

  return 0;
}

static void nr_csi_im_power_estimation(const PHY_VARS_NR_UE *ue,
                                       const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu,
                                       uint32_t *interference_plus_noise_power,
                                       const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP])
{
  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  const uint16_t end_rb = csiim_config_pdu->start_rb + csiim_config_pdu->nr_of_rbs > csiim_config_pdu->bwp_size ?
                          csiim_config_pdu->bwp_size : csiim_config_pdu->start_rb + csiim_config_pdu->nr_of_rbs;

  int32_t count = 0;
  int32_t sum_re = 0;
  int32_t sum_im = 0;
  int32_t sum2_re = 0;
  int32_t sum2_im = 0;

  int l_csiim[4] = {-1, -1, -1, -1};

  for(int symb_idx = 0; symb_idx < 4; symb_idx++) {

    uint8_t symb = csiim_config_pdu->l_csiim[symb_idx];
    bool done = false;
    for (int symb_idx2 = 0; symb_idx2 < symb_idx; symb_idx2++) {
      if (l_csiim[symb_idx2] == symb) {
        done = true;
      }
    }

    if (done) {
      continue;
    }

    l_csiim[symb_idx] = symb;
    uint64_t symbol_offset = symb*frame_parms->ofdm_symbol_size;

    for (int ant_rx = 0; ant_rx < frame_parms->nb_antennas_rx; ant_rx++) {

      const c16_t *rx_signal = &rxdataF[ant_rx][symbol_offset];

      for (int rb = csiim_config_pdu->start_rb; rb < end_rb; rb++) {

        uint16_t sc0_offset = (frame_parms->first_carrier_offset + rb*NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;

        for (int sc_idx = 0; sc_idx < 4; sc_idx++) {

          uint16_t sc = sc0_offset + csiim_config_pdu->k_csiim[sc_idx];
          if (sc >= frame_parms->ofdm_symbol_size) {
            sc -= frame_parms->ofdm_symbol_size;
          }

#ifdef NR_CSIIM_DEBUG
          LOG_I(NR_PHY, "(ant_rx %i, sc %i) real %i, imag %i\n", ant_rx, sc, rx_signal[sc].r, rx_signal[sc].i);
#endif

          if (sc == 0) // skip DC for noise power estimation
            continue;
          sum_re += rx_signal[sc].r;
          sum_im += rx_signal[sc].i;
          sum2_re += rx_signal[sc].r * rx_signal[sc].r;
          sum2_im += rx_signal[sc].i * rx_signal[sc].i;
          count++;
        }
      }
    }
  }

  int32_t power_re = sum2_re / count - (sum_re / count) * (sum_re / count);
  int32_t power_im = sum2_im / count - (sum_im / count) * (sum_im / count);

  *interference_plus_noise_power = power_re + power_im;

#ifdef NR_CSIIM_DEBUG
  LOG_I(NR_PHY, "interference_plus_noise_power based on CSI-IM = %i\n", *interference_plus_noise_power);
#endif
}

void nr_ue_csi_im_procedures(PHY_VARS_NR_UE *ue,
                             const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP],
                             const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu)
{

#ifdef NR_CSIIM_DEBUG
  LOG_I(NR_PHY, "csiim_config_pdu->bwp_size = %i\n", csiim_config_pdu->bwp_size);
  LOG_I(NR_PHY, "csiim_config_pdu->bwp_start = %i\n", csiim_config_pdu->bwp_start);
  LOG_I(NR_PHY, "csiim_config_pdu->subcarrier_spacing = %i\n", csiim_config_pdu->subcarrier_spacing);
  LOG_I(NR_PHY, "csiim_config_pdu->start_rb = %i\n", csiim_config_pdu->start_rb);
  LOG_I(NR_PHY, "csiim_config_pdu->nr_of_rbs = %i\n", csiim_config_pdu->nr_of_rbs);
  LOG_I(NR_PHY, "csiim_config_pdu->k_csiim = %i.%i.%i.%i\n", csiim_config_pdu->k_csiim[0], csiim_config_pdu->k_csiim[1], csiim_config_pdu->k_csiim[2], csiim_config_pdu->k_csiim[3]);
  LOG_I(NR_PHY, "csiim_config_pdu->l_csiim = %i.%i.%i.%i\n", csiim_config_pdu->l_csiim[0], csiim_config_pdu->l_csiim[1], csiim_config_pdu->l_csiim[2], csiim_config_pdu->l_csiim[3]);
#endif

  nr_csi_im_power_estimation(ue, csiim_config_pdu, &ue->nr_csi_info->interference_plus_noise_power, rxdataF);
  ue->nr_csi_info->csi_im_meas_computed = true;
}

void nr_ue_csi_rs_procedures(PHY_VARS_NR_UE *ue,
                             const UE_nr_rxtx_proc_t *proc,
                             const c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP],
                             fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu)
{

#ifdef NR_CSIRS_DEBUG
  LOG_I(NR_PHY, "csirs_config_pdu->subcarrier_spacing = %i\n", csirs_config_pdu->subcarrier_spacing);
  LOG_I(NR_PHY, "csirs_config_pdu->cyclic_prefix = %i\n", csirs_config_pdu->cyclic_prefix);
  LOG_I(NR_PHY, "csirs_config_pdu->start_rb = %i\n", csirs_config_pdu->start_rb);
  LOG_I(NR_PHY, "csirs_config_pdu->nr_of_rbs = %i\n", csirs_config_pdu->nr_of_rbs);
  LOG_I(NR_PHY, "csirs_config_pdu->csi_type = %i (0:TRS, 1:CSI-RS NZP, 2:CSI-RS ZP)\n", csirs_config_pdu->csi_type);
  LOG_I(NR_PHY, "csirs_config_pdu->row = %i\n", csirs_config_pdu->row);
  LOG_I(NR_PHY, "csirs_config_pdu->freq_domain = %i\n", csirs_config_pdu->freq_domain);
  LOG_I(NR_PHY, "csirs_config_pdu->symb_l0 = %i\n", csirs_config_pdu->symb_l0);
  LOG_I(NR_PHY, "csirs_config_pdu->symb_l1 = %i\n", csirs_config_pdu->symb_l1);
  LOG_I(NR_PHY, "csirs_config_pdu->cdm_type = %i\n", csirs_config_pdu->cdm_type);
  LOG_I(NR_PHY, "csirs_config_pdu->freq_density = %i (0: dot5 (even RB), 1: dot5 (odd RB), 2: one, 3: three)\n", csirs_config_pdu->freq_density);
  LOG_I(NR_PHY, "csirs_config_pdu->scramb_id = %i\n", csirs_config_pdu->scramb_id);
  LOG_I(NR_PHY, "csirs_config_pdu->power_control_offset = %i\n", csirs_config_pdu->power_control_offset);
  LOG_I(NR_PHY, "csirs_config_pdu->power_control_offset_ss = %i\n", csirs_config_pdu->power_control_offset_ss);
#endif

  if(csirs_config_pdu->csi_type == 0) {
    LOG_E(NR_PHY, "Handling of CSI-RS for tracking not handled yet at PHY\n");
    return;
  }

  if(csirs_config_pdu->csi_type == 2) {
    LOG_E(NR_PHY, "Handling of ZP CSI-RS not handled yet at PHY\n");
    return;
  }

  const NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  csi_mapping_parms_t mapping_parms = get_csi_mapping_parms(csirs_config_pdu->row,
                                                            csirs_config_pdu->freq_domain,
                                                            csirs_config_pdu->symb_l0,
                                                            csirs_config_pdu->symb_l1);
  nr_csi_info_t *csi_info = ue->nr_csi_info;
  nr_generate_csi_rs(frame_parms,
                     &mapping_parms,
                     AMP,
                     proc->nr_slot_rx,
                     csirs_config_pdu->freq_density,
                     csirs_config_pdu->start_rb,
                     csirs_config_pdu->nr_of_rbs,
                     csirs_config_pdu->symb_l0,
                     csirs_config_pdu->symb_l1,
                     csirs_config_pdu->row,
                     csirs_config_pdu->scramb_id,
                     csirs_config_pdu->power_control_offset_ss,
                     csirs_config_pdu->cdm_type,
                     csi_info->csi_rs_generated_signal);

  csi_info->csi_rs_generated_signal_bits = log2_approx(AMP);

  /* OFDM symbol size \times sizeof(c16_t) is always multiple of 64. Since the
  estimates have first RB at start of buffer we don't need padding for 32 or 64
  byte alignment. */
  c16_t csi_rs_ls_estimated_channel[frame_parms->nb_antennas_rx][mapping_parms.ports][frame_parms->ofdm_symbol_size];
  c16_t csi_rs_estimated_channel_freq[frame_parms->nb_antennas_rx][mapping_parms.ports][frame_parms->ofdm_symbol_size];

  int CDM_group_size = get_cdm_group_size(csirs_config_pdu->cdm_type);
  c16_t csi_rs_received_signal[frame_parms->nb_antennas_rx][frame_parms->samples_per_slot_wCP];
  uint32_t rsrp = 0;
  int rsrp_dBm = 0;
  nr_get_csi_rs_signal(ue,
                       proc,
                       csirs_config_pdu,
                       csi_info,
                       &mapping_parms,
                       CDM_group_size,
                       csi_rs_received_signal,
                       &rsrp,
                       &rsrp_dBm,
                       rxdataF);


  if (csirs_config_pdu->measurement_bitmap == 0) {
    LOG_D(NR_PHY, "No CSI-RS measurements configured\n");
    return;
  }

  uint32_t noise_power = 0;
  int16_t log2_re = 0;
  int16_t log2_maxh = 0;
  // if we need to measure only RSRP no need to do channel estimation
  if (csirs_config_pdu->measurement_bitmap > 1)
    nr_csi_rs_channel_estimation(frame_parms,
                                 csirs_config_pdu,
                                 csi_info,
                                 (const c16_t **)csi_info->csi_rs_generated_signal,
                                 csi_rs_received_signal,
                                 &mapping_parms,
                                 CDM_group_size,
                                 csi_rs_ls_estimated_channel,
                                 csi_rs_estimated_channel_freq,
                                 &log2_re,
                                 &log2_maxh,
                                 &noise_power);

  // bit 1 in bitmap to indicate RI measurement
  uint8_t rank_indicator = 0;
  if (csirs_config_pdu->measurement_bitmap & 2) {
    rank_indicator = nr_csi_rs_ri_estimation(ue, csirs_config_pdu, mapping_parms.ports, csi_rs_estimated_channel_freq, log2_maxh);
  }

  csi_rs_pmi_t pmi = {0};
  uint8_t cqi = 0;
  int32_t precoded_sinr_dB = 0;
  // bit 3 in bitmap to indicate RI measurment
  if (csirs_config_pdu->measurement_bitmap & 8) {
    nr_csi_rs_pmi_estimation(ue,
                             csirs_config_pdu,
                             mapping_parms.ports,
                             csi_rs_estimated_channel_freq,
                             csi_info->csi_im_meas_computed ? csi_info->interference_plus_noise_power : noise_power,
                             rank_indicator,
                             &pmi,
                             &precoded_sinr_dB);

    // bit 4 in bitmap to indicate RI measurment
    if(csirs_config_pdu->measurement_bitmap & 16)
      nr_csi_rs_cqi_estimation(precoded_sinr_dB, &cqi);
  }

  switch (csirs_config_pdu->measurement_bitmap) {
    case 1 :
      LOG_I(NR_PHY, "[UE %d] RSRP = %i dBm\n", ue->Mod_id, rsrp_dBm);
      break;
    case 26 :
      LOG_I(NR_PHY, "RI = %i i1 = %i.%i.%i, i2 = %i, SINR = %i dB, CQI = %i\n",
            rank_indicator + 1, pmi.i_1_1, pmi.i_1_2, pmi.i_1_3, pmi.i_2, precoded_sinr_dB, cqi);
      break;
    case 27 :
      LOG_I(NR_PHY, "RSRP = %i dBm, RI = %i i1 = %i.%i.%i, i2 = %i, SINR = %i dB, CQI = %i\n",
            rsrp_dBm, rank_indicator + 1, pmi.i_1_1, pmi.i_1_2, pmi.i_1_3, pmi.i_2, precoded_sinr_dB, cqi);
      break;
    default :
      AssertFatal(false, "Not supported measurement configuration\n");
  }

  // Send CSI measurements to MAC
  if (!ue->if_inst || !ue->if_inst->dl_indication)
    return;

  fapi_nr_l1_measurements_t l1_measurements = {
    .gNB_index = proc->gNB_id,
    .meas_type = NFAPI_NR_CSI_MEAS,
    .Nid_cell = frame_parms->Nid_cell,
    .is_neighboring_cell = false,
    .rsrp_dBm = rsrp_dBm,
    .rank_indicator = rank_indicator,
    .i_1_1 = pmi.i_1_1,
    .i_1_2 = pmi.i_1_2,
    .i_1_3 = pmi.i_1_3,
    .i_2 = pmi.i_2,
    .cqi = cqi,
    .radiolink_monitoring = RLM_no_monitoring, // TODO do be activated in case of RLM based on CSI-RS
  };
  nr_downlink_indication_t dl_indication;
  fapi_nr_rx_indication_t rx_ind = {0};
  nr_fill_dl_indication(&dl_indication, NULL, &rx_ind, proc, ue, NULL);
  nr_fill_rx_indication(&rx_ind, FAPI_NR_MEAS_IND, ue, 0, 0, NULL, 1, proc, (void *)&l1_measurements, NULL);
  ue->if_inst->dl_indication(&dl_indication);
}

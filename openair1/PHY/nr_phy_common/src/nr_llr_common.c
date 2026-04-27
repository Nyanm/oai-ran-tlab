/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_llr_common.h"
#include "nr_phy_common.h"
#include "PHY/sse_intrin.h"
#include "PHY/TOOLS/tools_defs.h"
#include "common/utils/assertions.h"
#ifdef __aarch64__
#define USE_128BIT
#endif

void nr_compute_llr(const c16_t *rxdataF_comp,
                    const c16_t *ch_mag,
                    const c16_t *ch_magb,
                    const c16_t *ch_magc,
                    int16_t *llr,
                    uint32_t nb_re,
                    uint8_t mod_order)
{
  switch(mod_order) {
    case 2:
      nr_qpsk_llr(rxdataF_comp, llr, nb_re);
      break;
    case 4:
      nr_16qam_llr(rxdataF_comp, ch_mag, llr, nb_re);
      break;
    case 6:
    nr_64qam_llr(rxdataF_comp, ch_mag, ch_magb, llr, nb_re);
      break;
    case 8:
    nr_256qam_llr(rxdataF_comp, ch_mag, ch_magb, ch_magc, llr, nb_re);
      break;
    default:
      AssertFatal(false, "nr_compute_llr: invalid Qm value, Qm = %d\n", mod_order);
      break;
  }
}

/*
 * This function computes the LLRs of stream 0 (s_0) in presence of the interfering stream 1 (s_1) assuming that both symbols are
 * QPSK. It can be used for both MU-MIMO interference-aware receiver or for SU-MIMO receivers.
 *
 * Input:
 *   stream0_in:  MF filter output for 1st stream, i.e., y0' = h0'*y0
 *   stream1_in:  MF filter output for 2nd stream, i.e., y1' = h1'*y0
 *   rho01:       Channel cross correlation, i.e., rho01 = h0'*h1
 *   length:      Number of resource elements
 *
 * Output:
 *   stream0_out: Output LLRs for 1st stream
 */
#ifdef USE_128BIT
static inline void nr_qpsk_llr_2layers_core_epi16(simde__m128i desired_r,
                                                  simde__m128i desired_i,
                                                  simde__m128i interferer_r,
                                                  simde__m128i interferer_i,
                                                  simde__m128i rhor,
                                                  simde__m128i rhoi,
                                                  simde__m128i one_over_2_sqrt_2,
                                                  simde__m128i *llr_r,
                                                  simde__m128i *llr_i)
{
  simde__m128i rho_p = simde_mm_adds_epi16(rhor, rhoi);
  rho_p = simde_mm_mulhi_epi16(rho_p, one_over_2_sqrt_2);
  simde__m128i rho_m = simde_mm_subs_epi16(rhor, rhoi);
  rho_m = simde_mm_mulhi_epi16(rho_m, one_over_2_sqrt_2);

  simde__m128i abs_psi_rpm = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_p, interferer_r));
  simde__m128i abs_psi_imm = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_m, interferer_i));
  simde__m128i abs_psi_rmm = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_m, interferer_r));
  simde__m128i abs_psi_ipm = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_p, interferer_i));
  simde__m128i abs_psi_rpp = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_p, interferer_r));
  simde__m128i abs_psi_imp = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_m, interferer_i));
  simde__m128i abs_psi_rmp = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_m, interferer_r));
  simde__m128i abs_psi_ipp = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_p, interferer_i));

  simde__m128i bit_met_num_re_p = simde_mm_adds_epi16(simde_mm_adds_epi16(abs_psi_rpm, abs_psi_imm), desired_r);
  bit_met_num_re_p = simde_mm_adds_epi16(bit_met_num_re_p, desired_i);
  simde__m128i bit_met_num_re_m = simde_mm_adds_epi16(simde_mm_adds_epi16(abs_psi_rmm, abs_psi_ipp), desired_r);
  bit_met_num_re_m = simde_mm_subs_epi16(bit_met_num_re_m, desired_i);
  simde__m128i bit_met_den_re_p = simde_mm_adds_epi16(simde_mm_subs_epi16(simde_mm_adds_epi16(abs_psi_rmp, abs_psi_ipm), desired_r), desired_i);
  simde__m128i bit_met_den_re_m = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_adds_epi16(abs_psi_rpp, abs_psi_imp), desired_r), desired_i);

  simde__m128i bit_met_num_im_p = simde_mm_adds_epi16(simde_mm_adds_epi16(abs_psi_rpm, abs_psi_imm), desired_r);
  bit_met_num_im_p = simde_mm_adds_epi16(bit_met_num_im_p, desired_i);
  simde__m128i bit_met_num_im_m = simde_mm_adds_epi16(simde_mm_subs_epi16(simde_mm_adds_epi16(abs_psi_rmp, abs_psi_ipm), desired_r), desired_i);
  simde__m128i bit_met_den_im_p = simde_mm_adds_epi16(simde_mm_adds_epi16(abs_psi_rmm, abs_psi_ipp), desired_r);
  bit_met_den_im_p = simde_mm_subs_epi16(bit_met_den_im_p, desired_i);
  simde__m128i bit_met_den_im_m = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_adds_epi16(abs_psi_rpp, abs_psi_imp), desired_r), desired_i);

  simde__m128i logmax_num_re = simde_mm_max_epi16(bit_met_num_re_p, bit_met_num_re_m);
  simde__m128i logmax_den_re = simde_mm_max_epi16(bit_met_den_re_p, bit_met_den_re_m);
  simde__m128i logmax_num_im = simde_mm_max_epi16(bit_met_num_im_p, bit_met_num_im_m);
  simde__m128i logmax_den_im = simde_mm_max_epi16(bit_met_den_im_p, bit_met_den_im_m);

  *llr_r = simde_mm_subs_epi16(logmax_num_re, logmax_den_re);
  *llr_i = simde_mm_subs_epi16(logmax_num_im, logmax_den_im);
}

static inline void nr_qpsk_llr_2layers_store_epi16(simde__m128i *stream_out, int idx, uint32_t length, simde__m128i llr_r, simde__m128i llr_i)
{
  simde_mm_storeu_si128(&stream_out[idx], simde_mm_unpacklo_epi16(llr_r, llr_i));
  if (idx < ((length >> 1) - 1))
    simde_mm_storeu_si128(&stream_out[idx + 1], simde_mm_unpackhi_epi16(llr_r, llr_i));
}
#else
static inline void nr_qpsk_llr_2layers_core_epi16(simde__m256i desired_r,
                                                  simde__m256i desired_i,
                                                  simde__m256i interferer_r,
                                                  simde__m256i interferer_i,
                                                  simde__m256i rhor,
                                                  simde__m256i rhoi,
                                                  simde__m256i one_over_2_sqrt_2,
                                                  simde__m256i *llr_r,
                                                  simde__m256i *llr_i)
{
  simde__m256i rho_p = simde_mm256_adds_epi16(rhor, rhoi);
  rho_p = simde_mm256_mulhi_epi16(rho_p, one_over_2_sqrt_2);
  simde__m256i rho_m = simde_mm256_subs_epi16(rhor, rhoi);
  rho_m = simde_mm256_mulhi_epi16(rho_m, one_over_2_sqrt_2);

  simde__m256i abs_psi_rpm = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_p, interferer_r));
  simde__m256i abs_psi_imm = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_m, interferer_i));
  simde__m256i abs_psi_rmm = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_m, interferer_r));
  simde__m256i abs_psi_ipm = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_p, interferer_i));
  simde__m256i abs_psi_rpp = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_p, interferer_r));
  simde__m256i abs_psi_imp = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_m, interferer_i));
  simde__m256i abs_psi_rmp = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_m, interferer_r));
  simde__m256i abs_psi_ipp = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_p, interferer_i));

  simde__m256i bit_met_num_re_p = simde_mm256_adds_epi16(simde_mm256_adds_epi16(abs_psi_rpm, abs_psi_imm), desired_r);
  bit_met_num_re_p = simde_mm256_adds_epi16(bit_met_num_re_p, desired_i);
  simde__m256i bit_met_num_re_m = simde_mm256_adds_epi16(simde_mm256_adds_epi16(abs_psi_rmm, abs_psi_ipp), desired_r);
  bit_met_num_re_m = simde_mm256_subs_epi16(bit_met_num_re_m, desired_i);
  simde__m256i bit_met_den_re_p = simde_mm256_adds_epi16(simde_mm256_subs_epi16(simde_mm256_adds_epi16(abs_psi_rmp, abs_psi_ipm), desired_r), desired_i);
  simde__m256i bit_met_den_re_m = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_adds_epi16(abs_psi_rpp, abs_psi_imp), desired_r), desired_i);

  simde__m256i bit_met_num_im_p = simde_mm256_adds_epi16(simde_mm256_adds_epi16(abs_psi_rpm, abs_psi_imm), desired_r);
  bit_met_num_im_p = simde_mm256_adds_epi16(bit_met_num_im_p, desired_i);
  simde__m256i bit_met_num_im_m = simde_mm256_adds_epi16(simde_mm256_subs_epi16(simde_mm256_adds_epi16(abs_psi_rmp, abs_psi_ipm), desired_r), desired_i);
  simde__m256i bit_met_den_im_p = simde_mm256_adds_epi16(simde_mm256_adds_epi16(abs_psi_rmm, abs_psi_ipp), desired_r);
  bit_met_den_im_p = simde_mm256_subs_epi16(bit_met_den_im_p, desired_i);
  simde__m256i bit_met_den_im_m = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_adds_epi16(abs_psi_rpp, abs_psi_imp), desired_r), desired_i);

  simde__m256i logmax_num_re = simde_mm256_max_epi16(bit_met_num_re_p, bit_met_num_re_m);
  simde__m256i logmax_den_re = simde_mm256_max_epi16(bit_met_den_re_p, bit_met_den_re_m);
  simde__m256i logmax_num_im = simde_mm256_max_epi16(bit_met_num_im_p, bit_met_num_im_m);
  simde__m256i logmax_den_im = simde_mm256_max_epi16(bit_met_den_im_p, bit_met_den_im_m);

  *llr_r = simde_mm256_subs_epi16(logmax_num_re, logmax_den_re);
  *llr_i = simde_mm256_subs_epi16(logmax_num_im, logmax_den_im);
}

static inline void nr_qpsk_llr_2layers_store_epi16(simde__m256i *stream_out, int idx, uint32_t length, simde__m256i llr_r, simde__m256i llr_i)
{
  simde__m128i *stream0_128i_out = (simde__m128i *)&stream_out[idx];
  simde__m128i *llr_r_128 = (simde__m128i *)&llr_r;
  simde__m128i *llr_i_128 = (simde__m128i *)&llr_i;
  simde_mm_storeu_si128(&stream0_128i_out[0], simde_mm_unpacklo_epi16(llr_r_128[0], llr_i_128[0]));
  simde_mm_storeu_si128(&stream0_128i_out[1], simde_mm_unpackhi_epi16(llr_r_128[0], llr_i_128[0]));
  if (idx < ((length >> 2) - 1)) {
    simde__m128i *stream1_128i_out = (simde__m128i *)&stream_out[idx + 1];
    simde_mm_storeu_si128(&stream1_128i_out[0], simde_mm_unpacklo_epi16(llr_r_128[1], llr_i_128[1]));
    simde_mm_storeu_si128(&stream1_128i_out[1], simde_mm_unpackhi_epi16(llr_r_128[1], llr_i_128[1]));
  }
}
#endif

static void nr_qpsk_llr_2layers_dual(c16_t *stream0_in,
                                     c16_t *stream1_in,
                                     int16_t *stream0_out,
                                     int16_t *stream1_out,
                                     c16_t *rho01,
                                     c16_t *rho10,
                                     uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho01_128i = (simde__m128i *)rho01;
  simde__m128i *rho10_128i = (simde__m128i *)rho10;
  simde__m128i *stream0_128i_in = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128i_in = (simde__m128i *)stream1_in;
  simde__m128i *stream0_128i_out = (simde__m128i *)stream0_out;
  simde__m128i *stream1_128i_out = (simde__m128i *)stream1_out;
  simde__m128i one_over_2_sqrt_2 = simde_mm_set1_epi16(23170);

  for (int i = 0; i < length >> 2; i += 2) {
    simde__m128i y0r, y0i, y1r, y1i;
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128i_in[i], stream0_128i_in[i + 1]);
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128i_in[i], stream1_128i_in[i + 1]);

    simde__m128i y0r_sqrt2 = simde_mm_mulhrs_epi16(y0r, one_over_2_sqrt_2);
    simde__m128i y0i_sqrt2 = simde_mm_mulhrs_epi16(y0i, one_over_2_sqrt_2);
    simde__m128i y1r_sqrt2 = simde_mm_mulhrs_epi16(y1r, one_over_2_sqrt_2);
    simde__m128i y1i_sqrt2 = simde_mm_mulhrs_epi16(y1i, one_over_2_sqrt_2);
    simde__m128i y0r_half = simde_mm_srai_epi16(y0r, 1);
    simde__m128i y0i_half = simde_mm_srai_epi16(y0i, 1);
    simde__m128i y1r_half = simde_mm_srai_epi16(y1r, 1);
    simde__m128i y1i_half = simde_mm_srai_epi16(y1i, 1);

    simde__m128i rho01r, rho01i, rho10r, rho10i;
    oai_mm_separate_real_imag_parts(&rho01r, &rho01i, rho01_128i[i], rho01_128i[i + 1]);
    oai_mm_separate_real_imag_parts(&rho10r, &rho10i, rho10_128i[i], rho10_128i[i + 1]);

    simde__m128i llr0r, llr0i, llr1r, llr1i;
    nr_qpsk_llr_2layers_core_epi16(y0r_sqrt2, y0i_sqrt2, y1r_half, y1i_half, rho01r, rho01i, one_over_2_sqrt_2, &llr0r, &llr0i);
    nr_qpsk_llr_2layers_core_epi16(y1r_sqrt2, y1i_sqrt2, y0r_half, y0i_half, rho10r, rho10i, one_over_2_sqrt_2, &llr1r, &llr1i);

    nr_qpsk_llr_2layers_store_epi16(stream0_128i_out, i, length, llr0r, llr0i);
    nr_qpsk_llr_2layers_store_epi16(stream1_128i_out, i, length, llr1r, llr1i);
  }
#else
  simde__m256i *rho01_256i = (simde__m256i *)rho01;
  simde__m256i *rho10_256i = (simde__m256i *)rho10;
  simde__m256i *stream0_256i_in = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256i_in = (simde__m256i *)stream1_in;
  simde__m256i *stream0_256i_out = (simde__m256i *)stream0_out;
  simde__m256i *stream1_256i_out = (simde__m256i *)stream1_out;
  simde__m256i one_over_2_sqrt_2 = simde_mm256_set1_epi16(23170);

  for (int i = 0; i < length >> 3; i += 2) {
    simde__m256i y0r, y0i, y1r, y1i;
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256i_in[i], stream0_256i_in[i + 1]);
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256i_in[i], stream1_256i_in[i + 1]);

    simde__m256i y0r_sqrt2 = simde_mm256_mulhrs_epi16(y0r, one_over_2_sqrt_2);
    simde__m256i y0i_sqrt2 = simde_mm256_mulhrs_epi16(y0i, one_over_2_sqrt_2);
    simde__m256i y1r_sqrt2 = simde_mm256_mulhrs_epi16(y1r, one_over_2_sqrt_2);
    simde__m256i y1i_sqrt2 = simde_mm256_mulhrs_epi16(y1i, one_over_2_sqrt_2);
    simde__m256i y0r_half = simde_mm256_srai_epi16(y0r, 1);
    simde__m256i y0i_half = simde_mm256_srai_epi16(y0i, 1);
    simde__m256i y1r_half = simde_mm256_srai_epi16(y1r, 1);
    simde__m256i y1i_half = simde_mm256_srai_epi16(y1i, 1);

    simde__m256i rho01r, rho01i, rho10r, rho10i;
    oai_mm256_separate_real_imag_parts(&rho01r, &rho01i, rho01_256i[i], rho01_256i[i + 1]);
    oai_mm256_separate_real_imag_parts(&rho10r, &rho10i, rho10_256i[i], rho10_256i[i + 1]);

    simde__m256i llr0r, llr0i, llr1r, llr1i;
    nr_qpsk_llr_2layers_core_epi16(y0r_sqrt2, y0i_sqrt2, y1r_half, y1i_half, rho01r, rho01i, one_over_2_sqrt_2, &llr0r, &llr0i);
    nr_qpsk_llr_2layers_core_epi16(y1r_sqrt2, y1i_sqrt2, y0r_half, y0i_half, rho10r, rho10i, one_over_2_sqrt_2, &llr1r, &llr1i);

    nr_qpsk_llr_2layers_store_epi16(stream0_256i_out, i, length, llr0r, llr0i);
    nr_qpsk_llr_2layers_store_epi16(stream1_256i_out, i, length, llr1r, llr1i);
  }
#endif
}

void nr_qpsk_llr_2layers(c16_t *stream0_in, c16_t *stream1_in, int16_t *stream0_out, c16_t *rho01, uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho01_128i = (simde__m128i *)rho01;
  simde__m128i *stream0_128i_in = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128i_in = (simde__m128i *)stream1_in;
  simde__m128i *stream0_128i_out = (simde__m128i *)stream0_out;
  simde__m128i one_over_2_sqrt_2 = simde_mm_set1_epi16(23170);

  for (int i = 0; i < length >> 2; i += 2) {
    simde__m128i y0r, y0i, y1r, y1i, rhor, rhoi, llr_r, llr_i;
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128i_in[i], stream0_128i_in[i + 1]);
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128i_in[i], stream1_128i_in[i + 1]);
    oai_mm_separate_real_imag_parts(&rhor, &rhoi, rho01_128i[i], rho01_128i[i + 1]);

    nr_qpsk_llr_2layers_core_epi16(simde_mm_mulhrs_epi16(y0r, one_over_2_sqrt_2),
                                   simde_mm_mulhrs_epi16(y0i, one_over_2_sqrt_2),
                                   simde_mm_srai_epi16(y1r, 1),
                                   simde_mm_srai_epi16(y1i, 1),
                                   rhor,
                                   rhoi,
                                   one_over_2_sqrt_2,
                                   &llr_r,
                                   &llr_i);
    nr_qpsk_llr_2layers_store_epi16(stream0_128i_out, i, length, llr_r, llr_i);
  }
#else
  simde__m256i *rho01_256i = (simde__m256i *)rho01;
  simde__m256i *stream0_256i_in = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256i_in = (simde__m256i *)stream1_in;
  simde__m256i *stream0_256i_out = (simde__m256i *)stream0_out;
  simde__m256i one_over_2_sqrt_2 = simde_mm256_set1_epi16(23170);

  for (int i = 0; i < length >> 3; i += 2) {
    simde__m256i y0r, y0i, y1r, y1i, rhor, rhoi, llr_r, llr_i;
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256i_in[i], stream0_256i_in[i + 1]);
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256i_in[i], stream1_256i_in[i + 1]);
    oai_mm256_separate_real_imag_parts(&rhor, &rhoi, rho01_256i[i], rho01_256i[i + 1]);

    nr_qpsk_llr_2layers_core_epi16(simde_mm256_mulhrs_epi16(y0r, one_over_2_sqrt_2),
                                   simde_mm256_mulhrs_epi16(y0i, one_over_2_sqrt_2),
                                   simde_mm256_srai_epi16(y1r, 1),
                                   simde_mm256_srai_epi16(y1i, 1),
                                   rhor,
                                   rhoi,
                                   one_over_2_sqrt_2,
                                   &llr_r,
                                   &llr_i);
    nr_qpsk_llr_2layers_store_epi16(stream0_256i_out, i, length, llr_r, llr_i);
  }
#endif
}

#ifdef USE_128BIT
// calculate interference magnitude
// tmp_result = ones in shorts corr. to interval 2<=x<=4, tmp_result2 interval < 2, tmp_result3 interval 4<x<6 and tmp_result4
// interval x>6
static inline simde__m128i interference_abs_64qam_epi16(simde__m128i psi, 
                                                        simde__m128i int_ch_mag, 
                                                        simde__m128i int_two_ch_mag, 
                                                        simde__m128i int_three_ch_mag, 
                                                        simde__m128i c1, 
                                                        simde__m128i c3, 
                                                        simde__m128i c5, 
                                                        simde__m128i c7) 
{
  simde__m128i tmp_result  = simde_mm_cmpgt_epi16(int_two_ch_mag, psi);
  simde__m128i tmp_result3 = simde_mm_xor_si128(tmp_result, allones128());
  simde__m128i tmp_result2 = simde_mm_cmpgt_epi16(int_ch_mag, psi);
  tmp_result  = simde_mm_xor_si128(tmp_result, tmp_result2);
  simde__m128i tmp_result4 = simde_mm_cmpgt_epi16(psi, int_three_ch_mag);
  tmp_result3 = simde_mm_xor_si128(tmp_result3, tmp_result4);
  tmp_result  = simde_mm_and_si128(tmp_result, c3);
  tmp_result2 = simde_mm_and_si128(tmp_result2, c1);
  tmp_result3 = simde_mm_and_si128(tmp_result3, c5);
  tmp_result4 = simde_mm_and_si128(tmp_result4, c7);
  tmp_result  = simde_mm_or_si128(tmp_result, tmp_result2);
  tmp_result3 = simde_mm_or_si128(tmp_result3, tmp_result4);
  return simde_mm_or_si128(tmp_result, tmp_result3);
}

// Calculates psi_a = psi_r * a_r + psi_i * a_i
static inline simde__m128i prodsum_psi_a_epi16(simde__m128i psi_r, simde__m128i a_r, simde__m128i psi_i, simde__m128i a_i)
{
  simde__m128i tmp_result = simde_mm_mulhi_epi16(psi_r, a_r);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  simde__m128i tmp_result2 = simde_mm_mulhi_epi16(psi_i, a_i);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  return simde_mm_adds_epi16(tmp_result, tmp_result2);
}

// Calculate interference magnitude
static inline simde__m128i interference_abs_epi16(simde__m128i psi, simde__m128i int_ch_mag, simde__m128i c1, simde__m128i c2)
{
  simde__m128i tmp_result = simde_mm_cmplt_epi16(psi, int_ch_mag);
  simde__m128i tmp_result2 = simde_mm_xor_si128(tmp_result,allones128());
  tmp_result = simde_mm_and_si128(tmp_result, c1);
  tmp_result2 = simde_mm_and_si128(tmp_result2, c2);
  return simde_mm_or_si128(tmp_result, tmp_result2);
}

// Calculates a_sq = int_ch_mag * (a_r^2 + a_i^2) * scale_factor
static inline simde__m128i square_a_epi16(simde__m128i a_r, simde__m128i a_i, simde__m128i int_ch_mag, simde__m128i scale_factor)
{
  simde__m128i tmp_result = simde_mm_mulhi_epi16(a_r, a_r);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm_mulhi_epi16(tmp_result, scale_factor);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm_mulhi_epi16(tmp_result, int_ch_mag);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  simde__m128i tmp_result2 = simde_mm_mulhi_epi16(a_i, a_i);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm_mulhi_epi16(tmp_result2, scale_factor);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm_mulhi_epi16(tmp_result2, int_ch_mag);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  return simde_mm_adds_epi16(tmp_result, tmp_result2);
}

// calculates a_sq = int_ch_mag*(a_r^2 + a_i^2)*scale_factor for 64-QAM
static inline simde__m128i square_a_64qam_epi16(simde__m128i a_r, simde__m128i a_i, simde__m128i int_ch_mag, simde__m128i scale_factor)
{
  simde__m128i tmp_result = simde_mm_mulhi_epi16(a_r, a_r);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm_mulhi_epi16(tmp_result, scale_factor);
  tmp_result = simde_mm_slli_epi16(tmp_result, 3);
  tmp_result = simde_mm_mulhi_epi16(tmp_result, int_ch_mag);
  tmp_result = simde_mm_slli_epi16(tmp_result, 1);
  simde__m128i tmp_result2 = simde_mm_mulhi_epi16(a_i, a_i);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm_mulhi_epi16(tmp_result2, scale_factor);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 3);
  tmp_result2 = simde_mm_mulhi_epi16(tmp_result2, int_ch_mag);
  tmp_result2 = simde_mm_slli_epi16(tmp_result2, 1);
  return simde_mm_adds_epi16(tmp_result, tmp_result2);
}

static inline simde__m128i max_epi16(simde__m128i m0, simde__m128i m1, simde__m128i m2, simde__m128i m3, simde__m128i m4, simde__m128i m5, simde__m128i m6, simde__m128i m7)
{
  simde__m128i a0 = simde_mm_max_epi16(m0, m1);
  simde__m128i a1 = simde_mm_max_epi16(m2, m3);
  simde__m128i a2 = simde_mm_max_epi16(m4, m5);
  simde__m128i a3 = simde_mm_max_epi16(m6, m7);
  simde__m128i b0 = simde_mm_max_epi16(a0, a1);
  simde__m128i b1 = simde_mm_max_epi16(a2, a3);
  return simde_mm_max_epi16(b0, b1);
}

#else

// calculate interference magnitude
// tmp_result = ones in shorts corr. to interval 2<=x<=4, tmp_result2 interval < 2, tmp_result3 interval 4<x<6 and tmp_result4
// interval x>6
static inline simde__m256i interference_abs_64qam_epi16_256(simde__m256i psi, 
                                                            simde__m256i int_ch_mag, 
                                                            simde__m256i int_two_ch_mag, 
                                                            simde__m256i int_three_ch_mag, 
                                                            simde__m256i c1, 
                                                            simde__m256i c3, 
                                                            simde__m256i c5, 
                                                            simde__m256i c7)
{
  simde__m256i tmp_result = simde_mm256_cmpgt_epi16(int_two_ch_mag, psi);
  simde__m256i tmp_result3 = simde_mm256_xor_si256(tmp_result, allones256());
  simde__m256i tmp_result2 = simde_mm256_cmpgt_epi16(int_ch_mag, psi);
  tmp_result = simde_mm256_xor_si256(tmp_result, tmp_result2);
  simde__m256i tmp_result4 = simde_mm256_cmpgt_epi16(psi, int_three_ch_mag);
  tmp_result3 = simde_mm256_xor_si256(tmp_result3, tmp_result4);
  tmp_result = simde_mm256_and_si256(tmp_result, c3);
  tmp_result2 = simde_mm256_and_si256(tmp_result2, c1);
  tmp_result3 = simde_mm256_and_si256(tmp_result3, c5);
  tmp_result4 = simde_mm256_and_si256(tmp_result4, c7);
  tmp_result = simde_mm256_or_si256(tmp_result, tmp_result2);
  tmp_result3 = simde_mm256_or_si256(tmp_result3, tmp_result4);
  return simde_mm256_or_si256(tmp_result, tmp_result3);
}

// calculates psi_a = psi_r*a_r + psi_i*a_i
static inline simde__m256i prodsum_psi_a_epi16_256(simde__m256i psi_r, simde__m256i a_r, simde__m256i psi_i, simde__m256i a_i)
{
  simde__m256i tmp_result = simde_mm256_mulhi_epi16(psi_r, a_r);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  simde__m256i tmp_result2 = simde_mm256_mulhi_epi16(psi_i, a_i);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  return simde_mm256_adds_epi16(tmp_result, tmp_result2);
}

// Calculate interference magnitude
static inline simde__m256i interference_abs_epi16_256(simde__m256i psi, simde__m256i int_ch_mag, simde__m256i c1, simde__m256i c2)
{
  simde__m256i tmp_result = simde_mm256_cmpgt_epi16(int_ch_mag, psi);
  simde__m256i tmp_result2 = simde_mm256_xor_si256(tmp_result,  allones256());
  tmp_result = simde_mm256_and_si256(tmp_result, c1);
  tmp_result2 = simde_mm256_and_si256(tmp_result2, c2);
  return simde_mm256_or_si256(tmp_result, tmp_result2);
}

// Calculates a_sq = int_ch_mag * (a_r^2 + a_i^2) * scale_factor
static inline simde__m256i square_a_epi16_256(simde__m256i a_r, simde__m256i a_i, simde__m256i int_ch_mag, simde__m256i scale_factor)
{
  simde__m256i tmp_result = simde_mm256_mulhi_epi16(a_r, a_r);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm256_mulhi_epi16(tmp_result, scale_factor);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm256_mulhi_epi16(tmp_result, int_ch_mag);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  simde__m256i tmp_result2 = simde_mm256_mulhi_epi16(a_i, a_i);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm256_mulhi_epi16(tmp_result2, scale_factor);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm256_mulhi_epi16(tmp_result2, int_ch_mag);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  return simde_mm256_adds_epi16(tmp_result, tmp_result2);
}

// calculates a_sq = int_ch_mag*(a_r^2 + a_i^2)*scale_factor for 64-QAM
static inline simde__m256i square_a_64qam_epi16_256(simde__m256i a_r, simde__m256i a_i, simde__m256i int_ch_mag, simde__m256i scale_factor)
{
  simde__m256i tmp_result = simde_mm256_mulhi_epi16(a_r, a_r);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  tmp_result = simde_mm256_mulhi_epi16(tmp_result, scale_factor);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 3);
  tmp_result = simde_mm256_mulhi_epi16(tmp_result, int_ch_mag);
  tmp_result = simde_mm256_slli_epi16(tmp_result, 1);
  simde__m256i tmp_result2 = simde_mm256_mulhi_epi16(a_i, a_i);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  tmp_result2 = simde_mm256_mulhi_epi16(tmp_result2, scale_factor);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 3);
  tmp_result2 = simde_mm256_mulhi_epi16(tmp_result2, int_ch_mag);
  tmp_result2 = simde_mm256_slli_epi16(tmp_result2, 1);
  return simde_mm256_adds_epi16(tmp_result, tmp_result2);
}

static inline simde__m256i max_epi16_256(simde__m256i m0, simde__m256i m1, simde__m256i m2, simde__m256i m3, simde__m256i m4, simde__m256i m5, simde__m256i m6, simde__m256i m7)
{
  simde__m256i a0 = simde_mm256_max_epi16(m0, m1);
  simde__m256i a1 = simde_mm256_max_epi16(m2, m3);
  simde__m256i a2 = simde_mm256_max_epi16(m4, m5);
  simde__m256i a3 = simde_mm256_max_epi16(m6, m7);
  simde__m256i b0 = simde_mm256_max_epi16(a0, a1);
  simde__m256i b1 = simde_mm256_max_epi16(a2, a3);
  return simde_mm256_max_epi16(b0, b1);
}

#endif

/*
 * This function computes the LLRs of stream 0 (s_0) in presence of the interfering stream 1 (s_1) assuming that both symbols are
 * 16QAM. It can be used for both MU-MIMO interference-aware receiver or for SU-MIMO receivers.
 *
 * Input:
 *   stream0_in:  MF filter output for 1st stream, i.e., y0' = h0'*y0
 *   stream1_in:  MF filter output for 2nd stream, i.e., y1' = h1'*y0
 *   ch_mag:      2*h0/sqrt(10), [Re0 Im0 Re1 Im1] s.t. Im0=Re0, Im1=Re1, etc
 *   ch_mag_i:    2*h1/sqrt(10), [Re0 Im0 Re1 Im1] s.t. Im0=Re0, Im1=Re1, etc
 *   rho01:       Channel cross correlation, i.e., rho01 = h0'*h1
 *   length:      Number of resource elements
 *
 * Output:
 *   stream0_out: Output LLRs for 1st stream
 */
void nr_16qam_llr_2layers(c16_t *stream0_in,
                          c16_t *stream1_in,
                          c16_t *ch_mag,
                          c16_t *ch_mag_i,
                          int16_t *stream0_out,
                          c16_t *rho01,
                          uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho01_128i = (simde__m128i *)rho01;
  simde__m128i *stream0_128i_in = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128i_in = (simde__m128i *)stream1_in;
  simde__m128i *stream0_128i_out = (simde__m128i *)stream0_out;
  simde__m128i *ch_mag_128i = (simde__m128i *)ch_mag;
  simde__m128i *ch_mag_128i_i = (simde__m128i *)ch_mag_i;

  simde__m128i ONE_OVER_SQRT_10 = simde_mm_set1_epi16(20724); // round(1/sqrt(10)*2^16)
  simde__m128i ONE_OVER_SQRT_10_Q15 = simde_mm_set1_epi16(10362); // round(1/sqrt(10)*2^15)
  simde__m128i THREE_OVER_SQRT_10 = simde_mm_set1_epi16(31086); // round(3/sqrt(10)*2^15)
  simde__m128i SQRT_10_OVER_FOUR = simde_mm_set1_epi16(25905); // round(sqrt(10)/4*2^15)
  simde__m128i ONE_OVER_TWO_SQRT_10 = simde_mm_set1_epi16(10362); // round(1/2/sqrt(10)*2^16)
  simde__m128i NINE_OVER_TWO_SQRT_10 = simde_mm_set1_epi16(23315); // round(9/2/sqrt(10)*2^14)
  simde__m128i ch_mag_des, ch_mag_int;
  simde__m128i y0r_over_sqrt10;
  simde__m128i y0i_over_sqrt10;
  simde__m128i y0r_three_over_sqrt10;
  simde__m128i y0i_three_over_sqrt10;
  simde__m128i ch_mag_over_10;
  simde__m128i ch_mag_over_2;
  simde__m128i ch_mag_9_over_10;

  simde__m128i xmm0;
  simde__m128i xmm1;
  simde__m128i xmm2;
  simde__m128i xmm3;
  simde__m128i xmm4;
  simde__m128i xmm5;
  simde__m128i xmm6;
  simde__m128i xmm7;

  simde__m128i rho_rpi;
  simde__m128i rho_rmi;
  simde__m128i rho_rs[8];
  simde__m128i psi_rs[16];
  simde__m128i psi_is[16];
  simde__m128i a_rs[16];
  simde__m128i a_is[16];
  simde__m128i psi_as[16];
  simde__m128i a_sqs[16];
  simde__m128i y0_s[8];

  simde__m128i y0r;
  simde__m128i y0i;
  simde__m128i y1r;
  simde__m128i y1i;

  // In one iteration, we deal with 8 REs
  for (int i = 0; i < length >> 2; i += 2) {

    // Get rho
    oai_mm_separate_real_imag_parts(&xmm2, &xmm3, rho01_128i[i], rho01_128i[i + 1]);
    rho_rpi = simde_mm_adds_epi16(xmm2, xmm3); // rho = Re(rho) + Im(rho)
    rho_rmi = simde_mm_subs_epi16(xmm2, xmm3); // rho* = Re(rho) - Im(rho)

    // Compute the different rhos
    rho_rs[0] = simde_mm_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_10);
    rho_rs[4] = simde_mm_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_10);
    rho_rs[3] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_10), 1);
    rho_rs[7] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_10), 1);

    xmm4 = simde_mm_mulhi_epi16(xmm2, ONE_OVER_SQRT_10); // Re(rho)
    xmm5 = simde_mm_mulhi_epi16(xmm3, THREE_OVER_SQRT_10); // Im(rho)
    xmm5 = simde_mm_slli_epi16(xmm5, 1);

    rho_rs[1] = simde_mm_adds_epi16(xmm4, xmm5);
    rho_rs[5] = simde_mm_subs_epi16(xmm4, xmm5);

    xmm6 = simde_mm_mulhi_epi16(xmm2, THREE_OVER_SQRT_10); // Re(rho)
    xmm7 = simde_mm_mulhi_epi16(xmm3, ONE_OVER_SQRT_10); // Im(rho)
    xmm6 = simde_mm_slli_epi16(xmm6, 1);

    rho_rs[2] = simde_mm_adds_epi16(xmm6, xmm7);
    rho_rs[6] = simde_mm_subs_epi16(xmm6, xmm7);

    // Rearrange interfering MF output
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128i_in[i], stream1_128i_in[i + 1]);

    // |  [Re(rho)+ Im(rho)]/sqrt(10) - y1r  |
    for(int j=0; j<8; j++){ // psi_rs[0~7], rho_rs[0~7]
      psi_rs[j]  = simde_mm_abs_epi16(  simde_mm_subs_epi16(rho_rs[j], y1r)  );
    }
    for(int j=8; j<16; j++){ // psi_rs[8~16], rho_rs[4,5,6,7,0,1,2,3]
      psi_rs[j]  = simde_mm_abs_epi16(  simde_mm_adds_epi16(rho_rs[(j-4) & 7], y1r)  );
    }
    const uint8_t rho_rs_indexes[16] = {4,6,5,7,0,2,1,3,0,2,1,3,4,6,5,7};
    for(int k=0; k<16; k+=8){ // psi_is[0~15], sub(rho_rs[4,6,5,7]), add(rho_rs[0,2,1,3]), sub(rho_rs[0,2,1,3]), add(rho_rs[4,6,5,7])
      for(int j=k; j<k+4; j++){
        psi_is[j]    = simde_mm_abs_epi16(  simde_mm_subs_epi16(rho_rs[rho_rs_indexes[j]], y1i)  );
        psi_is[j+4]  = simde_mm_abs_epi16(  simde_mm_adds_epi16(rho_rs[rho_rs_indexes[j+4]], y1i)  );
      }
    }

    // Rearrange desired MF output
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128i_in[i], stream0_128i_in[i + 1]);

    // Rearrange desired channel magnitudes
    // [|h|^2(1),|h|^2(2),|h|^2(3),|h|^2(4)]*(2/sqrt(10))
    oai_mm_separate_real_imag_parts(&ch_mag_des, &xmm2, ch_mag_128i[i], ch_mag_128i[i + 1]);

    // Rearrange interfering channel magnitudes
    oai_mm_separate_real_imag_parts(&ch_mag_int, &xmm2, ch_mag_128i_i[i], ch_mag_128i_i[i + 1]);

    // Scale MF output of desired signal
    y0r_over_sqrt10 = simde_mm_mulhi_epi16(y0r, ONE_OVER_SQRT_10);
    y0i_over_sqrt10 = simde_mm_mulhi_epi16(y0i, ONE_OVER_SQRT_10);
    y0r_three_over_sqrt10 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0r, THREE_OVER_SQRT_10), 1);
    y0i_three_over_sqrt10 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0i, THREE_OVER_SQRT_10), 1);

    // Compute necessary combination of required terms
    y0_s[0] = simde_mm_adds_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
    y0_s[4] = simde_mm_subs_epi16(y0r_over_sqrt10, y0i_over_sqrt10);

    y0_s[1] = simde_mm_adds_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
    y0_s[5] = simde_mm_subs_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);

    y0_s[2] = simde_mm_adds_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
    y0_s[6] = simde_mm_subs_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);

    y0_s[3] = simde_mm_adds_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);
    y0_s[7] = simde_mm_subs_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);

    for(int j=0; j<16; j++){
      // Compute optimal interfering symbol magnitude
      a_rs[j] = interference_abs_epi16(psi_rs[j], ch_mag_int, ONE_OVER_SQRT_10_Q15, THREE_OVER_SQRT_10);
      a_is[j] = interference_abs_epi16(psi_is[j], ch_mag_int, ONE_OVER_SQRT_10_Q15, THREE_OVER_SQRT_10);

      // Calculation of groups of two terms in the bit metric involving product of psi and interference magnitude
      psi_as[j] = prodsum_psi_a_epi16(psi_rs[j], a_rs[j], psi_is[j], a_is[j]);

      // squared interference magnitude times int. ch. power
      a_sqs[j] = square_a_epi16(a_rs[j], a_is[j], ch_mag_int, SQRT_10_OVER_FOUR);
    }

    // Computing different multiples of channel norms
    ch_mag_over_10 = simde_mm_mulhi_epi16(ch_mag_des, ONE_OVER_TWO_SQRT_10);
    ch_mag_over_2 = simde_mm_mulhi_epi16(ch_mag_des, SQRT_10_OVER_FOUR);
    ch_mag_over_2 = simde_mm_slli_epi16(ch_mag_over_2, 1);
    ch_mag_9_over_10 = simde_mm_mulhi_epi16(ch_mag_des, NINE_OVER_TWO_SQRT_10);
    ch_mag_9_over_10 = simde_mm_slli_epi16(ch_mag_9_over_10, 2);

    /// Compute bit metrics (lambda)
    simde__m128i bit_mets[16];
    for(int j=0; j<8; j+=4){
      bit_mets[j+0] = simde_mm_subs_epi16(psi_as[j+0], a_sqs[j+0]);
      bit_mets[j+0] = simde_mm_adds_epi16(bit_mets[j+0], y0_s[j+0]);
      bit_mets[j+0] = simde_mm_subs_epi16(bit_mets[j+0], ch_mag_over_10);

      bit_mets[j+1] = simde_mm_subs_epi16(psi_as[j+1], a_sqs[j+1]);
      bit_mets[j+1] = simde_mm_adds_epi16(bit_mets[j+1], y0_s[j+1]);
      bit_mets[j+1] = simde_mm_subs_epi16(bit_mets[j+1], ch_mag_over_2);
      bit_mets[j+2] = simde_mm_subs_epi16(psi_as[j+2], a_sqs[j+2]);
      bit_mets[j+2] = simde_mm_adds_epi16(bit_mets[j+2], y0_s[j+2]);
      bit_mets[j+2] = simde_mm_subs_epi16(bit_mets[j+2], ch_mag_over_2);

      bit_mets[j+3] = simde_mm_subs_epi16(psi_as[j+3], a_sqs[j+3]);
      bit_mets[j+3] = simde_mm_adds_epi16(bit_mets[j+3], y0_s[j+3]);
      bit_mets[j+3] = simde_mm_subs_epi16(bit_mets[j+3], ch_mag_9_over_10);
    }
    for(int j=8; j<16; j+=4){
      bit_mets[j+0] = simde_mm_subs_epi16(psi_as[j+0], a_sqs[j+0]);
      bit_mets[j+0] = simde_mm_subs_epi16(bit_mets[j+0], y0_s[(j - 4) & 0x07]);
      bit_mets[j+0] = simde_mm_subs_epi16(bit_mets[j+0], ch_mag_over_10);

      bit_mets[j+1] = simde_mm_subs_epi16(psi_as[j+1], a_sqs[j+1]);
      bit_mets[j+1] = simde_mm_subs_epi16(bit_mets[j+1], y0_s[(j - 3) & 0x07]);
      bit_mets[j+1] = simde_mm_subs_epi16(bit_mets[j+1], ch_mag_over_2);
      bit_mets[j+2] = simde_mm_subs_epi16(psi_as[j+2], a_sqs[j+2]);
      bit_mets[j+2] = simde_mm_subs_epi16(bit_mets[j+2], y0_s[(j - 2) & 0x07]);
      bit_mets[j+2] = simde_mm_subs_epi16(bit_mets[j+2], ch_mag_over_2);

      bit_mets[j+3] = simde_mm_subs_epi16(psi_as[j+3], a_sqs[j+3]);
      bit_mets[j+3] = simde_mm_subs_epi16(bit_mets[j+3], y0_s[(j - 1) & 0x07]);
      bit_mets[j+3] = simde_mm_subs_epi16(bit_mets[j+3], ch_mag_9_over_10);
    }
    /// Compute the LLRs

    // LLR = lambda(c==1) - lambda(c==0)

    // LLR of the first bit: Bit = 1
    simde__m128i logmax_num_re0 = max_epi16(bit_mets[8],  bit_mets[9],  bit_mets[10], bit_mets[11], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
    // LLR of the first bit: Bit = 0
    simde__m128i logmax_den_re0 = max_epi16(bit_mets[0],  bit_mets[1],  bit_mets[2],  bit_mets[3],  bit_mets[4],  bit_mets[5],  bit_mets[6],  bit_mets[7]);
    // LLR of the second bit: Bit = 1
    simde__m128i logmax_num_re1 = max_epi16(bit_mets[4],  bit_mets[5],  bit_mets[6],  bit_mets[7],  bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
    // LLR of the second bit: Bit = 0
    simde__m128i logmax_den_re1 = max_epi16(bit_mets[0],  bit_mets[1],  bit_mets[3],  bit_mets[2],  bit_mets[8],  bit_mets[9],  bit_mets[10], bit_mets[11]);
    // LLR of the third bit: Bit = 1
    simde__m128i logmax_num_im0 = max_epi16(bit_mets[2],  bit_mets[3],  bit_mets[6],  bit_mets[7],  bit_mets[10], bit_mets[11], bit_mets[14], bit_mets[15]);
    // LLR of the third bit: Bit = 0
    simde__m128i logmax_den_im0 = max_epi16(bit_mets[0],  bit_mets[1],  bit_mets[4],  bit_mets[5],  bit_mets[8],  bit_mets[9],  bit_mets[12], bit_mets[13]);
    // LLR of the fourth bit: Bit = 1
    simde__m128i logmax_num_im1 = max_epi16(bit_mets[1],  bit_mets[3],  bit_mets[5],  bit_mets[7],  bit_mets[9],  bit_mets[11], bit_mets[13], bit_mets[15]);
    // LLR of the fourth bit: Bit = 0
    simde__m128i logmax_den_im1 = max_epi16(bit_mets[0],  bit_mets[2],  bit_mets[4],  bit_mets[6],  bit_mets[8],  bit_mets[10], bit_mets[12], bit_mets[14]);


    y0r = simde_mm_subs_epi16(logmax_den_re0, logmax_num_re0); // LLR of first bit  [L1(1), L1(2), L1(3), L1(4), L1(5), L1(6), L1(7), L1(8)]
    y1r = simde_mm_subs_epi16(logmax_den_re1, logmax_num_re1); // LLR of second bit [L2(1), L2(2), L2(3), L2(4), L2(5), L2(6), L2(7), L2(8)]
    y0i = simde_mm_subs_epi16(logmax_den_im0, logmax_num_im0); // LLR of third bit  [L3(1), L3(2), L3(3), L3(4), L3(5), L3(6), L3(7), L3(8)]
    y1i = simde_mm_subs_epi16(logmax_den_im1, logmax_num_im1); // LLR of fourth bit [L4(1), L4(2), L4(3), L4(4), L4(5), L4(6), L4(7), L4(8)]

    // Pack LLRs in output
    xmm0 = simde_mm_unpacklo_epi16(y0r, y1r); // [L1(1), L2(1), L1(2), L2(2), L1(3), L2(3), L1(4), L2(4)]
    xmm1 = simde_mm_unpackhi_epi16(y0r, y1r); // [L1(5), L2(5), L1(6), L2(6), L1(7), L2(7), L1(8), L2(8)]
    xmm2 = simde_mm_unpacklo_epi16(y0i, y1i); // [L3(1), L4(1), L3(2), L4(2), L3(3), L4(3), L3(4), L4(4)]
    xmm3 = simde_mm_unpackhi_epi16(y0i, y1i); // [L3(5), L4(5), L3(6), L4(6), L3(7), L4(7), L3(8), L4(8)]
    stream0_128i_out[2 * i + 0] = simde_mm_unpacklo_epi32(xmm0, xmm2); // 8 LLRs, 2 REs
    stream0_128i_out[2 * i + 1] = simde_mm_unpackhi_epi32(xmm0, xmm2); // 8 LLRs, 2 REs
    stream0_128i_out[2 * i + 2] = simde_mm_unpacklo_epi32(xmm1, xmm3); // 8 LLRs, 2 REs
    stream0_128i_out[2 * i + 3] = simde_mm_unpackhi_epi32(xmm1, xmm3); // 8 LLRs, 2 REs
  }
#else
  simde__m256i *rho01_256i = (simde__m256i *)rho01;
  simde__m256i *stream0_256i_in = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256i_in = (simde__m256i *)stream1_in;
  simde__m256i *stream0_256i_out = (simde__m256i *)stream0_out;
  simde__m256i *ch_mag_256i = (simde__m256i *)ch_mag;
  simde__m256i *ch_mag_256i_i = (simde__m256i *)ch_mag_i;

  simde__m256i ONE_OVER_SQRT_10 = simde_mm256_set1_epi16(20724); // round(1/sqrt(10)*2^16)
  simde__m256i ONE_OVER_SQRT_10_Q15 = simde_mm256_set1_epi16(10362); // round(1/sqrt(10)*2^15)
  simde__m256i THREE_OVER_SQRT_10 = simde_mm256_set1_epi16(31086); // round(3/sqrt(10)*2^15)
  simde__m256i SQRT_10_OVER_FOUR = simde_mm256_set1_epi16(25905); // round(sqrt(10)/4*2^15)
  simde__m256i ONE_OVER_TWO_SQRT_10 = simde_mm256_set1_epi16(10362); // round(1/2/sqrt(10)*2^16)
  simde__m256i NINE_OVER_TWO_SQRT_10 = simde_mm256_set1_epi16(23315); // round(9/2/sqrt(10)*2^14)
  simde__m256i ch_mag_des, ch_mag_int;
  simde__m256i y0r_over_sqrt10;
  simde__m256i y0i_over_sqrt10;
  simde__m256i y0r_three_over_sqrt10;
  simde__m256i y0i_three_over_sqrt10;
  simde__m256i ch_mag_over_10;
  simde__m256i ch_mag_over_2;
  simde__m256i ch_mag_9_over_10;

  simde__m256i xmm2;
  simde__m256i xmm3;
  simde__m256i xmm4;
  simde__m256i xmm5;
  simde__m256i xmm6;
  simde__m256i xmm7;

  simde__m256i rho_rpi;
  simde__m256i rho_rmi;
  simde__m256i rho_rs[8];
  simde__m256i psi_rs[16];
  simde__m256i psi_is[16];
  simde__m256i a_rs[16];
  simde__m256i a_is[16];
  simde__m256i psi_as[16];
  simde__m256i a_sqs[16];
  simde__m256i y0_s[8];

  simde__m256i y0r;
  simde__m256i y0i;
  simde__m256i y1r;
  simde__m256i y1i;

  // In one iteration, we deal with 8 REs
  for (int i = 0; i < length >> 3; i += 2) {

    // Get rho
    oai_mm256_separate_real_imag_parts(&xmm2, &xmm3, rho01_256i[i], rho01_256i[i + 1]);
    rho_rpi = simde_mm256_adds_epi16(xmm2, xmm3); // rho = Re(rho) + Im(rho)
    rho_rmi = simde_mm256_subs_epi16(xmm2, xmm3); // rho* = Re(rho) - Im(rho)

    // Compute the different rhos
    rho_rs[0] = simde_mm256_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_10);
    rho_rs[4] = simde_mm256_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_10);
    rho_rs[3] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_10), 1);
    rho_rs[7] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_10), 1);

    xmm4 = simde_mm256_mulhi_epi16(xmm2, ONE_OVER_SQRT_10); // Re(rho)
    xmm5 = simde_mm256_mulhi_epi16(xmm3, THREE_OVER_SQRT_10); // Im(rho)
    xmm5 = simde_mm256_slli_epi16(xmm5, 1);

    rho_rs[1] = simde_mm256_adds_epi16(xmm4, xmm5);
    rho_rs[5] = simde_mm256_subs_epi16(xmm4, xmm5);

    xmm6 = simde_mm256_mulhi_epi16(xmm2, THREE_OVER_SQRT_10); // Re(rho)
    xmm7 = simde_mm256_mulhi_epi16(xmm3, ONE_OVER_SQRT_10); // Im(rho)
    xmm6 = simde_mm256_slli_epi16(xmm6, 1);

    rho_rs[2] = simde_mm256_adds_epi16(xmm6, xmm7);
    rho_rs[6] = simde_mm256_subs_epi16(xmm6, xmm7);

    // Rearrange interfering MF output
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256i_in[i], stream1_256i_in[i + 1]);

    // |  [Re(rho)+ Im(rho)]/sqrt(10) - y1r  |
    for(int j=0; j<8; j++){ // psi_rs[0~7], rho_rs[0~7]
      psi_rs[j]  = simde_mm256_abs_epi16(  simde_mm256_subs_epi16(rho_rs[j], y1r)  );
    }
    for(int j=8; j<16; j++){ // psi_rs[8~16], rho_rs[4,5,6,7,0,1,2,3]
      psi_rs[j]  = simde_mm256_abs_epi16(  simde_mm256_adds_epi16(rho_rs[(j-4) & 7], y1r)  );
    }
    const uint8_t rho_rs_indexes[16] = {4,6,5,7,0,2,1,3,0,2,1,3,4,6,5,7};
    for(int k=0; k<16; k+=8){ // psi_is[0~15], sub(rho_rs[4,6,5,7]), add(rho_rs[0,2,1,3]), sub(rho_rs[0,2,1,3]), add(rho_rs[4,6,5,7])
      for(int j=k; j<k+4; j++){
        psi_is[j]    = simde_mm256_abs_epi16(  simde_mm256_subs_epi16(rho_rs[rho_rs_indexes[j]], y1i)  );
        psi_is[j+4]  = simde_mm256_abs_epi16(  simde_mm256_adds_epi16(rho_rs[rho_rs_indexes[j+4]], y1i)  );
      }
    }

    // Rearrange desired MF output
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256i_in[i], stream0_256i_in[i + 1]);

    // Rearrange desired channel magnitudes
    // [|h|^2(1),|h|^2(2),|h|^2(3),|h|^2(4)]*(2/sqrt(10))
    oai_mm256_separate_real_imag_parts(&ch_mag_des, &xmm2, ch_mag_256i[i], ch_mag_256i[i + 1]);

    // Rearrange interfering channel magnitudes
    oai_mm256_separate_real_imag_parts(&ch_mag_int, &xmm2, ch_mag_256i_i[i], ch_mag_256i_i[i + 1]);

    // Scale MF output of desired signal
    y0r_over_sqrt10 = simde_mm256_mulhi_epi16(y0r, ONE_OVER_SQRT_10);
    y0i_over_sqrt10 = simde_mm256_mulhi_epi16(y0i, ONE_OVER_SQRT_10);
    y0r_three_over_sqrt10 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0r, THREE_OVER_SQRT_10), 1);
    y0i_three_over_sqrt10 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0i, THREE_OVER_SQRT_10), 1);

    // Compute necessary combination of required terms
    y0_s[0] = simde_mm256_adds_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
    y0_s[4] = simde_mm256_subs_epi16(y0r_over_sqrt10, y0i_over_sqrt10);

    y0_s[1] = simde_mm256_adds_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
    y0_s[5] = simde_mm256_subs_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);

    y0_s[2] = simde_mm256_adds_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
    y0_s[6] = simde_mm256_subs_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);

    y0_s[3] = simde_mm256_adds_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);
    y0_s[7] = simde_mm256_subs_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);

    for(int j=0; j<16; j++){
      // Compute optimal interfering symbol magnitude
      a_rs[j] = interference_abs_epi16_256(psi_rs[j], ch_mag_int, ONE_OVER_SQRT_10_Q15, THREE_OVER_SQRT_10);
      a_is[j] = interference_abs_epi16_256(psi_is[j], ch_mag_int, ONE_OVER_SQRT_10_Q15, THREE_OVER_SQRT_10);

      // Calculation of groups of two terms in the bit metric involving product of psi and interference magnitude
      psi_as[j] = prodsum_psi_a_epi16_256(psi_rs[j], a_rs[j], psi_is[j], a_is[j]);

      // squared interference magnitude times int. ch. power
      a_sqs[j] = square_a_epi16_256(a_rs[j], a_is[j], ch_mag_int, SQRT_10_OVER_FOUR);
    }

    // Computing different multiples of channel norms
    ch_mag_over_10 = simde_mm256_mulhi_epi16(ch_mag_des, ONE_OVER_TWO_SQRT_10);
    ch_mag_over_2 = simde_mm256_mulhi_epi16(ch_mag_des, SQRT_10_OVER_FOUR);
    ch_mag_over_2 = simde_mm256_slli_epi16(ch_mag_over_2, 1);
    ch_mag_9_over_10 = simde_mm256_mulhi_epi16(ch_mag_des, NINE_OVER_TWO_SQRT_10);
    ch_mag_9_over_10 = simde_mm256_slli_epi16(ch_mag_9_over_10, 2);

    /// Compute bit metrics (lambda)

    simde__m256i bit_mets[16];
    for(int j=0; j<8; j+=4){
      bit_mets[j+0] = simde_mm256_subs_epi16(psi_as[j+0], a_sqs[j+0]);
      bit_mets[j+0] = simde_mm256_adds_epi16(bit_mets[j+0], y0_s[j+0]);
      bit_mets[j+0] = simde_mm256_subs_epi16(bit_mets[j+0], ch_mag_over_10);

      bit_mets[j+1] = simde_mm256_subs_epi16(psi_as[j+1], a_sqs[j+1]);
      bit_mets[j+1] = simde_mm256_adds_epi16(bit_mets[j+1], y0_s[j+1]);
      bit_mets[j+1] = simde_mm256_subs_epi16(bit_mets[j+1], ch_mag_over_2);
      bit_mets[j+2] = simde_mm256_subs_epi16(psi_as[j+2], a_sqs[j+2]);
      bit_mets[j+2] = simde_mm256_adds_epi16(bit_mets[j+2], y0_s[j+2]);
      bit_mets[j+2] = simde_mm256_subs_epi16(bit_mets[j+2], ch_mag_over_2);

      bit_mets[j+3] = simde_mm256_subs_epi16(psi_as[j+3], a_sqs[j+3]);
      bit_mets[j+3] = simde_mm256_adds_epi16(bit_mets[j+3], y0_s[j+3]);
      bit_mets[j+3] = simde_mm256_subs_epi16(bit_mets[j+3], ch_mag_9_over_10);
    }
    for(int j=8; j<16; j+=4){
      bit_mets[j+0] = simde_mm256_subs_epi16(psi_as[j+0], a_sqs[j+0]);
      bit_mets[j+0] = simde_mm256_subs_epi16(bit_mets[j+0], y0_s[(j - 4) & 0x07]);
      bit_mets[j+0] = simde_mm256_subs_epi16(bit_mets[j+0], ch_mag_over_10);

      bit_mets[j+1] = simde_mm256_subs_epi16(psi_as[j+1], a_sqs[j+1]);
      bit_mets[j+1] = simde_mm256_subs_epi16(bit_mets[j+1], y0_s[(j - 3) & 0x07]);
      bit_mets[j+1] = simde_mm256_subs_epi16(bit_mets[j+1], ch_mag_over_2);
      bit_mets[j+2] = simde_mm256_subs_epi16(psi_as[j+2], a_sqs[j+2]);
      bit_mets[j+2] = simde_mm256_subs_epi16(bit_mets[j+2], y0_s[(j - 2) & 0x07]);
      bit_mets[j+2] = simde_mm256_subs_epi16(bit_mets[j+2], ch_mag_over_2);

      bit_mets[j+3] = simde_mm256_subs_epi16(psi_as[j+3], a_sqs[j+3]);
      bit_mets[j+3] = simde_mm256_subs_epi16(bit_mets[j+3], y0_s[(j - 1) & 0x07]);
      bit_mets[j+3] = simde_mm256_subs_epi16(bit_mets[j+3], ch_mag_9_over_10);
    }

    /// Compute the LLRs

    // LLR = lambda(c==1) - lambda(c==0)

    // LLR of the first bit: Bit = 1
    simde__m256i logmax_num_re0 = max_epi16_256(bit_mets[8],  bit_mets[9],  bit_mets[10], bit_mets[11], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
    // LLR of the first bit: Bit = 0
    simde__m256i logmax_den_re0 = max_epi16_256(bit_mets[0],  bit_mets[1],  bit_mets[2],  bit_mets[3],  bit_mets[4],  bit_mets[5],  bit_mets[6],  bit_mets[7]);
    // LLR of the second bit: Bit = 1
    simde__m256i logmax_num_re1 = max_epi16_256(bit_mets[4],  bit_mets[5],  bit_mets[6],  bit_mets[7],  bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
    // LLR of the second bit: Bit = 0
    simde__m256i logmax_den_re1 = max_epi16_256(bit_mets[0],  bit_mets[1],  bit_mets[3],  bit_mets[2],  bit_mets[8],  bit_mets[9],  bit_mets[10], bit_mets[11]);
    // LLR of the third bit: Bit = 1
    simde__m256i logmax_num_im0 = max_epi16_256(bit_mets[2],  bit_mets[3],  bit_mets[6],  bit_mets[7],  bit_mets[10], bit_mets[11], bit_mets[14], bit_mets[15]);
    // LLR of the third bit: Bit = 0
    simde__m256i logmax_den_im0 = max_epi16_256(bit_mets[0],  bit_mets[1],  bit_mets[4],  bit_mets[5],  bit_mets[8],  bit_mets[9],  bit_mets[12], bit_mets[13]);
    // LLR of the fourth bit: Bit = 1
    simde__m256i logmax_num_im1 = max_epi16_256(bit_mets[1],  bit_mets[3],  bit_mets[5],  bit_mets[7],  bit_mets[9],  bit_mets[11], bit_mets[13], bit_mets[15]);
    // LLR of the fourth bit: Bit = 0
    simde__m256i logmax_den_im1 = max_epi16_256(bit_mets[0],  bit_mets[2],  bit_mets[4],  bit_mets[6],  bit_mets[8],  bit_mets[10], bit_mets[12], bit_mets[14]);

    y0r = simde_mm256_subs_epi16(logmax_den_re0, logmax_num_re0); // LLR of first bit  [L1(1), L1(2), L1(3), L1(4), L1(5), L1(6), L1(7), L1(8)...]
    y1r = simde_mm256_subs_epi16(logmax_den_re1, logmax_num_re1); // LLR of second bit [L2(1), L2(2), L2(3), L2(4), L2(5), L2(6), L2(7), L2(8)...]
    y0i = simde_mm256_subs_epi16(logmax_den_im0, logmax_num_im0); // LLR of third bit  [L3(1), L3(2), L3(3), L3(4), L3(5), L3(6), L3(7), L3(8)...]
    y1i = simde_mm256_subs_epi16(logmax_den_im1, logmax_num_im1); // LLR of fourth bit [L4(1), L4(2), L4(3), L4(4), L4(5), L4(6), L4(7), L4(8)...]

    // Pack LLRs in output
    simde__m128i* y0r_128 = (simde__m128i *) &y0r;
    simde__m128i* y1r_128 = (simde__m128i *) &y1r;
    simde__m128i* y0i_128 = (simde__m128i *) &y0i;
    simde__m128i* y1i_128 = (simde__m128i *) &y1i;
    simde__m128i xmm0_128, xmm1_128, xmm2_128, xmm3_128;
    xmm0_128 = simde_mm_unpacklo_epi16(y0r_128[0], y1r_128[0]); // [L1(1), L2(1), L1(2), L2(2), L1(3), L2(3), L1(4), L2(4)]
    xmm1_128 = simde_mm_unpackhi_epi16(y0r_128[0], y1r_128[0]); // [L1(5), L2(5), L1(6), L2(6), L1(7), L2(7), L1(8), L2(8)]
    xmm2_128 = simde_mm_unpacklo_epi16(y0i_128[0], y1i_128[0]); // [L3(1), L4(1), L3(2), L4(2), L3(3), L4(3), L3(4), L4(4)]
    xmm3_128 = simde_mm_unpackhi_epi16(y0i_128[0], y1i_128[0]); // [L3(5), L4(5), L3(6), L4(6), L3(7), L4(7), L3(8), L4(8)]
    simde__m128i* stream0_128i_out = (simde__m128i *) &stream0_256i_out[2 * i + 0];
    stream0_128i_out[0] = simde_mm_unpacklo_epi32(xmm0_128, xmm2_128); // 8 LLRs, 2 REs
    stream0_128i_out[1] = simde_mm_unpackhi_epi32(xmm0_128, xmm2_128); // 8 LLRs, 2 REs
    stream0_128i_out[2] = simde_mm_unpacklo_epi32(xmm1_128, xmm3_128); // 8 LLRs, 2 REs
    stream0_128i_out[3] = simde_mm_unpackhi_epi32(xmm1_128, xmm3_128); // 8 LLRs, 2 REs
    xmm0_128 = simde_mm_unpacklo_epi16(y0r_128[1], y1r_128[1]); // [L1(9), L2(9), L1(10), L2(10), L1(11), L2(11), L1(12), L2(12)]
    xmm1_128 = simde_mm_unpackhi_epi16(y0r_128[1], y1r_128[1]); // [L1(13), L2(13), L1(14), L2(14), L1(15), L2(15), L1(16), L2(16)]
    xmm2_128 = simde_mm_unpacklo_epi16(y0i_128[1], y1i_128[1]); // [L3(9), L4(9), L3(10), L4(10), L3(11), L4(11), L3(12), L4(12)]
    xmm3_128 = simde_mm_unpackhi_epi16(y0i_128[1], y1i_128[1]); // [L3(13), L4(13), L3(14), L4(14), L3(15), L4(15), L3(16), L4(16)]
    stream0_128i_out = (simde__m128i *) &stream0_256i_out[2 * i + 2];
    stream0_128i_out[0] = simde_mm_unpacklo_epi32(xmm0_128, xmm2_128); // 8 LLRs, 2 REs
    stream0_128i_out[1] = simde_mm_unpackhi_epi32(xmm0_128, xmm2_128); // 8 LLRs, 2 REs
    stream0_128i_out[2] = simde_mm_unpacklo_epi32(xmm1_128, xmm3_128); // 8 LLRs, 2 REs
    stream0_128i_out[3] = simde_mm_unpackhi_epi32(xmm1_128, xmm3_128); // 8 LLRs, 2 REs
  }
#endif
}

#ifdef USE_128BIT
static inline void nr_16qam_llr_2layers_core_epi16(simde__m128i y0r,
                                                   simde__m128i y0i,
                                                   simde__m128i y1r,
                                                   simde__m128i y1i,
                                                   simde__m128i ch_mag_des,
                                                   simde__m128i ch_mag_int,
                                                   simde__m128i rho_re,
                                                   simde__m128i rho_im,
                                                   simde__m128i *llr_re0,
                                                   simde__m128i *llr_re1,
                                                   simde__m128i *llr_im0,
                                                   simde__m128i *llr_im1)
{
  const simde__m128i one_over_sqrt_10 = simde_mm_set1_epi16(20724);
  const simde__m128i one_over_sqrt_10_q15 = simde_mm_set1_epi16(10362);
  const simde__m128i three_over_sqrt_10 = simde_mm_set1_epi16(31086);
  const simde__m128i sqrt_10_over_four = simde_mm_set1_epi16(25905);
  const simde__m128i one_over_two_sqrt_10 = simde_mm_set1_epi16(10362);
  const simde__m128i nine_over_two_sqrt_10 = simde_mm_set1_epi16(23315);
  const uint8_t rho_rs_indexes[16] = {4,6,5,7,0,2,1,3,0,2,1,3,4,6,5,7};

  simde__m128i rho_rs[8];
  simde__m128i psi_rs[16];
  simde__m128i psi_is[16];
  simde__m128i a_rs[16];
  simde__m128i a_is[16];
  simde__m128i psi_as[16];
  simde__m128i a_sqs[16];
  simde__m128i y0_s[8];
  simde__m128i bit_mets[16];

  simde__m128i rho_rpi = simde_mm_adds_epi16(rho_re, rho_im);
  simde__m128i rho_rmi = simde_mm_subs_epi16(rho_re, rho_im);
  rho_rs[0] = simde_mm_mulhi_epi16(rho_rpi, one_over_sqrt_10);
  rho_rs[4] = simde_mm_mulhi_epi16(rho_rmi, one_over_sqrt_10);
  rho_rs[3] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_rpi, three_over_sqrt_10), 1);
  rho_rs[7] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_rmi, three_over_sqrt_10), 1);

  simde__m128i xmm4 = simde_mm_mulhi_epi16(rho_re, one_over_sqrt_10);
  simde__m128i xmm5 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_im, three_over_sqrt_10), 1);
  rho_rs[1] = simde_mm_adds_epi16(xmm4, xmm5);
  rho_rs[5] = simde_mm_subs_epi16(xmm4, xmm5);

  simde__m128i xmm6 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_re, three_over_sqrt_10), 1);
  simde__m128i xmm7 = simde_mm_mulhi_epi16(rho_im, one_over_sqrt_10);
  rho_rs[2] = simde_mm_adds_epi16(xmm6, xmm7);
  rho_rs[6] = simde_mm_subs_epi16(xmm6, xmm7);

  for (int j = 0; j < 8; j++)
    psi_rs[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[j], y1r));
  for (int j = 8; j < 16; j++)
    psi_rs[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[(j - 4) & 7], y1r));
  for (int k = 0; k < 16; k += 8) {
    for (int j = k; j < k + 4; j++) {
      psi_is[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[rho_rs_indexes[j]], y1i));
      psi_is[j + 4] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[rho_rs_indexes[j + 4]], y1i));
    }
  }

  simde__m128i y0r_over_sqrt10 = simde_mm_mulhi_epi16(y0r, one_over_sqrt_10);
  simde__m128i y0i_over_sqrt10 = simde_mm_mulhi_epi16(y0i, one_over_sqrt_10);
  simde__m128i y0r_three_over_sqrt10 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0r, three_over_sqrt_10), 1);
  simde__m128i y0i_three_over_sqrt10 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0i, three_over_sqrt_10), 1);

  y0_s[0] = simde_mm_adds_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
  y0_s[4] = simde_mm_subs_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
  y0_s[1] = simde_mm_adds_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[5] = simde_mm_subs_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[2] = simde_mm_adds_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
  y0_s[6] = simde_mm_subs_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
  y0_s[3] = simde_mm_adds_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[7] = simde_mm_subs_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);

  for (int j = 0; j < 16; j++) {
    a_rs[j] = interference_abs_epi16(psi_rs[j], ch_mag_int, one_over_sqrt_10_q15, three_over_sqrt_10);
    a_is[j] = interference_abs_epi16(psi_is[j], ch_mag_int, one_over_sqrt_10_q15, three_over_sqrt_10);
    psi_as[j] = prodsum_psi_a_epi16(psi_rs[j], a_rs[j], psi_is[j], a_is[j]);
    a_sqs[j] = square_a_epi16(a_rs[j], a_is[j], ch_mag_int, sqrt_10_over_four);
  }

  simde__m128i ch_mag_over_10 = simde_mm_mulhi_epi16(ch_mag_des, one_over_two_sqrt_10);
  simde__m128i ch_mag_over_2 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, sqrt_10_over_four), 1);
  simde__m128i ch_mag_9_over_10 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, nine_over_two_sqrt_10), 2);

  for (int j = 0; j < 8; j += 4) {
    bit_mets[j + 0] = simde_mm_subs_epi16(simde_mm_adds_epi16(simde_mm_subs_epi16(psi_as[j + 0], a_sqs[j + 0]), y0_s[j + 0]), ch_mag_over_10);
    bit_mets[j + 1] = simde_mm_subs_epi16(simde_mm_adds_epi16(simde_mm_subs_epi16(psi_as[j + 1], a_sqs[j + 1]), y0_s[j + 1]), ch_mag_over_2);
    bit_mets[j + 2] = simde_mm_subs_epi16(simde_mm_adds_epi16(simde_mm_subs_epi16(psi_as[j + 2], a_sqs[j + 2]), y0_s[j + 2]), ch_mag_over_2);
    bit_mets[j + 3] = simde_mm_subs_epi16(simde_mm_adds_epi16(simde_mm_subs_epi16(psi_as[j + 3], a_sqs[j + 3]), y0_s[j + 3]), ch_mag_9_over_10);
  }
  for (int j = 8; j < 16; j += 4) {
    bit_mets[j + 0] = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_subs_epi16(psi_as[j + 0], a_sqs[j + 0]), y0_s[(j - 4) & 0x07]), ch_mag_over_10);
    bit_mets[j + 1] = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_subs_epi16(psi_as[j + 1], a_sqs[j + 1]), y0_s[(j - 3) & 0x07]), ch_mag_over_2);
    bit_mets[j + 2] = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_subs_epi16(psi_as[j + 2], a_sqs[j + 2]), y0_s[(j - 2) & 0x07]), ch_mag_over_2);
    bit_mets[j + 3] = simde_mm_subs_epi16(simde_mm_subs_epi16(simde_mm_subs_epi16(psi_as[j + 3], a_sqs[j + 3]), y0_s[(j - 1) & 0x07]), ch_mag_9_over_10);
  }

  simde__m128i logmax_num_re0 = max_epi16(bit_mets[8], bit_mets[9], bit_mets[10], bit_mets[11], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
  simde__m128i logmax_den_re0 = max_epi16(bit_mets[0], bit_mets[1], bit_mets[2], bit_mets[3], bit_mets[4], bit_mets[5], bit_mets[6], bit_mets[7]);
  simde__m128i logmax_num_re1 = max_epi16(bit_mets[4], bit_mets[5], bit_mets[6], bit_mets[7], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
  simde__m128i logmax_den_re1 = max_epi16(bit_mets[0], bit_mets[1], bit_mets[3], bit_mets[2], bit_mets[8], bit_mets[9], bit_mets[10], bit_mets[11]);
  simde__m128i logmax_num_im0 = max_epi16(bit_mets[2], bit_mets[3], bit_mets[6], bit_mets[7], bit_mets[10], bit_mets[11], bit_mets[14], bit_mets[15]);
  simde__m128i logmax_den_im0 = max_epi16(bit_mets[0], bit_mets[1], bit_mets[4], bit_mets[5], bit_mets[8], bit_mets[9], bit_mets[12], bit_mets[13]);
  simde__m128i logmax_num_im1 = max_epi16(bit_mets[1], bit_mets[3], bit_mets[5], bit_mets[7], bit_mets[9], bit_mets[11], bit_mets[13], bit_mets[15]);
  simde__m128i logmax_den_im1 = max_epi16(bit_mets[0], bit_mets[2], bit_mets[4], bit_mets[6], bit_mets[8], bit_mets[10], bit_mets[12], bit_mets[14]);

  *llr_re0 = simde_mm_subs_epi16(logmax_den_re0, logmax_num_re0);
  *llr_re1 = simde_mm_subs_epi16(logmax_den_re1, logmax_num_re1);
  *llr_im0 = simde_mm_subs_epi16(logmax_den_im0, logmax_num_im0);
  *llr_im1 = simde_mm_subs_epi16(logmax_den_im1, logmax_num_im1);
}

static inline void nr_16qam_llr_2layers_store_epi16(simde__m128i *stream_out,
                                                    int i,
                                                    simde__m128i llr_re0,
                                                    simde__m128i llr_re1,
                                                    simde__m128i llr_im0,
                                                    simde__m128i llr_im1)
{
  simde__m128i xmm0 = simde_mm_unpacklo_epi16(llr_re0, llr_re1);
  simde__m128i xmm1 = simde_mm_unpackhi_epi16(llr_re0, llr_re1);
  simde__m128i xmm2 = simde_mm_unpacklo_epi16(llr_im0, llr_im1);
  simde__m128i xmm3 = simde_mm_unpackhi_epi16(llr_im0, llr_im1);
  stream_out[2 * i + 0] = simde_mm_unpacklo_epi32(xmm0, xmm2);
  stream_out[2 * i + 1] = simde_mm_unpackhi_epi32(xmm0, xmm2);
  stream_out[2 * i + 2] = simde_mm_unpacklo_epi32(xmm1, xmm3);
  stream_out[2 * i + 3] = simde_mm_unpackhi_epi32(xmm1, xmm3);
}
#else
static inline void nr_16qam_llr_2layers_core_epi16(simde__m256i y0r,
                                                   simde__m256i y0i,
                                                   simde__m256i y1r,
                                                   simde__m256i y1i,
                                                   simde__m256i ch_mag_des,
                                                   simde__m256i ch_mag_int,
                                                   simde__m256i rho_re,
                                                   simde__m256i rho_im,
                                                   simde__m256i *llr_re0,
                                                   simde__m256i *llr_re1,
                                                   simde__m256i *llr_im0,
                                                   simde__m256i *llr_im1)
{
  const simde__m256i one_over_sqrt_10 = simde_mm256_set1_epi16(20724);
  const simde__m256i one_over_sqrt_10_q15 = simde_mm256_set1_epi16(10362);
  const simde__m256i three_over_sqrt_10 = simde_mm256_set1_epi16(31086);
  const simde__m256i sqrt_10_over_four = simde_mm256_set1_epi16(25905);
  const simde__m256i one_over_two_sqrt_10 = simde_mm256_set1_epi16(10362);
  const simde__m256i nine_over_two_sqrt_10 = simde_mm256_set1_epi16(23315);
  const uint8_t rho_rs_indexes[16] = {4,6,5,7,0,2,1,3,0,2,1,3,4,6,5,7};

  simde__m256i rho_rs[8];
  simde__m256i psi_rs[16];
  simde__m256i psi_is[16];
  simde__m256i a_rs[16];
  simde__m256i a_is[16];
  simde__m256i psi_as[16];
  simde__m256i a_sqs[16];
  simde__m256i y0_s[8];
  simde__m256i bit_mets[16];

  simde__m256i rho_rpi = simde_mm256_adds_epi16(rho_re, rho_im);
  simde__m256i rho_rmi = simde_mm256_subs_epi16(rho_re, rho_im);
  rho_rs[0] = simde_mm256_mulhi_epi16(rho_rpi, one_over_sqrt_10);
  rho_rs[4] = simde_mm256_mulhi_epi16(rho_rmi, one_over_sqrt_10);
  rho_rs[3] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_rpi, three_over_sqrt_10), 1);
  rho_rs[7] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_rmi, three_over_sqrt_10), 1);

  simde__m256i xmm4 = simde_mm256_mulhi_epi16(rho_re, one_over_sqrt_10);
  simde__m256i xmm5 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_im, three_over_sqrt_10), 1);
  rho_rs[1] = simde_mm256_adds_epi16(xmm4, xmm5);
  rho_rs[5] = simde_mm256_subs_epi16(xmm4, xmm5);

  simde__m256i xmm6 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_re, three_over_sqrt_10), 1);
  simde__m256i xmm7 = simde_mm256_mulhi_epi16(rho_im, one_over_sqrt_10);
  rho_rs[2] = simde_mm256_adds_epi16(xmm6, xmm7);
  rho_rs[6] = simde_mm256_subs_epi16(xmm6, xmm7);

  for (int j = 0; j < 8; j++)
    psi_rs[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[j], y1r));
  for (int j = 8; j < 16; j++)
    psi_rs[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[(j - 4) & 7], y1r));
  for (int k = 0; k < 16; k += 8) {
    for (int j = k; j < k + 4; j++) {
      psi_is[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[rho_rs_indexes[j]], y1i));
      psi_is[j + 4] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[rho_rs_indexes[j + 4]], y1i));
    }
  }

  simde__m256i y0r_over_sqrt10 = simde_mm256_mulhi_epi16(y0r, one_over_sqrt_10);
  simde__m256i y0i_over_sqrt10 = simde_mm256_mulhi_epi16(y0i, one_over_sqrt_10);
  simde__m256i y0r_three_over_sqrt10 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0r, three_over_sqrt_10), 1);
  simde__m256i y0i_three_over_sqrt10 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0i, three_over_sqrt_10), 1);

  y0_s[0] = simde_mm256_adds_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
  y0_s[4] = simde_mm256_subs_epi16(y0r_over_sqrt10, y0i_over_sqrt10);
  y0_s[1] = simde_mm256_adds_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[5] = simde_mm256_subs_epi16(y0r_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[2] = simde_mm256_adds_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
  y0_s[6] = simde_mm256_subs_epi16(y0r_three_over_sqrt10, y0i_over_sqrt10);
  y0_s[3] = simde_mm256_adds_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);
  y0_s[7] = simde_mm256_subs_epi16(y0r_three_over_sqrt10, y0i_three_over_sqrt10);

  for (int j = 0; j < 16; j++) {
    a_rs[j] = interference_abs_epi16_256(psi_rs[j], ch_mag_int, one_over_sqrt_10_q15, three_over_sqrt_10);
    a_is[j] = interference_abs_epi16_256(psi_is[j], ch_mag_int, one_over_sqrt_10_q15, three_over_sqrt_10);
    psi_as[j] = prodsum_psi_a_epi16_256(psi_rs[j], a_rs[j], psi_is[j], a_is[j]);
    a_sqs[j] = square_a_epi16_256(a_rs[j], a_is[j], ch_mag_int, sqrt_10_over_four);
  }

  simde__m256i ch_mag_over_10 = simde_mm256_mulhi_epi16(ch_mag_des, one_over_two_sqrt_10);
  simde__m256i ch_mag_over_2 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, sqrt_10_over_four), 1);
  simde__m256i ch_mag_9_over_10 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, nine_over_two_sqrt_10), 2);

  for (int j = 0; j < 8; j += 4) {
    bit_mets[j + 0] = simde_mm256_subs_epi16(simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_as[j + 0], a_sqs[j + 0]), y0_s[j + 0]), ch_mag_over_10);
    bit_mets[j + 1] = simde_mm256_subs_epi16(simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_as[j + 1], a_sqs[j + 1]), y0_s[j + 1]), ch_mag_over_2);
    bit_mets[j + 2] = simde_mm256_subs_epi16(simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_as[j + 2], a_sqs[j + 2]), y0_s[j + 2]), ch_mag_over_2);
    bit_mets[j + 3] = simde_mm256_subs_epi16(simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_as[j + 3], a_sqs[j + 3]), y0_s[j + 3]), ch_mag_9_over_10);
  }
  for (int j = 8; j < 16; j += 4) {
    bit_mets[j + 0] = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_as[j + 0], a_sqs[j + 0]), y0_s[(j - 4) & 0x07]), ch_mag_over_10);
    bit_mets[j + 1] = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_as[j + 1], a_sqs[j + 1]), y0_s[(j - 3) & 0x07]), ch_mag_over_2);
    bit_mets[j + 2] = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_as[j + 2], a_sqs[j + 2]), y0_s[(j - 2) & 0x07]), ch_mag_over_2);
    bit_mets[j + 3] = simde_mm256_subs_epi16(simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_as[j + 3], a_sqs[j + 3]), y0_s[(j - 1) & 0x07]), ch_mag_9_over_10);
  }

  simde__m256i logmax_num_re0 = max_epi16_256(bit_mets[8], bit_mets[9], bit_mets[10], bit_mets[11], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
  simde__m256i logmax_den_re0 = max_epi16_256(bit_mets[0], bit_mets[1], bit_mets[2], bit_mets[3], bit_mets[4], bit_mets[5], bit_mets[6], bit_mets[7]);
  simde__m256i logmax_num_re1 = max_epi16_256(bit_mets[4], bit_mets[5], bit_mets[6], bit_mets[7], bit_mets[12], bit_mets[13], bit_mets[14], bit_mets[15]);
  simde__m256i logmax_den_re1 = max_epi16_256(bit_mets[0], bit_mets[1], bit_mets[3], bit_mets[2], bit_mets[8], bit_mets[9], bit_mets[10], bit_mets[11]);
  simde__m256i logmax_num_im0 = max_epi16_256(bit_mets[2], bit_mets[3], bit_mets[6], bit_mets[7], bit_mets[10], bit_mets[11], bit_mets[14], bit_mets[15]);
  simde__m256i logmax_den_im0 = max_epi16_256(bit_mets[0], bit_mets[1], bit_mets[4], bit_mets[5], bit_mets[8], bit_mets[9], bit_mets[12], bit_mets[13]);
  simde__m256i logmax_num_im1 = max_epi16_256(bit_mets[1], bit_mets[3], bit_mets[5], bit_mets[7], bit_mets[9], bit_mets[11], bit_mets[13], bit_mets[15]);
  simde__m256i logmax_den_im1 = max_epi16_256(bit_mets[0], bit_mets[2], bit_mets[4], bit_mets[6], bit_mets[8], bit_mets[10], bit_mets[12], bit_mets[14]);

  *llr_re0 = simde_mm256_subs_epi16(logmax_den_re0, logmax_num_re0);
  *llr_re1 = simde_mm256_subs_epi16(logmax_den_re1, logmax_num_re1);
  *llr_im0 = simde_mm256_subs_epi16(logmax_den_im0, logmax_num_im0);
  *llr_im1 = simde_mm256_subs_epi16(logmax_den_im1, logmax_num_im1);
}

static inline void nr_16qam_llr_2layers_store_epi16(simde__m256i *stream_out,
                                                    int i,
                                                    simde__m256i llr_re0,
                                                    simde__m256i llr_re1,
                                                    simde__m256i llr_im0,
                                                    simde__m256i llr_im1)
{
  simde__m128i *y0r_128 = (simde__m128i *)&llr_re0;
  simde__m128i *y1r_128 = (simde__m128i *)&llr_re1;
  simde__m128i *y0i_128 = (simde__m128i *)&llr_im0;
  simde__m128i *y1i_128 = (simde__m128i *)&llr_im1;
  simde__m128i *stream0_128i_out = (simde__m128i *)&stream_out[2 * i];
  for (int lane = 0; lane < 2; lane++) {
    simde__m128i xmm0_128 = simde_mm_unpacklo_epi16(y0r_128[lane], y1r_128[lane]);
    simde__m128i xmm1_128 = simde_mm_unpackhi_epi16(y0r_128[lane], y1r_128[lane]);
    simde__m128i xmm2_128 = simde_mm_unpacklo_epi16(y0i_128[lane], y1i_128[lane]);
    simde__m128i xmm3_128 = simde_mm_unpackhi_epi16(y0i_128[lane], y1i_128[lane]);
    stream0_128i_out[4 * lane + 0] = simde_mm_unpacklo_epi32(xmm0_128, xmm2_128);
    stream0_128i_out[4 * lane + 1] = simde_mm_unpackhi_epi32(xmm0_128, xmm2_128);
    stream0_128i_out[4 * lane + 2] = simde_mm_unpacklo_epi32(xmm1_128, xmm3_128);
    stream0_128i_out[4 * lane + 3] = simde_mm_unpackhi_epi32(xmm1_128, xmm3_128);
  }
}
#endif

static void nr_16qam_llr_2layers_dual(c16_t *stream0_in,
                                      c16_t *stream1_in,
                                      c16_t *ch_mag0,
                                      c16_t *ch_mag1,
                                      int16_t *stream0_out,
                                      int16_t *stream1_out,
                                      c16_t *rho0,
                                      c16_t *rho1,
                                      uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho0_128 = (simde__m128i *)rho0;
  simde__m128i *rho1_128 = (simde__m128i *)rho1;
  simde__m128i *stream0_128 = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128 = (simde__m128i *)stream1_in;
  simde__m128i *ch_mag0_128 = (simde__m128i *)ch_mag0;
  simde__m128i *ch_mag1_128 = (simde__m128i *)ch_mag1;
  simde__m128i *out0_128 = (simde__m128i *)stream0_out;
  simde__m128i *out1_128 = (simde__m128i *)stream1_out;

  for (int i = 0; i < length >> 2; i += 2) {
    simde__m128i y0r, y0i, y1r, y1i;
    simde__m128i ch0, ch1, dummy;
    simde__m128i rho0r, rho0i, rho1r, rho1i;
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128[i], stream0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128[i], stream1_128[i + 1]);
    oai_mm_separate_real_imag_parts(&ch0, &dummy, ch_mag0_128[i], ch_mag0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&ch1, &dummy, ch_mag1_128[i], ch_mag1_128[i + 1]);
    oai_mm_separate_real_imag_parts(&rho0r, &rho0i, rho0_128[i], rho0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&rho1r, &rho1i, rho1_128[i], rho1_128[i + 1]);

    simde__m128i llr00, llr01, llr02, llr03;
    simde__m128i llr10, llr11, llr12, llr13;
    nr_16qam_llr_2layers_core_epi16(y0r, y0i, y1r, y1i, ch0, ch1, rho0r, rho0i, &llr00, &llr01, &llr02, &llr03);
    nr_16qam_llr_2layers_core_epi16(y1r, y1i, y0r, y0i, ch1, ch0, rho1r, rho1i, &llr10, &llr11, &llr12, &llr13);
    nr_16qam_llr_2layers_store_epi16(out0_128, i, llr00, llr01, llr02, llr03);
    nr_16qam_llr_2layers_store_epi16(out1_128, i, llr10, llr11, llr12, llr13);
  }
#else
  simde__m256i *rho0_256 = (simde__m256i *)rho0;
  simde__m256i *rho1_256 = (simde__m256i *)rho1;
  simde__m256i *stream0_256 = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256 = (simde__m256i *)stream1_in;
  simde__m256i *ch_mag0_256 = (simde__m256i *)ch_mag0;
  simde__m256i *ch_mag1_256 = (simde__m256i *)ch_mag1;
  simde__m256i *out0_256 = (simde__m256i *)stream0_out;
  simde__m256i *out1_256 = (simde__m256i *)stream1_out;

  for (int i = 0; i < length >> 3; i += 2) {
    simde__m256i y0r, y0i, y1r, y1i;
    simde__m256i ch0, ch1, dummy;
    simde__m256i rho0r, rho0i, rho1r, rho1i;
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256[i], stream0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256[i], stream1_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&ch0, &dummy, ch_mag0_256[i], ch_mag0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&ch1, &dummy, ch_mag1_256[i], ch_mag1_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&rho0r, &rho0i, rho0_256[i], rho0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&rho1r, &rho1i, rho1_256[i], rho1_256[i + 1]);

    simde__m256i llr00, llr01, llr02, llr03;
    simde__m256i llr10, llr11, llr12, llr13;
    nr_16qam_llr_2layers_core_epi16(y0r, y0i, y1r, y1i, ch0, ch1, rho0r, rho0i, &llr00, &llr01, &llr02, &llr03);
    nr_16qam_llr_2layers_core_epi16(y1r, y1i, y0r, y0i, ch1, ch0, rho1r, rho1i, &llr10, &llr11, &llr12, &llr13);
    nr_16qam_llr_2layers_store_epi16(out0_256, i, llr00, llr01, llr02, llr03);
    nr_16qam_llr_2layers_store_epi16(out1_256, i, llr10, llr11, llr12, llr13);
  }
#endif
}

/*
 * This function computes the LLRs of stream 0 (s_0) in presence of the interfering stream 1 (s_1) assuming that both symbols are
 * 64QAM. It can be used for both MU-MIMO interference-aware receiver or for SU-MIMO receivers.
 *
 * Input:
 *   stream0_in:  MF filter output for 1st stream, i.e., y0' = h0'*y0
 *   stream1_in:  MF filter output for 2nd stream, i.e., y1' = h1'*y0
 *   ch_mag:      4*h0/sqrt(42), [Re0 Im0 Re1 Im1] s.t. Im0=Re0, Im1=Re1, etc
 *   ch_mag_i:    4*h0/sqrt(42), [Re0 Im0 Re1 Im1] s.t. Im0=Re0, Im1=Re1, etc
 *   rho01:       Channel cross correlation, i.e., rho01 = h0'*h1
 *   length:      Number of resource elements
 *
 * Output:
 *   stream0_out: Output LLRs for 1st stream
 */
void nr_64qam_llr_2layers(c16_t *stream0_in,
                          c16_t *stream1_in,
                          c16_t *ch_mag,
                          c16_t *ch_mag_i,
                          int16_t *stream0_out,
                          c16_t *rho01,
                          uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho01_128i = (simde__m128i *)rho01;
  simde__m128i *stream0_128i_in = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128i_in = (simde__m128i *)stream1_in;
  simde__m128i *ch_mag_128i = (simde__m128i *)ch_mag;
  simde__m128i *ch_mag_128i_i = (simde__m128i *)ch_mag_i;

  simde__m128i ONE_OVER_SQRT_42              = simde_mm_set1_epi16(10112); // round(1/sqrt(42)*2^16)
  simde__m128i THREE_OVER_SQRT_42            = simde_mm_set1_epi16(30337); // round(3/sqrt(42)*2^16)
  simde__m128i FIVE_OVER_SQRT_42             = simde_mm_set1_epi16(25281); // round(5/sqrt(42)*2^15)
  simde__m128i SEVEN_OVER_SQRT_42            = simde_mm_set1_epi16(17697); // round(7/sqrt(42)*2^14) Q2.14
  simde__m128i ONE_OVER_SQRT_2               = simde_mm_set1_epi16(23170); // round(1/sqrt(2)*2^15)
  simde__m128i ONE_OVER_SQRT_2_42            = simde_mm_set1_epi16(3575); // round(1/sqrt(2*42)*2^15)
  simde__m128i THREE_OVER_SQRT_2_42          = simde_mm_set1_epi16(10726); // round(3/sqrt(2*42)*2^15)
  simde__m128i FIVE_OVER_SQRT_2_42           = simde_mm_set1_epi16(17876); // round(5/sqrt(2*42)*2^15)
  simde__m128i SEVEN_OVER_SQRT_2_42          = simde_mm_set1_epi16(25027); // round(7/sqrt(2*42)*2^15)
  simde__m128i FORTYNINE_OVER_FOUR_SQRT_42   = simde_mm_set1_epi16(30969); // round(49/(4*sqrt(42))*2^14), Q2.14
  simde__m128i THIRTYSEVEN_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(23385); // round(37/(4*sqrt(42))*2^14), Q2.14
  simde__m128i TWENTYFIVE_OVER_FOUR_SQRT_42  = simde_mm_set1_epi16(31601); // round(25/(4*sqrt(42))*2^15)
  simde__m128i TWENTYNINE_OVER_FOUR_SQRT_42  = simde_mm_set1_epi16(18329); // round(29/(4*sqrt(42))*2^15), Q2.14
  simde__m128i SEVENTEEN_OVER_FOUR_SQRT_42   = simde_mm_set1_epi16(21489); // round(17/(4*sqrt(42))*2^15)
  simde__m128i NINE_OVER_FOUR_SQRT_42        = simde_mm_set1_epi16(11376); // round(9/(4*sqrt(42))*2^15)
  simde__m128i THIRTEEN_OVER_FOUR_SQRT_42    = simde_mm_set1_epi16(16433); // round(13/(4*sqrt(42))*2^15)
  simde__m128i FIVE_OVER_FOUR_SQRT_42        = simde_mm_set1_epi16(6320); // round(5/(4*sqrt(42))*2^15)
  simde__m128i ONE_OVER_FOUR_SQRT_42         = simde_mm_set1_epi16(1264); // round(1/(4*sqrt(42))*2^15)
  simde__m128i SQRT_42_OVER_FOUR             = simde_mm_set1_epi16(13272); // round(sqrt(42)/4*2^13), Q3.12

  simde__m128i ch_mag_des;
  simde__m128i ch_mag_int;
  simde__m128i y0r_one_over_sqrt_21;
  simde__m128i y0r_three_over_sqrt_21;
  simde__m128i y0r_five_over_sqrt_21;
  simde__m128i y0r_seven_over_sqrt_21;
  simde__m128i y0i_one_over_sqrt_21;
  simde__m128i y0i_three_over_sqrt_21;
  simde__m128i y0i_five_over_sqrt_21;
  simde__m128i y0i_seven_over_sqrt_21;
  simde__m128i ch_mag_int_with_sigma2;
  simde__m128i two_ch_mag_int_with_sigma2;
  simde__m128i three_ch_mag_int_with_sigma2;

  for (int i = 0; i < length >> 2; i += 2) {

    // Get rho
    simde__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
    oai_mm_separate_real_imag_parts(&xmm2, &xmm3, rho01_128i[i], rho01_128i[i + 1]);

    simde__m128i rho_rpi = simde_mm_adds_epi16(xmm2, xmm3); // rho = Re(rho) + Im(rho)
    simde__m128i rho_rmi = simde_mm_subs_epi16(xmm2, xmm3); // rho* = Re(rho) - Im(rho)

    // Compute the different rhos
    simde__m128i rho_rs[32];
    rho_rs[27] = simde_mm_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_42);
    rho_rs[28] = simde_mm_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_42);
    rho_rs[18] = simde_mm_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_42);
    rho_rs[21] = simde_mm_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_42);
    rho_rs[9] = simde_mm_mulhi_epi16(rho_rpi, FIVE_OVER_SQRT_42);
    rho_rs[14] = simde_mm_mulhi_epi16(rho_rmi, FIVE_OVER_SQRT_42);
    rho_rs[0] = simde_mm_mulhi_epi16(rho_rpi, SEVEN_OVER_SQRT_42);
    rho_rs[7] = simde_mm_mulhi_epi16(rho_rmi, SEVEN_OVER_SQRT_42);

    rho_rs[9] = simde_mm_slli_epi16(rho_rs[9], 1);
    rho_rs[14] = simde_mm_slli_epi16(rho_rs[14], 1);
    rho_rs[0] = simde_mm_slli_epi16(rho_rs[0], 2);
    rho_rs[7] = simde_mm_slli_epi16(rho_rs[7], 2);

    xmm4 = simde_mm_mulhi_epi16(xmm2, ONE_OVER_SQRT_42);
    xmm5 = simde_mm_mulhi_epi16(xmm3, ONE_OVER_SQRT_42);
    xmm6 = simde_mm_mulhi_epi16(xmm3, THREE_OVER_SQRT_42);
    xmm7 = simde_mm_mulhi_epi16(xmm3, FIVE_OVER_SQRT_42);
    xmm8 = simde_mm_mulhi_epi16(xmm3, SEVEN_OVER_SQRT_42);
    xmm7 = simde_mm_slli_epi16(xmm7, 1);
    xmm8 = simde_mm_slli_epi16(xmm8, 2);

    rho_rs[26] = simde_mm_adds_epi16(xmm4, xmm6);
    rho_rs[29] = simde_mm_subs_epi16(xmm4, xmm6);
    rho_rs[25] = simde_mm_adds_epi16(xmm4, xmm7);
    rho_rs[30] = simde_mm_subs_epi16(xmm4, xmm7);
    rho_rs[24] = simde_mm_adds_epi16(xmm4, xmm8);
    rho_rs[31] = simde_mm_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm_mulhi_epi16(xmm2, THREE_OVER_SQRT_42);
    rho_rs[19] = simde_mm_adds_epi16(xmm4, xmm5);
    rho_rs[20] = simde_mm_subs_epi16(xmm4, xmm5);
    rho_rs[17] = simde_mm_adds_epi16(xmm4, xmm7);
    rho_rs[22] = simde_mm_subs_epi16(xmm4, xmm7);
    rho_rs[16] = simde_mm_adds_epi16(xmm4, xmm8);
    rho_rs[23] = simde_mm_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm_mulhi_epi16(xmm2, FIVE_OVER_SQRT_42);
    xmm4 = simde_mm_slli_epi16(xmm4, 1);
    rho_rs[11] = simde_mm_adds_epi16(xmm4, xmm5);
    rho_rs[12] = simde_mm_subs_epi16(xmm4, xmm5);
    rho_rs[10] = simde_mm_adds_epi16(xmm4, xmm6);
    rho_rs[13] = simde_mm_subs_epi16(xmm4, xmm6);
    rho_rs[8] = simde_mm_adds_epi16(xmm4, xmm8);
    rho_rs[15] = simde_mm_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm_mulhi_epi16(xmm2, SEVEN_OVER_SQRT_42);
    xmm4 = simde_mm_slli_epi16(xmm4, 2);
    rho_rs[3] = simde_mm_adds_epi16(xmm4, xmm5);
    rho_rs[4] = simde_mm_subs_epi16(xmm4, xmm5);
    rho_rs[2] = simde_mm_adds_epi16(xmm4, xmm6);
    rho_rs[5] = simde_mm_subs_epi16(xmm4, xmm6);
    rho_rs[1] = simde_mm_adds_epi16(xmm4, xmm7);
    rho_rs[6] = simde_mm_subs_epi16(xmm4, xmm7);

    // Rearrange interfering MF output
    simde__m128i y1r, y1i;
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128i_in[i], stream1_128i_in[i + 1]);

    // Psi_r calculation from rho_rpi or rho_rmi
    xmm0 = simde_mm_set1_epi16(0); // ZERO for abs_pi16
    xmm2 = simde_mm_subs_epi16(rho_rs[0], y1r);

    simde__m128i psi_r_s[64];
    for(int j=0; j<32; j++)  // psi_r_s[0~31], rho_rs[0~31]
      psi_r_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[j], y1r));
    for(int j=32; j<64; j++) // psi_r_s[32~64], rho_rs[31~0]
      psi_r_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[63 - j], y1r));

    // simde__m128i psi_i calculation from rho_rpi or rho_rmi
    simde__m128i psi_i_s[64];
    const uint8_t rho_rs_index[32] = {7,15,23,31,24,16,8,0,6,14,22,30,25,17,9,1,5,13,21,29,26,18,10,2,4,12,20,28,27,19,11,3};
    for(int k=0; k<32; k+=8){  // psi_i_s[0~31]
      for(int j=k; j<k+4; j++)
        psi_i_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[rho_rs_index[j]], y1i));
      for(int j=k+4; j<k+8; j++)
        psi_i_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[rho_rs_index[j]], y1i));
    }
    for(int k=32; k<64; k+=8){ // psi_i_s[32~64]
      for(int j=k; j<k+4; j++)
        psi_i_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[rho_rs_index[63-j]], y1i));
      for(int j=k+4; j<k+8; j++)
        psi_i_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[rho_rs_index[63-j]], y1i));
    }

    // Rearrange desired MF output
    simde__m128i y0r, y0i;
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128i_in[i], stream0_128i_in[i + 1]);

    // Rearrange desired channel magnitudes
    // [|h|^2(1),|h|^2(1),|h|^2(2),|h|^2(2),...,,|h|^2(7),|h|^2(7)]*(2/sqrt(10))
    // xmm2 is dummy variable that contains the same values as ch_mag_des
    oai_mm_separate_real_imag_parts(&ch_mag_des, &xmm2, ch_mag_128i[i], ch_mag_128i[i + 1]);

    // Rearrange interfering channel magnitudes
    oai_mm_separate_real_imag_parts(&ch_mag_int, &xmm2, ch_mag_128i_i[i], ch_mag_128i_i[i + 1]);

    y0r_one_over_sqrt_21   = simde_mm_mulhi_epi16(y0r, ONE_OVER_SQRT_42);
    y0r_three_over_sqrt_21 = simde_mm_mulhi_epi16(y0r, THREE_OVER_SQRT_42);
    y0r_five_over_sqrt_21  = simde_mm_mulhi_epi16(y0r, FIVE_OVER_SQRT_42);
    y0r_five_over_sqrt_21  = simde_mm_slli_epi16(y0r_five_over_sqrt_21, 1);
    y0r_seven_over_sqrt_21 = simde_mm_mulhi_epi16(y0r, SEVEN_OVER_SQRT_42);
    y0r_seven_over_sqrt_21 = simde_mm_slli_epi16(y0r_seven_over_sqrt_21, 2); // Q2.14

    y0i_one_over_sqrt_21   = simde_mm_mulhi_epi16(y0i, ONE_OVER_SQRT_42);
    y0i_three_over_sqrt_21 = simde_mm_mulhi_epi16(y0i, THREE_OVER_SQRT_42);
    y0i_five_over_sqrt_21  = simde_mm_mulhi_epi16(y0i, FIVE_OVER_SQRT_42);
    y0i_five_over_sqrt_21  = simde_mm_slli_epi16(y0i_five_over_sqrt_21, 1);
    y0i_seven_over_sqrt_21 = simde_mm_mulhi_epi16(y0i, SEVEN_OVER_SQRT_42);
    y0i_seven_over_sqrt_21 = simde_mm_slli_epi16(y0i_seven_over_sqrt_21, 2); // Q2.14

    simde__m128i y0_s[64];
    const simde__m128i y0r_over_s[8] = {y0r_seven_over_sqrt_21,y0r_five_over_sqrt_21,y0r_three_over_sqrt_21,y0r_one_over_sqrt_21};
    for(int j=0; j<32; j+=8){
      y0_s[j+0] = simde_mm_adds_epi16(y0r_over_s[j>>3], y0i_seven_over_sqrt_21);
      y0_s[j+1] = simde_mm_adds_epi16(y0r_over_s[j>>3], y0i_five_over_sqrt_21);
      y0_s[j+2] = simde_mm_adds_epi16(y0r_over_s[j>>3], y0i_three_over_sqrt_21);
      y0_s[j+3] = simde_mm_adds_epi16(y0r_over_s[j>>3], y0i_one_over_sqrt_21);
      y0_s[j+4] = simde_mm_subs_epi16(y0r_over_s[j>>3], y0i_one_over_sqrt_21);
      y0_s[j+5] = simde_mm_subs_epi16(y0r_over_s[j>>3], y0i_three_over_sqrt_21);
      y0_s[j+6] = simde_mm_subs_epi16(y0r_over_s[j>>3], y0i_five_over_sqrt_21);
      y0_s[j+7] = simde_mm_subs_epi16(y0r_over_s[j>>3], y0i_seven_over_sqrt_21);
    }

    ch_mag_int_with_sigma2 = simde_mm_srai_epi16(ch_mag_int, 1); // *2
    two_ch_mag_int_with_sigma2 = ch_mag_int; // *4
    three_ch_mag_int_with_sigma2 = simde_mm_adds_epi16(ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2); // *6
    simde__m128i a_r_s[64];
    simde__m128i a_i_s[64];
    simde__m128i psi_a_s[64];
    simde__m128i a_sq_s[64];
    for(int j=0; j<64; j++){
      // Detection of interference term
      a_r_s[j] = interference_abs_64qam_epi16(psi_r_s[j], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
      a_i_s[j] = interference_abs_64qam_epi16(psi_i_s[j], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
      
      // Calculation of a group of two terms in the bit metric involving product of psi and interference
      psi_a_s[j] = prodsum_psi_a_epi16(psi_r_s[j], a_r_s[j], psi_i_s[j], a_i_s[j]);

      // Multiply by sqrt(2)
      psi_a_s[j] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(psi_a_s[j], ONE_OVER_SQRT_2), 2);

      // Calculation of a group of two terms in the bit metric involving squares of interference
      a_sq_s[j] = square_a_64qam_epi16(a_r_s[j], a_i_s[j], ch_mag_int, SQRT_42_OVER_FOUR);
    }

    // Computing different multiples of ||h0||^2
    simde__m128i ch_mag_with_sigma2[10];
    enum ch_mag_over_42with_sigma2_vals {mag2=0, mag10, mag26, mag18, mag34, mag58, mag50, mag74, mag98};
    // x=1, y=1
    ch_mag_with_sigma2[mag2] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, ONE_OVER_FOUR_SQRT_42), 1);
    // x=1, y=3
    ch_mag_with_sigma2[mag10] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, FIVE_OVER_FOUR_SQRT_42), 1);
    // x=1, x=5
    ch_mag_with_sigma2[mag26] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, THIRTEEN_OVER_FOUR_SQRT_42), 1);
    // x=1, y=7
    ch_mag_with_sigma2[mag50] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
    // x=3, y=3
    ch_mag_with_sigma2[mag18] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, NINE_OVER_FOUR_SQRT_42), 1);
    // x=3, y=5
    ch_mag_with_sigma2[mag34] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, SEVENTEEN_OVER_FOUR_SQRT_42), 1);
    // x=3, y=7
    ch_mag_with_sigma2[mag58] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYNINE_OVER_FOUR_SQRT_42), 2);
    // x=5, y=5
    ch_mag_with_sigma2[mag50] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
    // x=5, y=7
    ch_mag_with_sigma2[mag74] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, THIRTYSEVEN_OVER_FOUR_SQRT_42), 2);
    // x=7, y=7
    ch_mag_with_sigma2[mag98] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, FORTYNINE_OVER_FOUR_SQRT_42), 2);

    // Computing Metrics
    simde__m128i bit_met_s[64];
    const enum ch_mag_over_42with_sigma2_vals table[] = {
      mag98, mag74, mag58, mag50, mag50, mag58, mag74, mag98, mag74, mag50, mag34, mag26, mag26, mag34, mag50, mag74,
      mag58, mag34, mag18, mag10, mag10, mag18, mag34, mag58, mag50, mag26, mag10, mag2,  mag2,  mag10, mag26, mag50};

    for(int i=0; i<32; i++){
      const simde__m128i x = simde_mm_adds_epi16(simde_mm_subs_epi16(psi_a_s[i], a_sq_s[i]), y0_s[i]);
      bit_met_s[i] = simde_mm_subs_epi16(x, ch_mag_with_sigma2[table[i]]);
    }
    for(int i=0; i<32; i++){
      const simde__m128i x = simde_mm_subs_epi16(simde_mm_subs_epi16(psi_a_s[32 + i], a_sq_s[32 + i]), y0_s[31 - i]);
      bit_met_s[32 + i] = simde_mm_subs_epi16(x, ch_mag_with_sigma2[table[31 - i]]);
    }

    // Detection for bits
    simde__m128i logmax_den_re0;
    simde__m128i logmax_num_re0;
    // Detection for 1st bit
    // bit = 1
    xmm0 = max_epi16(bit_met_s[56],bit_met_s[57],bit_met_s[58],bit_met_s[59],bit_met_s[60],bit_met_s[61],bit_met_s[62],bit_met_s[63]);
    xmm1 = max_epi16(bit_met_s[48],bit_met_s[49],bit_met_s[50],bit_met_s[51],bit_met_s[52],bit_met_s[53],bit_met_s[54],bit_met_s[55]);
    xmm2 = max_epi16(bit_met_s[40],bit_met_s[41],bit_met_s[42],bit_met_s[43],bit_met_s[44],bit_met_s[45],bit_met_s[46],bit_met_s[47]);
    xmm3 = max_epi16(bit_met_s[32],bit_met_s[33],bit_met_s[34],bit_met_s[35],bit_met_s[36],bit_met_s[37],bit_met_s[38],bit_met_s[39]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    // bit = 0
    xmm0 = max_epi16(bit_met_s[0],bit_met_s[1],bit_met_s[2],bit_met_s[3],bit_met_s[4],bit_met_s[5],bit_met_s[6],bit_met_s[7]);
    xmm1 = max_epi16(bit_met_s[8],bit_met_s[9],bit_met_s[10],bit_met_s[11],bit_met_s[12],bit_met_s[13],bit_met_s[14],bit_met_s[15]);
    xmm2 = max_epi16(bit_met_s[16],bit_met_s[17],bit_met_s[18],bit_met_s[19],bit_met_s[20],bit_met_s[21],bit_met_s[22],bit_met_s[23]);
    xmm3 = max_epi16(bit_met_s[24],bit_met_s[25],bit_met_s[26],bit_met_s[27],bit_met_s[28],bit_met_s[29],bit_met_s[30],bit_met_s[31]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    y0r = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 2nd bit
    // bit = 1
    xmm0 = max_epi16(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm1 = max_epi16(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm3 = max_epi16(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    // bit = 0
    xmm0 = max_epi16(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm1 = max_epi16(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    xmm2 = max_epi16(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    xmm3 = max_epi16(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    y1r = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 3rd bit
    xmm0 = max_epi16(bit_met_s[63],bit_met_s[62],bit_met_s[61],bit_met_s[60],bit_met_s[59],bit_met_s[58],bit_met_s[57],bit_met_s[56]);
    xmm1 = max_epi16(bit_met_s[55],bit_met_s[54],bit_met_s[53],bit_met_s[52],bit_met_s[51],bit_met_s[50],bit_met_s[49],bit_met_s[48]);
    xmm2 = max_epi16(bit_met_s[15],bit_met_s[14],bit_met_s[13],bit_met_s[12],bit_met_s[11],bit_met_s[10],bit_met_s[9],bit_met_s[8]);
    xmm3 = max_epi16(bit_met_s[7],bit_met_s[6],bit_met_s[5],bit_met_s[4],bit_met_s[3],bit_met_s[2],bit_met_s[1],bit_met_s[0]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16(bit_met_s[47],bit_met_s[46],bit_met_s[45],bit_met_s[44],bit_met_s[43],bit_met_s[42],bit_met_s[41],bit_met_s[40]);
    xmm1 = max_epi16(bit_met_s[39],bit_met_s[38],bit_met_s[37],bit_met_s[36],bit_met_s[35],bit_met_s[34],bit_met_s[33],bit_met_s[32]);
    xmm2 = max_epi16(bit_met_s[31],bit_met_s[30],bit_met_s[29],bit_met_s[28],bit_met_s[27],bit_met_s[26],bit_met_s[25],bit_met_s[24]);
    xmm3 = max_epi16(bit_met_s[23],bit_met_s[22],bit_met_s[21],bit_met_s[20],bit_met_s[19],bit_met_s[18],bit_met_s[17],bit_met_s[16]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    simde__m128i y2r = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 4th bit
    xmm0 = max_epi16(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    xmm1 = max_epi16(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    xmm2 = max_epi16(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm3 = max_epi16(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm1 = max_epi16(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm3 = max_epi16(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    y0i = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 5th bit
    xmm0 = max_epi16(bit_met_s[63],bit_met_s[62],bit_met_s[61],bit_met_s[60],bit_met_s[59],bit_met_s[58],bit_met_s[57],bit_met_s[56]);
    xmm1 = max_epi16(bit_met_s[39],bit_met_s[38],bit_met_s[37],bit_met_s[36],bit_met_s[35],bit_met_s[34],bit_met_s[33],bit_met_s[32]);
    xmm2 = max_epi16(bit_met_s[31],bit_met_s[30],bit_met_s[29],bit_met_s[28],bit_met_s[27],bit_met_s[26],bit_met_s[25],bit_met_s[24]);
    xmm3 = max_epi16(bit_met_s[7],bit_met_s[6],bit_met_s[5],bit_met_s[4],bit_met_s[3],bit_met_s[2],bit_met_s[1],bit_met_s[0]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16(bit_met_s[55],bit_met_s[54],bit_met_s[53],bit_met_s[52],bit_met_s[51],bit_met_s[50],bit_met_s[49],bit_met_s[48]);
    xmm1 = max_epi16(bit_met_s[47],bit_met_s[46],bit_met_s[45],bit_met_s[44],bit_met_s[43],bit_met_s[42],bit_met_s[41],bit_met_s[40]);
    xmm2 = max_epi16(bit_met_s[23],bit_met_s[22],bit_met_s[21],bit_met_s[20],bit_met_s[19],bit_met_s[18],bit_met_s[17],bit_met_s[16]);
    xmm3 = max_epi16(bit_met_s[15],bit_met_s[14],bit_met_s[13],bit_met_s[12],bit_met_s[11],bit_met_s[10],bit_met_s[9],bit_met_s[8]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    y1i = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 6th bit
    xmm0 = max_epi16(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    xmm1 = max_epi16(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm2 = max_epi16(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm3 = max_epi16(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm1 = max_epi16(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    xmm3 = max_epi16(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

    simde__m128i y2i = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Map to output stream, difficult to do in SIMD since we have 6 16bit LLRs
    for (int re = 0; re < 8; re++) {
      *stream0_out++ = ((short *)&y0r)[re];
      *stream0_out++ = ((short *)&y1r)[re];
      *stream0_out++ = ((short *)&y2r)[re];
      *stream0_out++ = ((short *)&y0i)[re];
      *stream0_out++ = ((short *)&y1i)[re];
      *stream0_out++ = ((short *)&y2i)[re];
    }
  }
#else
  simde__m256i *rho01_256i = (simde__m256i *)rho01;
  simde__m256i *stream0_256i_in = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256i_in = (simde__m256i *)stream1_in;
  simde__m256i *ch_mag_256i = (simde__m256i *)ch_mag;
  simde__m256i *ch_mag_256i_i = (simde__m256i *)ch_mag_i;

  simde__m256i ONE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(10112)); // round(1/sqrt(42)*2^16)
  simde__m256i THREE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(30337)); // round(3/sqrt(42)*2^16)
  simde__m256i FIVE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(25281)); // round(5/sqrt(42)*2^15)
  simde__m256i SEVEN_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(17697)); // round(7/sqrt(42)*2^14) Q2.14
  simde__m256i ONE_OVER_SQRT_2 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(23170)); // round(1/sqrt(2)*2^15)
  simde__m256i ONE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(3575)); // round(1/sqrt(2*42)*2^15)
  simde__m256i THREE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(10726)); // round(3/sqrt(2*42)*2^15)
  simde__m256i FIVE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(17876)); // round(5/sqrt(2*42)*2^15)
  simde__m256i SEVEN_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(25027)); // round(7/sqrt(2*42)*2^15)
  simde__m256i FORTYNINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(30969)); // round(49/(4*sqrt(42))*2^14), Q2.14
  simde__m256i THIRTYSEVEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(23385)); // round(37/(4*sqrt(42))*2^14), Q2.14
  simde__m256i TWENTYFIVE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(31601)); // round(25/(4*sqrt(42))*2^15)
  simde__m256i TWENTYNINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(18329)); // round(29/(4*sqrt(42))*2^15), Q2.14
  simde__m256i SEVENTEEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(21489)); // round(17/(4*sqrt(42))*2^15)
  simde__m256i NINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(11376)); // round(9/(4*sqrt(42))*2^15)
  simde__m256i THIRTEEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(16433)); // round(13/(4*sqrt(42))*2^15)
  simde__m256i FIVE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(6320)); // round(5/(4*sqrt(42))*2^15)
  simde__m256i ONE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(1264)); // round(1/(4*sqrt(42))*2^15)
  simde__m256i SQRT_42_OVER_FOUR = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(13272)); // round(sqrt(42)/4*2^13), Q3.12

  simde__m256i ch_mag_des;
  simde__m256i ch_mag_int;
  simde__m256i y0r_one_over_sqrt_21;
  simde__m256i y0r_three_over_sqrt_21;
  simde__m256i y0r_five_over_sqrt_21;
  simde__m256i y0r_seven_over_sqrt_21;
  simde__m256i y0i_one_over_sqrt_21;
  simde__m256i y0i_three_over_sqrt_21;
  simde__m256i y0i_five_over_sqrt_21;
  simde__m256i y0i_seven_over_sqrt_21;
  simde__m256i ch_mag_int_with_sigma2;
  simde__m256i two_ch_mag_int_with_sigma2;
  simde__m256i three_ch_mag_int_with_sigma2;

  uint32_t len256 = length >> 3;

  for (int i = 0; i < len256; i += 2) {

    // Get rho
    simde__m256i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
    oai_mm256_separate_real_imag_parts(&xmm2, &xmm3, rho01_256i[i], rho01_256i[i + 1]);

    simde__m256i rho_rpi = simde_mm256_adds_epi16(xmm2, xmm3); // rho = Re(rho) + Im(rho)
    simde__m256i rho_rmi = simde_mm256_subs_epi16(xmm2, xmm3); // rho* = Re(rho) - Im(rho)

    // Compute the different rhos
    simde__m256i rho_rs[32];
    rho_rs[27] = simde_mm256_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_42);
    rho_rs[28] = simde_mm256_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_42);
    rho_rs[18] = simde_mm256_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_42);
    rho_rs[21] = simde_mm256_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_42);
    rho_rs[9] = simde_mm256_mulhi_epi16(rho_rpi, FIVE_OVER_SQRT_42);
    rho_rs[14] = simde_mm256_mulhi_epi16(rho_rmi, FIVE_OVER_SQRT_42);
    rho_rs[0] = simde_mm256_mulhi_epi16(rho_rpi, SEVEN_OVER_SQRT_42);
    rho_rs[7] = simde_mm256_mulhi_epi16(rho_rmi, SEVEN_OVER_SQRT_42);

    rho_rs[9] = simde_mm256_slli_epi16(rho_rs[9], 1);
    rho_rs[14] = simde_mm256_slli_epi16(rho_rs[14], 1);
    rho_rs[0] = simde_mm256_slli_epi16(rho_rs[0], 2);
    rho_rs[7] = simde_mm256_slli_epi16(rho_rs[7], 2);

    xmm4 = simde_mm256_mulhi_epi16(xmm2, ONE_OVER_SQRT_42);
    xmm5 = simde_mm256_mulhi_epi16(xmm3, ONE_OVER_SQRT_42);
    xmm6 = simde_mm256_mulhi_epi16(xmm3, THREE_OVER_SQRT_42);
    xmm7 = simde_mm256_mulhi_epi16(xmm3, FIVE_OVER_SQRT_42);
    xmm8 = simde_mm256_mulhi_epi16(xmm3, SEVEN_OVER_SQRT_42);
    xmm7 = simde_mm256_slli_epi16(xmm7, 1);
    xmm8 = simde_mm256_slli_epi16(xmm8, 2);

    rho_rs[26] = simde_mm256_adds_epi16(xmm4, xmm6);
    rho_rs[29] = simde_mm256_subs_epi16(xmm4, xmm6);
    rho_rs[25] = simde_mm256_adds_epi16(xmm4, xmm7);
    rho_rs[30] = simde_mm256_subs_epi16(xmm4, xmm7);
    rho_rs[24] = simde_mm256_adds_epi16(xmm4, xmm8);
    rho_rs[31] = simde_mm256_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm256_mulhi_epi16(xmm2, THREE_OVER_SQRT_42);
    rho_rs[19] = simde_mm256_adds_epi16(xmm4, xmm5);
    rho_rs[20] = simde_mm256_subs_epi16(xmm4, xmm5);
    rho_rs[17] = simde_mm256_adds_epi16(xmm4, xmm7);
    rho_rs[22] = simde_mm256_subs_epi16(xmm4, xmm7);
    rho_rs[16] = simde_mm256_adds_epi16(xmm4, xmm8);
    rho_rs[23] = simde_mm256_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm256_mulhi_epi16(xmm2, FIVE_OVER_SQRT_42);
    xmm4 = simde_mm256_slli_epi16(xmm4, 1);
    rho_rs[11] = simde_mm256_adds_epi16(xmm4, xmm5);
    rho_rs[12] = simde_mm256_subs_epi16(xmm4, xmm5);
    rho_rs[10] = simde_mm256_adds_epi16(xmm4, xmm6);
    rho_rs[13] = simde_mm256_subs_epi16(xmm4, xmm6);
    rho_rs[8] = simde_mm256_adds_epi16(xmm4, xmm8);
    rho_rs[15] = simde_mm256_subs_epi16(xmm4, xmm8);

    xmm4 = simde_mm256_mulhi_epi16(xmm2, SEVEN_OVER_SQRT_42);
    xmm4 = simde_mm256_slli_epi16(xmm4, 2);
    rho_rs[3] = simde_mm256_adds_epi16(xmm4, xmm5);
    rho_rs[4] = simde_mm256_subs_epi16(xmm4, xmm5);
    rho_rs[2] = simde_mm256_adds_epi16(xmm4, xmm6);
    rho_rs[5] = simde_mm256_subs_epi16(xmm4, xmm6);
    rho_rs[1] = simde_mm256_adds_epi16(xmm4, xmm7);
    rho_rs[6] = simde_mm256_subs_epi16(xmm4, xmm7);

    // Rearrange interfering MF output
    simde__m256i y1r, y1i;
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256i_in[i], stream1_256i_in[i + 1]);

    // Psi_r calculation from rho_rpi or rho_rmi
    xmm0 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(0)); // ZERO for abs_pi16
    xmm2 = simde_mm256_subs_epi16(rho_rs[0], y1r);

    simde__m256i psi_r_s[64];
    for(int j=0; j<32; j++)  // psi_r_s[0~31], rho_rs[0~31]
      psi_r_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[j], y1r));
    for(int j=32; j<64; j++) // psi_r_s[32~64], rho_rs[31~0]
      psi_r_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[63 - j], y1r));

    // simde__m256i psi_i calculation from rho_rpi or rho_rmi
    simde__m256i psi_i_s[64];
    const uint8_t rho_rs_index[32] = {7,15,23,31,24,16,8,0,6,14,22,30,25,17,9,1,5,13,21,29,26,18,10,2,4,12,20,28,27,19,11,3};
    for(int k=0; k<32; k+=8){  // psi_i_s[0~31]
      for(int j=k; j<k+4; j++)
        psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[rho_rs_index[j]], y1i));
      for(int j=k+4; j<k+8; j++)
        psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[rho_rs_index[j]], y1i));
    }
    for(int k=32; k<64; k+=8){ // psi_i_s[32~64]
      for(int j=k; j<k+4; j++)
        psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[rho_rs_index[63-j]], y1i));
      for(int j=k+4; j<k+8; j++)
        psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[rho_rs_index[63-j]], y1i));
    }

    // Rearrange desired MF output
    simde__m256i y0r, y0i;
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256i_in[i], stream0_256i_in[i + 1]);

    // Rearrange desired channel magnitudes
    // [|h|^2(1),|h|^2(1),|h|^2(2),|h|^2(2),...,,|h|^2(7),|h|^2(7)]*(2/sqrt(10))
    // xmm2 is dummy variable that contains the same values as ch_mag_des
    oai_mm256_separate_real_imag_parts(&ch_mag_des, &xmm2, ch_mag_256i[i], ch_mag_256i[i + 1]);

    // Rearrange interfering channel magnitudes
    oai_mm256_separate_real_imag_parts(&ch_mag_int, &xmm2, ch_mag_256i_i[i], ch_mag_256i_i[i + 1]);

    y0r_one_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, ONE_OVER_SQRT_42);
    y0r_three_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, THREE_OVER_SQRT_42);
    y0r_five_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, FIVE_OVER_SQRT_42);
    y0r_five_over_sqrt_21 = simde_mm256_slli_epi16(y0r_five_over_sqrt_21, 1);
    y0r_seven_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, SEVEN_OVER_SQRT_42);
    y0r_seven_over_sqrt_21 = simde_mm256_slli_epi16(y0r_seven_over_sqrt_21, 2); // Q2.14

    y0i_one_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, ONE_OVER_SQRT_42);
    y0i_three_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, THREE_OVER_SQRT_42);
    y0i_five_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, FIVE_OVER_SQRT_42);
    y0i_five_over_sqrt_21 = simde_mm256_slli_epi16(y0i_five_over_sqrt_21, 1);
    y0i_seven_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, SEVEN_OVER_SQRT_42);
    y0i_seven_over_sqrt_21 = simde_mm256_slli_epi16(y0i_seven_over_sqrt_21, 2); // Q2.14

    simde__m256i y0_s[64];
    const simde__m256i y0r_over_s[8] = {y0r_seven_over_sqrt_21,y0r_five_over_sqrt_21,y0r_three_over_sqrt_21,y0r_one_over_sqrt_21};
    for(int j=0; j<32; j+=8){
      y0_s[j+0] = simde_mm256_adds_epi16(y0r_over_s[j>>3], y0i_seven_over_sqrt_21);
      y0_s[j+1] = simde_mm256_adds_epi16(y0r_over_s[j>>3], y0i_five_over_sqrt_21);
      y0_s[j+2] = simde_mm256_adds_epi16(y0r_over_s[j>>3], y0i_three_over_sqrt_21);
      y0_s[j+3] = simde_mm256_adds_epi16(y0r_over_s[j>>3], y0i_one_over_sqrt_21);
      y0_s[j+4] = simde_mm256_subs_epi16(y0r_over_s[j>>3], y0i_one_over_sqrt_21);
      y0_s[j+5] = simde_mm256_subs_epi16(y0r_over_s[j>>3], y0i_three_over_sqrt_21);
      y0_s[j+6] = simde_mm256_subs_epi16(y0r_over_s[j>>3], y0i_five_over_sqrt_21);
      y0_s[j+7] = simde_mm256_subs_epi16(y0r_over_s[j>>3], y0i_seven_over_sqrt_21);
    }

    ch_mag_int_with_sigma2 = simde_mm256_srai_epi16(ch_mag_int, 1); // *2
    two_ch_mag_int_with_sigma2 = ch_mag_int; // *4
    three_ch_mag_int_with_sigma2 = simde_mm256_adds_epi16(ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2); // *6
    simde__m256i a_r_s[64];
    simde__m256i a_i_s[64];
    simde__m256i psi_a_s[64];
    simde__m256i a_sq_s[64];
    for(int j=0; j<64; j++){
      // Detection of interference term
      a_r_s[j] = interference_abs_64qam_epi16_256(psi_r_s[j], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
      a_i_s[j] = interference_abs_64qam_epi16_256(psi_i_s[j], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
      
      // Calculation of a group of two terms in the bit metric involving product of psi and interference
      psi_a_s[j] = prodsum_psi_a_epi16_256(psi_r_s[j], a_r_s[j], psi_i_s[j], a_i_s[j]);

      // Multiply by sqrt(2)
      psi_a_s[j] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(psi_a_s[j], ONE_OVER_SQRT_2), 2);

      // Calculation of a group of two terms in the bit metric involving squares of interference
      a_sq_s[j] = square_a_64qam_epi16_256(a_r_s[j], a_i_s[j], ch_mag_int, SQRT_42_OVER_FOUR);
    }

    // Computing different multiples of ||h0||^2
    simde__m256i ch_mag_with_sigma2[10];
    enum ch_mag_over_42with_sigma2_vals {mag2=0, mag10, mag26, mag18, mag34, mag58, mag50, mag74, mag98};
     // x=1, y=1
    ch_mag_with_sigma2[mag2] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, ONE_OVER_FOUR_SQRT_42), 1);
    // x=1, y=3
    ch_mag_with_sigma2[mag10] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, FIVE_OVER_FOUR_SQRT_42), 1);
    // x=1, x=5
    ch_mag_with_sigma2[mag26] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, THIRTEEN_OVER_FOUR_SQRT_42), 1);
    // x=1, y=7
    ch_mag_with_sigma2[mag50] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
    // x=3, y=3
    ch_mag_with_sigma2[mag18] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, NINE_OVER_FOUR_SQRT_42), 1);
    // x=3, y=5
    ch_mag_with_sigma2[mag34] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, SEVENTEEN_OVER_FOUR_SQRT_42), 1);
    // x=3, y=7
    ch_mag_with_sigma2[mag58] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYNINE_OVER_FOUR_SQRT_42), 2);
    // x=5, y=5
    ch_mag_with_sigma2[mag50] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
    // x=5, y=7
    ch_mag_with_sigma2[mag74] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, THIRTYSEVEN_OVER_FOUR_SQRT_42), 2);
    // x=7, y=7
    ch_mag_with_sigma2[mag98] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, FORTYNINE_OVER_FOUR_SQRT_42), 2);

    // Computing Metrics
    simde__m256i bit_met_s[64];
    const enum ch_mag_over_42with_sigma2_vals table[] = {
      mag98, mag74, mag58, mag50, mag50, mag58, mag74, mag98, mag74, mag50, mag34, mag26, mag26, mag34, mag50, mag74,
      mag58, mag34, mag18, mag10, mag10, mag18, mag34, mag58, mag50, mag26, mag10, mag2,  mag2,  mag10, mag26, mag50};

    for(int i=0; i<32; i++){
      const simde__m256i x = simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_a_s[i], a_sq_s[i]), y0_s[i]);
      bit_met_s[i] = simde_mm256_subs_epi16(x, ch_mag_with_sigma2[table[i]]);
    }
    for(int i=0; i<32; i++){
      const simde__m256i x = simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_a_s[32 + i], a_sq_s[32 + i]), y0_s[31 - i]);
      bit_met_s[32 + i] = simde_mm256_subs_epi16(x, ch_mag_with_sigma2[table[31 - i]]);
    }

    // Detection for bits
    simde__m256i logmax_den_re0;
    simde__m256i logmax_num_re0;
    // Detection for 1st bit
    // bit = 1
    xmm0 = max_epi16_256(bit_met_s[56],bit_met_s[57],bit_met_s[58],bit_met_s[59],bit_met_s[60],bit_met_s[61],bit_met_s[62],bit_met_s[63]);
    xmm1 = max_epi16_256(bit_met_s[48],bit_met_s[49],bit_met_s[50],bit_met_s[51],bit_met_s[52],bit_met_s[53],bit_met_s[54],bit_met_s[55]);
    xmm2 = max_epi16_256(bit_met_s[40],bit_met_s[41],bit_met_s[42],bit_met_s[43],bit_met_s[44],bit_met_s[45],bit_met_s[46],bit_met_s[47]);
    xmm3 = max_epi16_256(bit_met_s[32],bit_met_s[33],bit_met_s[34],bit_met_s[35],bit_met_s[36],bit_met_s[37],bit_met_s[38],bit_met_s[39]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    // bit = 0
    xmm0 = max_epi16_256(bit_met_s[0],bit_met_s[1],bit_met_s[2],bit_met_s[3],bit_met_s[4],bit_met_s[5],bit_met_s[6],bit_met_s[7]);
    xmm1 = max_epi16_256(bit_met_s[8],bit_met_s[9],bit_met_s[10],bit_met_s[11],bit_met_s[12],bit_met_s[13],bit_met_s[14],bit_met_s[15]);
    xmm2 = max_epi16_256(bit_met_s[16],bit_met_s[17],bit_met_s[18],bit_met_s[19],bit_met_s[20],bit_met_s[21],bit_met_s[22],bit_met_s[23]);
    xmm3 = max_epi16_256(bit_met_s[24],bit_met_s[25],bit_met_s[26],bit_met_s[27],bit_met_s[28],bit_met_s[29],bit_met_s[30],bit_met_s[31]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    y0r = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 2nd bit
    // bit = 1
    xmm0 = max_epi16_256(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm1 = max_epi16_256(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16_256(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm3 = max_epi16_256(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    // bit = 0
    xmm0 = max_epi16_256(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm1 = max_epi16_256(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    xmm2 = max_epi16_256(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    xmm3 = max_epi16_256(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    y1r = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 3rd bit
    xmm0 = max_epi16_256(bit_met_s[63],bit_met_s[62],bit_met_s[61],bit_met_s[60],bit_met_s[59],bit_met_s[58],bit_met_s[57],bit_met_s[56]);
    xmm1 = max_epi16_256(bit_met_s[55],bit_met_s[54],bit_met_s[53],bit_met_s[52],bit_met_s[51],bit_met_s[50],bit_met_s[49],bit_met_s[48]);
    xmm2 = max_epi16_256(bit_met_s[15],bit_met_s[14],bit_met_s[13],bit_met_s[12],bit_met_s[11],bit_met_s[10],bit_met_s[9],bit_met_s[8]);
    xmm3 = max_epi16_256(bit_met_s[7],bit_met_s[6],bit_met_s[5],bit_met_s[4],bit_met_s[3],bit_met_s[2],bit_met_s[1],bit_met_s[0]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16_256(bit_met_s[47],bit_met_s[46],bit_met_s[45],bit_met_s[44],bit_met_s[43],bit_met_s[42],bit_met_s[41],bit_met_s[40]);
    xmm1 = max_epi16_256(bit_met_s[39],bit_met_s[38],bit_met_s[37],bit_met_s[36],bit_met_s[35],bit_met_s[34],bit_met_s[33],bit_met_s[32]);
    xmm2 = max_epi16_256(bit_met_s[31],bit_met_s[30],bit_met_s[29],bit_met_s[28],bit_met_s[27],bit_met_s[26],bit_met_s[25],bit_met_s[24]);
    xmm3 = max_epi16_256(bit_met_s[23],bit_met_s[22],bit_met_s[21],bit_met_s[20],bit_met_s[19],bit_met_s[18],bit_met_s[17],bit_met_s[16]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    simde__m256i y2r = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 4th bit
    xmm0 = max_epi16_256(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    xmm1 = max_epi16_256(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    xmm2 = max_epi16_256(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm3 = max_epi16_256(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16_256(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm1 = max_epi16_256(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16_256(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm3 = max_epi16_256(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    y0i = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 5th bit
    xmm0 = max_epi16_256(bit_met_s[63],bit_met_s[62],bit_met_s[61],bit_met_s[60],bit_met_s[59],bit_met_s[58],bit_met_s[57],bit_met_s[56]);
    xmm1 = max_epi16_256(bit_met_s[39],bit_met_s[38],bit_met_s[37],bit_met_s[36],bit_met_s[35],bit_met_s[34],bit_met_s[33],bit_met_s[32]);
    xmm2 = max_epi16_256(bit_met_s[31],bit_met_s[30],bit_met_s[29],bit_met_s[28],bit_met_s[27],bit_met_s[26],bit_met_s[25],bit_met_s[24]);
    xmm3 = max_epi16_256(bit_met_s[7],bit_met_s[6],bit_met_s[5],bit_met_s[4],bit_met_s[3],bit_met_s[2],bit_met_s[1],bit_met_s[0]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16_256(bit_met_s[55],bit_met_s[54],bit_met_s[53],bit_met_s[52],bit_met_s[51],bit_met_s[50],bit_met_s[49],bit_met_s[48]);
    xmm1 = max_epi16_256(bit_met_s[47],bit_met_s[46],bit_met_s[45],bit_met_s[44],bit_met_s[43],bit_met_s[42],bit_met_s[41],bit_met_s[40]);
    xmm2 = max_epi16_256(bit_met_s[23],bit_met_s[22],bit_met_s[21],bit_met_s[20],bit_met_s[19],bit_met_s[18],bit_met_s[17],bit_met_s[16]);
    xmm3 = max_epi16_256(bit_met_s[15],bit_met_s[14],bit_met_s[13],bit_met_s[12],bit_met_s[11],bit_met_s[10],bit_met_s[9],bit_met_s[8]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    y1i = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Detection for 6th bit
    xmm0 = max_epi16_256(bit_met_s[0],bit_met_s[8],bit_met_s[16],bit_met_s[24],bit_met_s[32],bit_met_s[40],bit_met_s[48],bit_met_s[56]);
    xmm1 = max_epi16_256(bit_met_s[3],bit_met_s[11],bit_met_s[19],bit_met_s[27],bit_met_s[35],bit_met_s[43],bit_met_s[51],bit_met_s[59]);
    xmm2 = max_epi16_256(bit_met_s[4],bit_met_s[12],bit_met_s[20],bit_met_s[28],bit_met_s[36],bit_met_s[44],bit_met_s[52],bit_met_s[60]);
    xmm3 = max_epi16_256(bit_met_s[7],bit_met_s[15],bit_met_s[23],bit_met_s[31],bit_met_s[39],bit_met_s[47],bit_met_s[55],bit_met_s[63]);
    logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    xmm0 = max_epi16_256(bit_met_s[6],bit_met_s[14],bit_met_s[22],bit_met_s[30],bit_met_s[38],bit_met_s[46],bit_met_s[54],bit_met_s[62]);
    xmm1 = max_epi16_256(bit_met_s[5],bit_met_s[13],bit_met_s[21],bit_met_s[29],bit_met_s[37],bit_met_s[45],bit_met_s[53],bit_met_s[61]);
    xmm2 = max_epi16_256(bit_met_s[2],bit_met_s[10],bit_met_s[18],bit_met_s[26],bit_met_s[34],bit_met_s[42],bit_met_s[50],bit_met_s[58]);
    xmm3 = max_epi16_256(bit_met_s[1],bit_met_s[9],bit_met_s[17],bit_met_s[25],bit_met_s[33],bit_met_s[41],bit_met_s[49],bit_met_s[57]);
    logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

    simde__m256i y2i = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

    // Map to output stream, difficult to do in SIMD since we have 6 16bit LLRs
    for (int re = 0; re < 16; re++) {
      *stream0_out++ = ((short *)&y0r)[re];
      *stream0_out++ = ((short *)&y1r)[re];
      *stream0_out++ = ((short *)&y2r)[re];
      *stream0_out++ = ((short *)&y0i)[re];
      *stream0_out++ = ((short *)&y1i)[re];
      *stream0_out++ = ((short *)&y2i)[re];
    }
  }
#endif
}

#ifdef USE_128BIT
static inline void nr_64qam_llr_2layers_core_epi16(simde__m128i y0r,
                                                   simde__m128i y0i,
                                                   simde__m128i y1r,
                                                   simde__m128i y1i,
                                                   simde__m128i ch_mag_des,
                                                   simde__m128i ch_mag_int,
                                                   simde__m128i rho_re,
                                                   simde__m128i rho_im,
                                                   simde__m128i *llr0,
                                                   simde__m128i *llr1,
                                                   simde__m128i *llr2,
                                                   simde__m128i *llr3,
                                                   simde__m128i *llr4,
                                                   simde__m128i *llr5)
{
  const simde__m128i ONE_OVER_SQRT_42 = simde_mm_set1_epi16(10112);
  const simde__m128i THREE_OVER_SQRT_42 = simde_mm_set1_epi16(30337);
  const simde__m128i FIVE_OVER_SQRT_42 = simde_mm_set1_epi16(25281);
  const simde__m128i SEVEN_OVER_SQRT_42 = simde_mm_set1_epi16(17697);
  const simde__m128i ONE_OVER_SQRT_2 = simde_mm_set1_epi16(23170);
  const simde__m128i ONE_OVER_SQRT_2_42 = simde_mm_set1_epi16(3575);
  const simde__m128i THREE_OVER_SQRT_2_42 = simde_mm_set1_epi16(10726);
  const simde__m128i FIVE_OVER_SQRT_2_42 = simde_mm_set1_epi16(17876);
  const simde__m128i SEVEN_OVER_SQRT_2_42 = simde_mm_set1_epi16(25027);
  const simde__m128i FORTYNINE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(30969);
  const simde__m128i THIRTYSEVEN_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(23385);
  const simde__m128i TWENTYFIVE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(31601);
  const simde__m128i TWENTYNINE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(18329);
  const simde__m128i SEVENTEEN_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(21489);
  const simde__m128i NINE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(11376);
  const simde__m128i THIRTEEN_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(16433);
  const simde__m128i FIVE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(6320);
  const simde__m128i ONE_OVER_FOUR_SQRT_42 = simde_mm_set1_epi16(1264);
  const simde__m128i SQRT_42_OVER_FOUR = simde_mm_set1_epi16(13272);
  const uint8_t rho_rs_index[32] = {7,15,23,31,24,16,8,0,6,14,22,30,25,17,9,1,5,13,21,29,26,18,10,2,4,12,20,28,27,19,11,3};
  const enum {mag2=0, mag10, mag26, mag18, mag34, mag58, mag50, mag74, mag98} dummy_enum = mag2;
  (void)dummy_enum;
  const int table[32] = {
    mag98, mag74, mag58, mag50, mag50, mag58, mag74, mag98, mag74, mag50, mag34, mag26, mag26, mag34, mag50, mag74,
    mag58, mag34, mag18, mag10, mag10, mag18, mag34, mag58, mag50, mag26, mag10, mag2, mag2, mag10, mag26, mag50};

  simde__m128i rho_rs[32], psi_r_s[64], psi_i_s[64], y0_s[64];
  simde__m128i ch_mag_with_sigma2[10];

  simde__m128i rho_rpi = simde_mm_adds_epi16(rho_re, rho_im);
  simde__m128i rho_rmi = simde_mm_subs_epi16(rho_re, rho_im);
  rho_rs[27] = simde_mm_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_42);
  rho_rs[28] = simde_mm_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_42);
  rho_rs[18] = simde_mm_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_42);
  rho_rs[21] = simde_mm_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_42);
  rho_rs[9] = simde_mm_mulhi_epi16(rho_rpi, FIVE_OVER_SQRT_42);
  rho_rs[14] = simde_mm_mulhi_epi16(rho_rmi, FIVE_OVER_SQRT_42);
  rho_rs[0] = simde_mm_mulhi_epi16(rho_rpi, SEVEN_OVER_SQRT_42);
  rho_rs[7] = simde_mm_mulhi_epi16(rho_rmi, SEVEN_OVER_SQRT_42);
  rho_rs[9] = simde_mm_slli_epi16(rho_rs[9], 1);
  rho_rs[14] = simde_mm_slli_epi16(rho_rs[14], 1);
  rho_rs[0] = simde_mm_slli_epi16(rho_rs[0], 2);
  rho_rs[7] = simde_mm_slli_epi16(rho_rs[7], 2);

  simde__m128i xmm4 = simde_mm_mulhi_epi16(rho_re, ONE_OVER_SQRT_42);
  simde__m128i xmm5 = simde_mm_mulhi_epi16(rho_im, ONE_OVER_SQRT_42);
  simde__m128i xmm6 = simde_mm_mulhi_epi16(rho_im, THREE_OVER_SQRT_42);
  simde__m128i xmm7 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_im, FIVE_OVER_SQRT_42), 1);
  simde__m128i xmm8 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_im, SEVEN_OVER_SQRT_42), 2);
  rho_rs[26] = simde_mm_adds_epi16(xmm4, xmm6);
  rho_rs[29] = simde_mm_subs_epi16(xmm4, xmm6);
  rho_rs[25] = simde_mm_adds_epi16(xmm4, xmm7);
  rho_rs[30] = simde_mm_subs_epi16(xmm4, xmm7);
  rho_rs[24] = simde_mm_adds_epi16(xmm4, xmm8);
  rho_rs[31] = simde_mm_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm_mulhi_epi16(rho_re, THREE_OVER_SQRT_42);
  rho_rs[19] = simde_mm_adds_epi16(xmm4, xmm5);
  rho_rs[20] = simde_mm_subs_epi16(xmm4, xmm5);
  rho_rs[17] = simde_mm_adds_epi16(xmm4, xmm7);
  rho_rs[22] = simde_mm_subs_epi16(xmm4, xmm7);
  rho_rs[16] = simde_mm_adds_epi16(xmm4, xmm8);
  rho_rs[23] = simde_mm_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_re, FIVE_OVER_SQRT_42), 1);
  rho_rs[11] = simde_mm_adds_epi16(xmm4, xmm5);
  rho_rs[12] = simde_mm_subs_epi16(xmm4, xmm5);
  rho_rs[10] = simde_mm_adds_epi16(xmm4, xmm6);
  rho_rs[13] = simde_mm_subs_epi16(xmm4, xmm6);
  rho_rs[8] = simde_mm_adds_epi16(xmm4, xmm8);
  rho_rs[15] = simde_mm_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(rho_re, SEVEN_OVER_SQRT_42), 2);
  rho_rs[3] = simde_mm_adds_epi16(xmm4, xmm5);
  rho_rs[4] = simde_mm_subs_epi16(xmm4, xmm5);
  rho_rs[2] = simde_mm_adds_epi16(xmm4, xmm6);
  rho_rs[5] = simde_mm_subs_epi16(xmm4, xmm6);
  rho_rs[1] = simde_mm_adds_epi16(xmm4, xmm7);
  rho_rs[6] = simde_mm_subs_epi16(xmm4, xmm7);

  for (int j = 0; j < 32; j++)
    psi_r_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[j], y1r));
  for (int j = 32; j < 64; j++)
    psi_r_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[63 - j], y1r));
  for (int k = 0; k < 32; k += 8) {
    for (int j = k; j < k + 4; j++)
      psi_i_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[rho_rs_index[j]], y1i));
    for (int j = k + 4; j < k + 8; j++)
      psi_i_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[rho_rs_index[j]], y1i));
  }
  for (int k = 32; k < 64; k += 8) {
    for (int j = k; j < k + 4; j++)
      psi_i_s[j] = simde_mm_abs_epi16(simde_mm_subs_epi16(rho_rs[rho_rs_index[63 - j]], y1i));
    for (int j = k + 4; j < k + 8; j++)
      psi_i_s[j] = simde_mm_abs_epi16(simde_mm_adds_epi16(rho_rs[rho_rs_index[63 - j]], y1i));
  }

  simde__m128i y0r_one_over_sqrt_21 = simde_mm_mulhi_epi16(y0r, ONE_OVER_SQRT_42);
  simde__m128i y0r_three_over_sqrt_21 = simde_mm_mulhi_epi16(y0r, THREE_OVER_SQRT_42);
  simde__m128i y0r_five_over_sqrt_21 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0r, FIVE_OVER_SQRT_42), 1);
  simde__m128i y0r_seven_over_sqrt_21 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0r, SEVEN_OVER_SQRT_42), 2);
  simde__m128i y0i_one_over_sqrt_21 = simde_mm_mulhi_epi16(y0i, ONE_OVER_SQRT_42);
  simde__m128i y0i_three_over_sqrt_21 = simde_mm_mulhi_epi16(y0i, THREE_OVER_SQRT_42);
  simde__m128i y0i_five_over_sqrt_21 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0i, FIVE_OVER_SQRT_42), 1);
  simde__m128i y0i_seven_over_sqrt_21 = simde_mm_slli_epi16(simde_mm_mulhi_epi16(y0i, SEVEN_OVER_SQRT_42), 2);
  const simde__m128i y0r_over_s[4] = {y0r_seven_over_sqrt_21, y0r_five_over_sqrt_21, y0r_three_over_sqrt_21, y0r_one_over_sqrt_21};
  for (int j = 0; j < 32; j += 8) {
    y0_s[j + 0] = simde_mm_adds_epi16(y0r_over_s[j >> 3], y0i_seven_over_sqrt_21);
    y0_s[j + 1] = simde_mm_adds_epi16(y0r_over_s[j >> 3], y0i_five_over_sqrt_21);
    y0_s[j + 2] = simde_mm_adds_epi16(y0r_over_s[j >> 3], y0i_three_over_sqrt_21);
    y0_s[j + 3] = simde_mm_adds_epi16(y0r_over_s[j >> 3], y0i_one_over_sqrt_21);
    y0_s[j + 4] = simde_mm_subs_epi16(y0r_over_s[j >> 3], y0i_one_over_sqrt_21);
    y0_s[j + 5] = simde_mm_subs_epi16(y0r_over_s[j >> 3], y0i_three_over_sqrt_21);
    y0_s[j + 6] = simde_mm_subs_epi16(y0r_over_s[j >> 3], y0i_five_over_sqrt_21);
    y0_s[j + 7] = simde_mm_subs_epi16(y0r_over_s[j >> 3], y0i_seven_over_sqrt_21);
  }

  simde__m128i ch_mag_int_with_sigma2 = simde_mm_srai_epi16(ch_mag_int, 1);
  simde__m128i two_ch_mag_int_with_sigma2 = ch_mag_int;
  simde__m128i three_ch_mag_int_with_sigma2 = simde_mm_adds_epi16(ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2);
  ch_mag_with_sigma2[mag2] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, ONE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag10] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, FIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag26] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, THIRTEEN_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag50] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag18] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, NINE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag34] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, SEVENTEEN_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag58] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYNINE_OVER_FOUR_SQRT_42), 2);
  ch_mag_with_sigma2[mag50] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag74] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, THIRTYSEVEN_OVER_FOUR_SQRT_42), 2);
  ch_mag_with_sigma2[mag98] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(ch_mag_des, FORTYNINE_OVER_FOUR_SQRT_42), 2);

  simde__m128i bit_met_s[64];
  for (int i = 0; i < 32; i++) {
    const simde__m128i a_r = interference_abs_64qam_epi16(psi_r_s[i], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    const simde__m128i a_i = interference_abs_64qam_epi16(psi_i_s[i], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    simde__m128i psi_a = prodsum_psi_a_epi16(psi_r_s[i], a_r, psi_i_s[i], a_i);
    psi_a = simde_mm_slli_epi16(simde_mm_mulhi_epi16(psi_a, ONE_OVER_SQRT_2), 2);
    const simde__m128i a_sq = square_a_64qam_epi16(a_r, a_i, ch_mag_int, SQRT_42_OVER_FOUR);
    const simde__m128i x = simde_mm_adds_epi16(simde_mm_subs_epi16(psi_a, a_sq), y0_s[i]);
    bit_met_s[i] = simde_mm_subs_epi16(x, ch_mag_with_sigma2[table[i]]);
  }
  for (int i = 0; i < 32; i++) {
    const int idx = 32 + i;
    const simde__m128i a_r = interference_abs_64qam_epi16(psi_r_s[idx], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    const simde__m128i a_i = interference_abs_64qam_epi16(psi_i_s[idx], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    simde__m128i psi_a = prodsum_psi_a_epi16(psi_r_s[idx], a_r, psi_i_s[idx], a_i);
    psi_a = simde_mm_slli_epi16(simde_mm_mulhi_epi16(psi_a, ONE_OVER_SQRT_2), 2);
    const simde__m128i a_sq = square_a_64qam_epi16(a_r, a_i, ch_mag_int, SQRT_42_OVER_FOUR);
    const simde__m128i x = simde_mm_subs_epi16(simde_mm_subs_epi16(psi_a, a_sq), y0_s[31 - i]);
    bit_met_s[32 + i] = simde_mm_subs_epi16(x, ch_mag_with_sigma2[table[31 - i]]);
  }

  simde__m128i xmm0, xmm1, xmm2, xmm3;
  simde__m128i logmax_den_re0;
  simde__m128i logmax_num_re0;

  xmm0 = max_epi16(bit_met_s[56], bit_met_s[57], bit_met_s[58], bit_met_s[59], bit_met_s[60], bit_met_s[61], bit_met_s[62], bit_met_s[63]);
  xmm1 = max_epi16(bit_met_s[48], bit_met_s[49], bit_met_s[50], bit_met_s[51], bit_met_s[52], bit_met_s[53], bit_met_s[54], bit_met_s[55]);
  xmm2 = max_epi16(bit_met_s[40], bit_met_s[41], bit_met_s[42], bit_met_s[43], bit_met_s[44], bit_met_s[45], bit_met_s[46], bit_met_s[47]);
  xmm3 = max_epi16(bit_met_s[32], bit_met_s[33], bit_met_s[34], bit_met_s[35], bit_met_s[36], bit_met_s[37], bit_met_s[38], bit_met_s[39]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[0], bit_met_s[1], bit_met_s[2], bit_met_s[3], bit_met_s[4], bit_met_s[5], bit_met_s[6], bit_met_s[7]);
  xmm1 = max_epi16(bit_met_s[8], bit_met_s[9], bit_met_s[10], bit_met_s[11], bit_met_s[12], bit_met_s[13], bit_met_s[14], bit_met_s[15]);
  xmm2 = max_epi16(bit_met_s[16], bit_met_s[17], bit_met_s[18], bit_met_s[19], bit_met_s[20], bit_met_s[21], bit_met_s[22], bit_met_s[23]);
  xmm3 = max_epi16(bit_met_s[24], bit_met_s[25], bit_met_s[26], bit_met_s[27], bit_met_s[28], bit_met_s[29], bit_met_s[30], bit_met_s[31]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr0 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm1 = max_epi16(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm3 = max_epi16(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm1 = max_epi16(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  xmm2 = max_epi16(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  xmm3 = max_epi16(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr1 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16(bit_met_s[63], bit_met_s[62], bit_met_s[61], bit_met_s[60], bit_met_s[59], bit_met_s[58], bit_met_s[57], bit_met_s[56]);
  xmm1 = max_epi16(bit_met_s[55], bit_met_s[54], bit_met_s[53], bit_met_s[52], bit_met_s[51], bit_met_s[50], bit_met_s[49], bit_met_s[48]);
  xmm2 = max_epi16(bit_met_s[15], bit_met_s[14], bit_met_s[13], bit_met_s[12], bit_met_s[11], bit_met_s[10], bit_met_s[9], bit_met_s[8]);
  xmm3 = max_epi16(bit_met_s[7], bit_met_s[6], bit_met_s[5], bit_met_s[4], bit_met_s[3], bit_met_s[2], bit_met_s[1], bit_met_s[0]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[47], bit_met_s[46], bit_met_s[45], bit_met_s[44], bit_met_s[43], bit_met_s[42], bit_met_s[41], bit_met_s[40]);
  xmm1 = max_epi16(bit_met_s[39], bit_met_s[38], bit_met_s[37], bit_met_s[36], bit_met_s[35], bit_met_s[34], bit_met_s[33], bit_met_s[32]);
  xmm2 = max_epi16(bit_met_s[31], bit_met_s[30], bit_met_s[29], bit_met_s[28], bit_met_s[27], bit_met_s[26], bit_met_s[25], bit_met_s[24]);
  xmm3 = max_epi16(bit_met_s[23], bit_met_s[22], bit_met_s[21], bit_met_s[20], bit_met_s[19], bit_met_s[18], bit_met_s[17], bit_met_s[16]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr2 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  xmm1 = max_epi16(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  xmm2 = max_epi16(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm3 = max_epi16(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm1 = max_epi16(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm3 = max_epi16(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr3 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16(bit_met_s[63], bit_met_s[62], bit_met_s[61], bit_met_s[60], bit_met_s[59], bit_met_s[58], bit_met_s[57], bit_met_s[56]);
  xmm1 = max_epi16(bit_met_s[39], bit_met_s[38], bit_met_s[37], bit_met_s[36], bit_met_s[35], bit_met_s[34], bit_met_s[33], bit_met_s[32]);
  xmm2 = max_epi16(bit_met_s[31], bit_met_s[30], bit_met_s[29], bit_met_s[28], bit_met_s[27], bit_met_s[26], bit_met_s[25], bit_met_s[24]);
  xmm3 = max_epi16(bit_met_s[7], bit_met_s[6], bit_met_s[5], bit_met_s[4], bit_met_s[3], bit_met_s[2], bit_met_s[1], bit_met_s[0]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[55], bit_met_s[54], bit_met_s[53], bit_met_s[52], bit_met_s[51], bit_met_s[50], bit_met_s[49], bit_met_s[48]);
  xmm1 = max_epi16(bit_met_s[47], bit_met_s[46], bit_met_s[45], bit_met_s[44], bit_met_s[43], bit_met_s[42], bit_met_s[41], bit_met_s[40]);
  xmm2 = max_epi16(bit_met_s[23], bit_met_s[22], bit_met_s[21], bit_met_s[20], bit_met_s[19], bit_met_s[18], bit_met_s[17], bit_met_s[16]);
  xmm3 = max_epi16(bit_met_s[15], bit_met_s[14], bit_met_s[13], bit_met_s[12], bit_met_s[11], bit_met_s[10], bit_met_s[9], bit_met_s[8]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr4 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  xmm1 = max_epi16(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm2 = max_epi16(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm3 = max_epi16(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm1 = max_epi16(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  xmm3 = max_epi16(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  logmax_num_re0 = simde_mm_max_epi16(simde_mm_max_epi16(xmm0, xmm1), simde_mm_max_epi16(xmm2, xmm3));
  *llr5 = simde_mm_subs_epi16(logmax_num_re0, logmax_den_re0);
}

static inline void nr_64qam_llr_2layers_store_epi16(int16_t *stream_out,
                                                    simde__m128i llr0,
                                                    simde__m128i llr1,
                                                    simde__m128i llr2,
                                                    simde__m128i llr3,
                                                    simde__m128i llr4,
                                                    simde__m128i llr5)
{
  for (int re = 0; re < 8; re++) {
    *stream_out++ = ((short *)&llr0)[re];
    *stream_out++ = ((short *)&llr1)[re];
    *stream_out++ = ((short *)&llr2)[re];
    *stream_out++ = ((short *)&llr3)[re];
    *stream_out++ = ((short *)&llr4)[re];
    *stream_out++ = ((short *)&llr5)[re];
  }
}
#else
static inline void nr_64qam_llr_2layers_core_epi16(simde__m256i y0r,
                                                   simde__m256i y0i,
                                                   simde__m256i y1r,
                                                   simde__m256i y1i,
                                                   simde__m256i ch_mag_des,
                                                   simde__m256i ch_mag_int,
                                                   simde__m256i rho_re,
                                                   simde__m256i rho_im,
                                                   simde__m256i *llr0,
                                                   simde__m256i *llr1,
                                                   simde__m256i *llr2,
                                                   simde__m256i *llr3,
                                                   simde__m256i *llr4,
                                                   simde__m256i *llr5)
{
  const simde__m256i ONE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(10112));
  const simde__m256i THREE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(30337));
  const simde__m256i FIVE_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(25281));
  const simde__m256i SEVEN_OVER_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(17697));
  const simde__m256i ONE_OVER_SQRT_2 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(23170));
  const simde__m256i ONE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(3575));
  const simde__m256i THREE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(10726));
  const simde__m256i FIVE_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(17876));
  const simde__m256i SEVEN_OVER_SQRT_2_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(25027));
  const simde__m256i FORTYNINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(30969));
  const simde__m256i THIRTYSEVEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(23385));
  const simde__m256i TWENTYFIVE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(31601));
  const simde__m256i TWENTYNINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(18329));
  const simde__m256i SEVENTEEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(21489));
  const simde__m256i NINE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(11376));
  const simde__m256i THIRTEEN_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(16433));
  const simde__m256i FIVE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(6320));
  const simde__m256i ONE_OVER_FOUR_SQRT_42 = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(1264));
  const simde__m256i SQRT_42_OVER_FOUR = simde_mm256_broadcastw_epi16(simde_mm_set1_epi16(13272));
  const uint8_t rho_rs_index[32] = {7,15,23,31,24,16,8,0,6,14,22,30,25,17,9,1,5,13,21,29,26,18,10,2,4,12,20,28,27,19,11,3};
  const enum {mag2=0, mag10, mag26, mag18, mag34, mag58, mag50, mag74, mag98} dummy_enum = mag2;
  (void)dummy_enum;
  const int table[32] = {
    mag98, mag74, mag58, mag50, mag50, mag58, mag74, mag98, mag74, mag50, mag34, mag26, mag26, mag34, mag50, mag74,
    mag58, mag34, mag18, mag10, mag10, mag18, mag34, mag58, mag50, mag26, mag10, mag2, mag2, mag10, mag26, mag50};

  simde__m256i rho_rs[32], psi_r_s[64], psi_i_s[64], y0_s[64];
  simde__m256i ch_mag_with_sigma2[10];

  simde__m256i rho_rpi = simde_mm256_adds_epi16(rho_re, rho_im);
  simde__m256i rho_rmi = simde_mm256_subs_epi16(rho_re, rho_im);
  rho_rs[27] = simde_mm256_mulhi_epi16(rho_rpi, ONE_OVER_SQRT_42);
  rho_rs[28] = simde_mm256_mulhi_epi16(rho_rmi, ONE_OVER_SQRT_42);
  rho_rs[18] = simde_mm256_mulhi_epi16(rho_rpi, THREE_OVER_SQRT_42);
  rho_rs[21] = simde_mm256_mulhi_epi16(rho_rmi, THREE_OVER_SQRT_42);
  rho_rs[9] = simde_mm256_mulhi_epi16(rho_rpi, FIVE_OVER_SQRT_42);
  rho_rs[14] = simde_mm256_mulhi_epi16(rho_rmi, FIVE_OVER_SQRT_42);
  rho_rs[0] = simde_mm256_mulhi_epi16(rho_rpi, SEVEN_OVER_SQRT_42);
  rho_rs[7] = simde_mm256_mulhi_epi16(rho_rmi, SEVEN_OVER_SQRT_42);
  rho_rs[9] = simde_mm256_slli_epi16(rho_rs[9], 1);
  rho_rs[14] = simde_mm256_slli_epi16(rho_rs[14], 1);
  rho_rs[0] = simde_mm256_slli_epi16(rho_rs[0], 2);
  rho_rs[7] = simde_mm256_slli_epi16(rho_rs[7], 2);

  simde__m256i xmm4 = simde_mm256_mulhi_epi16(rho_re, ONE_OVER_SQRT_42);
  simde__m256i xmm5 = simde_mm256_mulhi_epi16(rho_im, ONE_OVER_SQRT_42);
  simde__m256i xmm6 = simde_mm256_mulhi_epi16(rho_im, THREE_OVER_SQRT_42);
  simde__m256i xmm7 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_im, FIVE_OVER_SQRT_42), 1);
  simde__m256i xmm8 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_im, SEVEN_OVER_SQRT_42), 2);
  rho_rs[26] = simde_mm256_adds_epi16(xmm4, xmm6);
  rho_rs[29] = simde_mm256_subs_epi16(xmm4, xmm6);
  rho_rs[25] = simde_mm256_adds_epi16(xmm4, xmm7);
  rho_rs[30] = simde_mm256_subs_epi16(xmm4, xmm7);
  rho_rs[24] = simde_mm256_adds_epi16(xmm4, xmm8);
  rho_rs[31] = simde_mm256_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm256_mulhi_epi16(rho_re, THREE_OVER_SQRT_42);
  rho_rs[19] = simde_mm256_adds_epi16(xmm4, xmm5);
  rho_rs[20] = simde_mm256_subs_epi16(xmm4, xmm5);
  rho_rs[17] = simde_mm256_adds_epi16(xmm4, xmm7);
  rho_rs[22] = simde_mm256_subs_epi16(xmm4, xmm7);
  rho_rs[16] = simde_mm256_adds_epi16(xmm4, xmm8);
  rho_rs[23] = simde_mm256_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_re, FIVE_OVER_SQRT_42), 1);
  rho_rs[11] = simde_mm256_adds_epi16(xmm4, xmm5);
  rho_rs[12] = simde_mm256_subs_epi16(xmm4, xmm5);
  rho_rs[10] = simde_mm256_adds_epi16(xmm4, xmm6);
  rho_rs[13] = simde_mm256_subs_epi16(xmm4, xmm6);
  rho_rs[8] = simde_mm256_adds_epi16(xmm4, xmm8);
  rho_rs[15] = simde_mm256_subs_epi16(xmm4, xmm8);

  xmm4 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(rho_re, SEVEN_OVER_SQRT_42), 2);
  rho_rs[3] = simde_mm256_adds_epi16(xmm4, xmm5);
  rho_rs[4] = simde_mm256_subs_epi16(xmm4, xmm5);
  rho_rs[2] = simde_mm256_adds_epi16(xmm4, xmm6);
  rho_rs[5] = simde_mm256_subs_epi16(xmm4, xmm6);
  rho_rs[1] = simde_mm256_adds_epi16(xmm4, xmm7);
  rho_rs[6] = simde_mm256_subs_epi16(xmm4, xmm7);

  for (int j = 0; j < 32; j++)
    psi_r_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[j], y1r));
  for (int j = 32; j < 64; j++)
    psi_r_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[63 - j], y1r));
  for (int k = 0; k < 32; k += 8) {
    for (int j = k; j < k + 4; j++)
      psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[rho_rs_index[j]], y1i));
    for (int j = k + 4; j < k + 8; j++)
      psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[rho_rs_index[j]], y1i));
  }
  for (int k = 32; k < 64; k += 8) {
    for (int j = k; j < k + 4; j++)
      psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_subs_epi16(rho_rs[rho_rs_index[63 - j]], y1i));
    for (int j = k + 4; j < k + 8; j++)
      psi_i_s[j] = simde_mm256_abs_epi16(simde_mm256_adds_epi16(rho_rs[rho_rs_index[63 - j]], y1i));
  }

  simde__m256i y0r_one_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, ONE_OVER_SQRT_42);
  simde__m256i y0r_three_over_sqrt_21 = simde_mm256_mulhi_epi16(y0r, THREE_OVER_SQRT_42);
  simde__m256i y0r_five_over_sqrt_21 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0r, FIVE_OVER_SQRT_42), 1);
  simde__m256i y0r_seven_over_sqrt_21 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0r, SEVEN_OVER_SQRT_42), 2);
  simde__m256i y0i_one_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, ONE_OVER_SQRT_42);
  simde__m256i y0i_three_over_sqrt_21 = simde_mm256_mulhi_epi16(y0i, THREE_OVER_SQRT_42);
  simde__m256i y0i_five_over_sqrt_21 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0i, FIVE_OVER_SQRT_42), 1);
  simde__m256i y0i_seven_over_sqrt_21 = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(y0i, SEVEN_OVER_SQRT_42), 2);
  const simde__m256i y0r_over_s[4] = {y0r_seven_over_sqrt_21, y0r_five_over_sqrt_21, y0r_three_over_sqrt_21, y0r_one_over_sqrt_21};
  for (int j = 0; j < 32; j += 8) {
    y0_s[j + 0] = simde_mm256_adds_epi16(y0r_over_s[j >> 3], y0i_seven_over_sqrt_21);
    y0_s[j + 1] = simde_mm256_adds_epi16(y0r_over_s[j >> 3], y0i_five_over_sqrt_21);
    y0_s[j + 2] = simde_mm256_adds_epi16(y0r_over_s[j >> 3], y0i_three_over_sqrt_21);
    y0_s[j + 3] = simde_mm256_adds_epi16(y0r_over_s[j >> 3], y0i_one_over_sqrt_21);
    y0_s[j + 4] = simde_mm256_subs_epi16(y0r_over_s[j >> 3], y0i_one_over_sqrt_21);
    y0_s[j + 5] = simde_mm256_subs_epi16(y0r_over_s[j >> 3], y0i_three_over_sqrt_21);
    y0_s[j + 6] = simde_mm256_subs_epi16(y0r_over_s[j >> 3], y0i_five_over_sqrt_21);
    y0_s[j + 7] = simde_mm256_subs_epi16(y0r_over_s[j >> 3], y0i_seven_over_sqrt_21);
  }

  simde__m256i ch_mag_int_with_sigma2 = simde_mm256_srai_epi16(ch_mag_int, 1);
  simde__m256i two_ch_mag_int_with_sigma2 = ch_mag_int;
  simde__m256i three_ch_mag_int_with_sigma2 = simde_mm256_adds_epi16(ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2);
  ch_mag_with_sigma2[mag2] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, ONE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag10] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, FIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag26] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, THIRTEEN_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag50] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag18] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, NINE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag34] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, SEVENTEEN_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag58] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYNINE_OVER_FOUR_SQRT_42), 2);
  ch_mag_with_sigma2[mag50] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, TWENTYFIVE_OVER_FOUR_SQRT_42), 1);
  ch_mag_with_sigma2[mag74] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, THIRTYSEVEN_OVER_FOUR_SQRT_42), 2);
  ch_mag_with_sigma2[mag98] = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(ch_mag_des, FORTYNINE_OVER_FOUR_SQRT_42), 2);

  simde__m256i bit_met_s[64];
  for (int i = 0; i < 32; i++) {
    const simde__m256i a_r = interference_abs_64qam_epi16_256(psi_r_s[i], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    const simde__m256i a_i = interference_abs_64qam_epi16_256(psi_i_s[i], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    simde__m256i psi_a = prodsum_psi_a_epi16_256(psi_r_s[i], a_r, psi_i_s[i], a_i);
    psi_a = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(psi_a, ONE_OVER_SQRT_2), 2);
    const simde__m256i a_sq = square_a_64qam_epi16_256(a_r, a_i, ch_mag_int, SQRT_42_OVER_FOUR);
    const simde__m256i x = simde_mm256_adds_epi16(simde_mm256_subs_epi16(psi_a, a_sq), y0_s[i]);
    bit_met_s[i] = simde_mm256_subs_epi16(x, ch_mag_with_sigma2[table[i]]);
  }
  for (int i = 0; i < 32; i++) {
    const int idx = 32 + i;
    const simde__m256i a_r = interference_abs_64qam_epi16_256(psi_r_s[idx], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    const simde__m256i a_i = interference_abs_64qam_epi16_256(psi_i_s[idx], ch_mag_int_with_sigma2, two_ch_mag_int_with_sigma2, three_ch_mag_int_with_sigma2, ONE_OVER_SQRT_2_42, THREE_OVER_SQRT_2_42, FIVE_OVER_SQRT_2_42, SEVEN_OVER_SQRT_2_42);
    simde__m256i psi_a = prodsum_psi_a_epi16_256(psi_r_s[idx], a_r, psi_i_s[idx], a_i);
    psi_a = simde_mm256_slli_epi16(simde_mm256_mulhi_epi16(psi_a, ONE_OVER_SQRT_2), 2);
    const simde__m256i a_sq = square_a_64qam_epi16_256(a_r, a_i, ch_mag_int, SQRT_42_OVER_FOUR);
    const simde__m256i x = simde_mm256_subs_epi16(simde_mm256_subs_epi16(psi_a, a_sq), y0_s[31 - i]);
    bit_met_s[32 + i] = simde_mm256_subs_epi16(x, ch_mag_with_sigma2[table[31 - i]]);
  }

  simde__m256i xmm0, xmm1, xmm2, xmm3;
  simde__m256i logmax_den_re0;
  simde__m256i logmax_num_re0;

  xmm0 = max_epi16_256(bit_met_s[56], bit_met_s[57], bit_met_s[58], bit_met_s[59], bit_met_s[60], bit_met_s[61], bit_met_s[62], bit_met_s[63]);
  xmm1 = max_epi16_256(bit_met_s[48], bit_met_s[49], bit_met_s[50], bit_met_s[51], bit_met_s[52], bit_met_s[53], bit_met_s[54], bit_met_s[55]);
  xmm2 = max_epi16_256(bit_met_s[40], bit_met_s[41], bit_met_s[42], bit_met_s[43], bit_met_s[44], bit_met_s[45], bit_met_s[46], bit_met_s[47]);
  xmm3 = max_epi16_256(bit_met_s[32], bit_met_s[33], bit_met_s[34], bit_met_s[35], bit_met_s[36], bit_met_s[37], bit_met_s[38], bit_met_s[39]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[0], bit_met_s[1], bit_met_s[2], bit_met_s[3], bit_met_s[4], bit_met_s[5], bit_met_s[6], bit_met_s[7]);
  xmm1 = max_epi16_256(bit_met_s[8], bit_met_s[9], bit_met_s[10], bit_met_s[11], bit_met_s[12], bit_met_s[13], bit_met_s[14], bit_met_s[15]);
  xmm2 = max_epi16_256(bit_met_s[16], bit_met_s[17], bit_met_s[18], bit_met_s[19], bit_met_s[20], bit_met_s[21], bit_met_s[22], bit_met_s[23]);
  xmm3 = max_epi16_256(bit_met_s[24], bit_met_s[25], bit_met_s[26], bit_met_s[27], bit_met_s[28], bit_met_s[29], bit_met_s[30], bit_met_s[31]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr0 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16_256(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm1 = max_epi16_256(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16_256(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm3 = max_epi16_256(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm1 = max_epi16_256(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  xmm2 = max_epi16_256(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  xmm3 = max_epi16_256(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr1 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16_256(bit_met_s[63], bit_met_s[62], bit_met_s[61], bit_met_s[60], bit_met_s[59], bit_met_s[58], bit_met_s[57], bit_met_s[56]);
  xmm1 = max_epi16_256(bit_met_s[55], bit_met_s[54], bit_met_s[53], bit_met_s[52], bit_met_s[51], bit_met_s[50], bit_met_s[49], bit_met_s[48]);
  xmm2 = max_epi16_256(bit_met_s[15], bit_met_s[14], bit_met_s[13], bit_met_s[12], bit_met_s[11], bit_met_s[10], bit_met_s[9], bit_met_s[8]);
  xmm3 = max_epi16_256(bit_met_s[7], bit_met_s[6], bit_met_s[5], bit_met_s[4], bit_met_s[3], bit_met_s[2], bit_met_s[1], bit_met_s[0]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[47], bit_met_s[46], bit_met_s[45], bit_met_s[44], bit_met_s[43], bit_met_s[42], bit_met_s[41], bit_met_s[40]);
  xmm1 = max_epi16_256(bit_met_s[39], bit_met_s[38], bit_met_s[37], bit_met_s[36], bit_met_s[35], bit_met_s[34], bit_met_s[33], bit_met_s[32]);
  xmm2 = max_epi16_256(bit_met_s[31], bit_met_s[30], bit_met_s[29], bit_met_s[28], bit_met_s[27], bit_met_s[26], bit_met_s[25], bit_met_s[24]);
  xmm3 = max_epi16_256(bit_met_s[23], bit_met_s[22], bit_met_s[21], bit_met_s[20], bit_met_s[19], bit_met_s[18], bit_met_s[17], bit_met_s[16]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr2 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16_256(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  xmm1 = max_epi16_256(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  xmm2 = max_epi16_256(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm3 = max_epi16_256(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm1 = max_epi16_256(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16_256(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm3 = max_epi16_256(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr3 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16_256(bit_met_s[63], bit_met_s[62], bit_met_s[61], bit_met_s[60], bit_met_s[59], bit_met_s[58], bit_met_s[57], bit_met_s[56]);
  xmm1 = max_epi16_256(bit_met_s[39], bit_met_s[38], bit_met_s[37], bit_met_s[36], bit_met_s[35], bit_met_s[34], bit_met_s[33], bit_met_s[32]);
  xmm2 = max_epi16_256(bit_met_s[31], bit_met_s[30], bit_met_s[29], bit_met_s[28], bit_met_s[27], bit_met_s[26], bit_met_s[25], bit_met_s[24]);
  xmm3 = max_epi16_256(bit_met_s[7], bit_met_s[6], bit_met_s[5], bit_met_s[4], bit_met_s[3], bit_met_s[2], bit_met_s[1], bit_met_s[0]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[55], bit_met_s[54], bit_met_s[53], bit_met_s[52], bit_met_s[51], bit_met_s[50], bit_met_s[49], bit_met_s[48]);
  xmm1 = max_epi16_256(bit_met_s[47], bit_met_s[46], bit_met_s[45], bit_met_s[44], bit_met_s[43], bit_met_s[42], bit_met_s[41], bit_met_s[40]);
  xmm2 = max_epi16_256(bit_met_s[23], bit_met_s[22], bit_met_s[21], bit_met_s[20], bit_met_s[19], bit_met_s[18], bit_met_s[17], bit_met_s[16]);
  xmm3 = max_epi16_256(bit_met_s[15], bit_met_s[14], bit_met_s[13], bit_met_s[12], bit_met_s[11], bit_met_s[10], bit_met_s[9], bit_met_s[8]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr4 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);

  xmm0 = max_epi16_256(bit_met_s[0], bit_met_s[8], bit_met_s[16], bit_met_s[24], bit_met_s[32], bit_met_s[40], bit_met_s[48], bit_met_s[56]);
  xmm1 = max_epi16_256(bit_met_s[3], bit_met_s[11], bit_met_s[19], bit_met_s[27], bit_met_s[35], bit_met_s[43], bit_met_s[51], bit_met_s[59]);
  xmm2 = max_epi16_256(bit_met_s[4], bit_met_s[12], bit_met_s[20], bit_met_s[28], bit_met_s[36], bit_met_s[44], bit_met_s[52], bit_met_s[60]);
  xmm3 = max_epi16_256(bit_met_s[7], bit_met_s[15], bit_met_s[23], bit_met_s[31], bit_met_s[39], bit_met_s[47], bit_met_s[55], bit_met_s[63]);
  logmax_den_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));

  xmm0 = max_epi16_256(bit_met_s[6], bit_met_s[14], bit_met_s[22], bit_met_s[30], bit_met_s[38], bit_met_s[46], bit_met_s[54], bit_met_s[62]);
  xmm1 = max_epi16_256(bit_met_s[5], bit_met_s[13], bit_met_s[21], bit_met_s[29], bit_met_s[37], bit_met_s[45], bit_met_s[53], bit_met_s[61]);
  xmm2 = max_epi16_256(bit_met_s[2], bit_met_s[10], bit_met_s[18], bit_met_s[26], bit_met_s[34], bit_met_s[42], bit_met_s[50], bit_met_s[58]);
  xmm3 = max_epi16_256(bit_met_s[1], bit_met_s[9], bit_met_s[17], bit_met_s[25], bit_met_s[33], bit_met_s[41], bit_met_s[49], bit_met_s[57]);
  logmax_num_re0 = simde_mm256_max_epi16(simde_mm256_max_epi16(xmm0, xmm1), simde_mm256_max_epi16(xmm2, xmm3));
  *llr5 = simde_mm256_subs_epi16(logmax_num_re0, logmax_den_re0);
}

static inline void nr_64qam_llr_2layers_store_epi16(int16_t *stream_out,
                                                    simde__m256i llr0,
                                                    simde__m256i llr1,
                                                    simde__m256i llr2,
                                                    simde__m256i llr3,
                                                    simde__m256i llr4,
                                                    simde__m256i llr5)
{
  for (int re = 0; re < 16; re++) {
    *stream_out++ = ((short *)&llr0)[re];
    *stream_out++ = ((short *)&llr1)[re];
    *stream_out++ = ((short *)&llr2)[re];
    *stream_out++ = ((short *)&llr3)[re];
    *stream_out++ = ((short *)&llr4)[re];
    *stream_out++ = ((short *)&llr5)[re];
  }
}
#endif

static void nr_64qam_llr_2layers_dual(c16_t *stream0_in,
                                      c16_t *stream1_in,
                                      c16_t *ch_mag0,
                                      c16_t *ch_mag1,
                                      int16_t *stream0_out,
                                      int16_t *stream1_out,
                                      c16_t *rho0,
                                      c16_t *rho1,
                                      uint32_t length)
{
#ifdef USE_128BIT
  simde__m128i *rho0_128 = (simde__m128i *)rho0;
  simde__m128i *rho1_128 = (simde__m128i *)rho1;
  simde__m128i *stream0_128 = (simde__m128i *)stream0_in;
  simde__m128i *stream1_128 = (simde__m128i *)stream1_in;
  simde__m128i *ch_mag0_128 = (simde__m128i *)ch_mag0;
  simde__m128i *ch_mag1_128 = (simde__m128i *)ch_mag1;
  int16_t *out0 = stream0_out;
  int16_t *out1 = stream1_out;
  for (int i = 0; i < length >> 2; i += 2) {
    simde__m128i y0r, y0i, y1r, y1i, ch0, ch1, dummy, rho0r, rho0i, rho1r, rho1i;
    oai_mm_separate_real_imag_parts(&y0r, &y0i, stream0_128[i], stream0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&y1r, &y1i, stream1_128[i], stream1_128[i + 1]);
    oai_mm_separate_real_imag_parts(&ch0, &dummy, ch_mag0_128[i], ch_mag0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&ch1, &dummy, ch_mag1_128[i], ch_mag1_128[i + 1]);
    oai_mm_separate_real_imag_parts(&rho0r, &rho0i, rho0_128[i], rho0_128[i + 1]);
    oai_mm_separate_real_imag_parts(&rho1r, &rho1i, rho1_128[i], rho1_128[i + 1]);
    simde__m128i llr00, llr01, llr02, llr03, llr04, llr05;
    simde__m128i llr10, llr11, llr12, llr13, llr14, llr15;
    nr_64qam_llr_2layers_core_epi16(y0r, y0i, y1r, y1i, ch0, ch1, rho0r, rho0i, &llr00, &llr01, &llr02, &llr03, &llr04, &llr05);
    nr_64qam_llr_2layers_core_epi16(y1r, y1i, y0r, y0i, ch1, ch0, rho1r, rho1i, &llr10, &llr11, &llr12, &llr13, &llr14, &llr15);
    nr_64qam_llr_2layers_store_epi16(out0, llr00, llr01, llr02, llr03, llr04, llr05);
    nr_64qam_llr_2layers_store_epi16(out1, llr10, llr11, llr12, llr13, llr14, llr15);
    out0 += 8 * 6;
    out1 += 8 * 6;
  }
#else
  simde__m256i *rho0_256 = (simde__m256i *)rho0;
  simde__m256i *rho1_256 = (simde__m256i *)rho1;
  simde__m256i *stream0_256 = (simde__m256i *)stream0_in;
  simde__m256i *stream1_256 = (simde__m256i *)stream1_in;
  simde__m256i *ch_mag0_256 = (simde__m256i *)ch_mag0;
  simde__m256i *ch_mag1_256 = (simde__m256i *)ch_mag1;
  int16_t *out0 = stream0_out;
  int16_t *out1 = stream1_out;
  uint32_t len256 = length >> 3;
  for (int i = 0; i < len256; i += 2) {
    simde__m256i y0r, y0i, y1r, y1i, ch0, ch1, dummy, rho0r, rho0i, rho1r, rho1i;
    oai_mm256_separate_real_imag_parts(&y0r, &y0i, stream0_256[i], stream0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&y1r, &y1i, stream1_256[i], stream1_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&ch0, &dummy, ch_mag0_256[i], ch_mag0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&ch1, &dummy, ch_mag1_256[i], ch_mag1_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&rho0r, &rho0i, rho0_256[i], rho0_256[i + 1]);
    oai_mm256_separate_real_imag_parts(&rho1r, &rho1i, rho1_256[i], rho1_256[i + 1]);
    simde__m256i llr00, llr01, llr02, llr03, llr04, llr05;
    simde__m256i llr10, llr11, llr12, llr13, llr14, llr15;
    nr_64qam_llr_2layers_core_epi16(y0r, y0i, y1r, y1i, ch0, ch1, rho0r, rho0i, &llr00, &llr01, &llr02, &llr03, &llr04, &llr05);
    nr_64qam_llr_2layers_core_epi16(y1r, y1i, y0r, y0i, ch1, ch0, rho1r, rho1i, &llr10, &llr11, &llr12, &llr13, &llr14, &llr15);
    nr_64qam_llr_2layers_store_epi16(out0, llr00, llr01, llr02, llr03, llr04, llr05);
    nr_64qam_llr_2layers_store_epi16(out1, llr10, llr11, llr12, llr13, llr14, llr15);
    out0 += 16 * 6;
    out1 += 16 * 6;
  }
#endif
}

static void nr_shift_llr_2layers(int16_t *llr_layer0,
                                 int16_t *llr_layer1,
                                 uint32_t nb_re,
                                 uint32_t rxdataF_ext_offset,
                                 uint8_t mod_order,
                                 int shift)
{
  simde__m128i *llr_layers0 = (simde__m128i *)llr_layer0;
  simde__m128i *llr_layers1 = (simde__m128i *)llr_layer1;

  uint8_t mem_offset = ((16 - ((long)llr_layers0)) & 0xF) >> 2;

  if (mem_offset > 0) {
    c16_t *llr_layers0_c16 = (c16_t *)llr_layer0;
    c16_t *llr_layers1_c16 = (c16_t *)llr_layer1;
    for (int i = 0; i < mem_offset; i++) {
      llr_layers0_c16[i] = c16Shift(llr_layers0_c16[i], shift);
      llr_layers1_c16[i] = c16Shift(llr_layers1_c16[i], shift);
    }
    llr_layers0 = (simde__m128i *)&llr_layer0[mem_offset * 2];
    llr_layers1 = (simde__m128i *)&llr_layer1[mem_offset * 2];
  }

  for (int i = 0; i < nb_re >> 2; i++) {
    llr_layers0[i] = simde_mm_srai_epi16(llr_layers0[i], shift);
    llr_layers1[i] = simde_mm_srai_epi16(llr_layers1[i], shift);
  }
}

void nr_compute_ML_llr(uint32_t rxdataF_ext_offset,
                       c16_t *rxdataF_comp0,
                       c16_t *rxdataF_comp1,
                       c16_t *ch_mag0,
                       c16_t *ch_mag1,
                       int16_t *llr_layers0,
                       int16_t *llr_layers1,
                       c16_t *rho0,
                       c16_t *rho1,
                       uint32_t nb_re,
                       uint8_t mod_order)
{
  switch (mod_order) {
    case 2:
      nr_qpsk_llr_2layers_dual(rxdataF_comp0, rxdataF_comp1, llr_layers0, llr_layers1, rho0, rho1, nb_re);
      nr_shift_llr_2layers((int16_t *)llr_layers0, (int16_t *)llr_layers1, nb_re, rxdataF_ext_offset >> 1, 2, 4);
      break;
    case 4:
      nr_16qam_llr_2layers_dual(rxdataF_comp0, rxdataF_comp1, ch_mag0, ch_mag1, llr_layers0, llr_layers1, rho0, rho1, nb_re);
      break;
    case 6:
      nr_64qam_llr_2layers_dual(rxdataF_comp0, rxdataF_comp1, ch_mag0, ch_mag1, llr_layers0, llr_layers1, rho0, rho1, nb_re);
      break;
    default:
      AssertFatal(1 == 0, "nr_compute_llr: invalid Qm value, Qm = %d\n", mod_order);
  }
}

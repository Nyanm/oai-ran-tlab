/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_channel_compensation.h"
#include "PHY/sse_intrin.h"
#include "PHY/TOOLS/tools_defs.h"
#include "common/utils/assertions.h"
#ifdef __aarch64__
#define USE_128BIT
#endif

void nr_channel_compensation(uint32_t buffer_length,
                             int nb_rx_ant,
                             c16_t rxFext[][buffer_length],
                             c16_t chFext[][nb_rx_ant][buffer_length],
                             c16_t rxF_ch_maga[][buffer_length],
                             c16_t rxF_ch_magb[][buffer_length],
                             c16_t rxF_ch_magc[][buffer_length],
                             c16_t **rxComp,
                             int nb_layers,
                             c16_t rho[][nb_layers][buffer_length],
                             int numLoopCnt,
                             int mod_order,
                             uint32_t bufOffset,
                             uint32_t output_shift)
{
  bool bIsModGreaterThanQpsk = mod_order > 2;
  bool bIsModGreaterThan16Qam = mod_order > 4;
  bool bIsModGreaterThan64Qam = mod_order > 6;

#ifdef USE_128BIT
  // 128-bit (AVX/SSE) path: each vector holds 4 complex samples vs 8 for AVX2,
  // so we need twice as many iterations.
  int numLoopCnt128 = numLoopCnt * 2;

  simde__m128i QAM_ampa_128 = simde_mm_setzero_si128();
  simde__m128i QAM_ampb_128 = simde_mm_setzero_si128();
  simde__m128i QAM_ampc_128 = simde_mm_setzero_si128();

  if (mod_order == 4) {
    QAM_ampa_128 = simde_mm_set1_epi16(QAM16_n1);
  } else if (mod_order == 6) {
    QAM_ampa_128 = simde_mm_set1_epi16(QAM64_n1);
    QAM_ampb_128 = simde_mm_set1_epi16(QAM64_n2);
  } else if (mod_order == 8) {
    QAM_ampa_128 = simde_mm_set1_epi16(QAM256_n1);
    QAM_ampb_128 = simde_mm_set1_epi16(QAM256_n2);
    QAM_ampc_128 = simde_mm_set1_epi16(QAM256_n3);
  }

  for (int aatx = 0; aatx < nb_layers; aatx++) {
    simde__m128i *rxComp_128 = (simde__m128i *)&rxComp[aatx * nb_rx_ant][bufOffset];
    simde__m128i *rxF_ch_maga_128 = (simde__m128i *)rxF_ch_maga[aatx];
    simde__m128i *rxF_ch_magb_128 = (simde__m128i *)rxF_ch_magb[aatx];
    simde__m128i *rxF_ch_magc_128 = (simde__m128i *)rxF_ch_magc[aatx];
    for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
      simde__m128i *rxF_128 = (simde__m128i *)rxFext[aarx];
      simde__m128i *chF_128 = (simde__m128i *)chFext[aatx][aarx];

      for (int i = 0; i < numLoopCnt128; i++) {
        // MRC
        simde__m128i comp = oai_mm_cpx_mult_conj(chF_128[i], rxF_128[i], output_shift);
        rxComp_128[i] = simde_mm_add_epi16(rxComp_128[i], comp);

        if (bIsModGreaterThanQpsk) {
          simde__m128i mag = oai_mm_smadd(chF_128[i], chF_128[i], output_shift); // |h|^2
          // pack and duplicate
          mag = simde_mm_packs_epi32(mag, mag);
          mag = simde_mm_unpacklo_epi16(mag, mag);

          rxF_ch_maga_128[i] = simde_mm_add_epi16(rxF_ch_maga_128[i], simde_mm_mulhrs_epi16(mag, QAM_ampa_128));

          if (bIsModGreaterThan16Qam)
            rxF_ch_magb_128[i] = simde_mm_add_epi16(rxF_ch_magb_128[i], simde_mm_mulhrs_epi16(mag, QAM_ampb_128));

          if (bIsModGreaterThan64Qam)
            rxF_ch_magc_128[i] = simde_mm_add_epi16(rxF_ch_magc_128[i], simde_mm_mulhrs_epi16(mag, QAM_ampc_128));
        }
      }
      if ((rho != NULL) && (nb_layers > 1)) {
        for (int atx = 0; atx < nb_layers; atx++) {
          simde__m128i *rho_128 = (simde__m128i *)rho[aatx][atx];
          simde__m128i *chF_128 = (simde__m128i *)chFext[aatx][aarx];
          simde__m128i *chF2_128 = (simde__m128i *)chFext[atx][aarx];
          for (int i = 0; i < numLoopCnt128; i++) {
            rho_128[i] = simde_mm_adds_epi16(rho_128[i], oai_mm_cpx_mult_conj(chF_128[i], chF2_128[i], output_shift));
          }
        }
      }
    }
  }
#else
  // 256-bit (AVX2) path: each vector holds 8 complex samples.
  simde__m256i QAM_ampa_256 = simde_mm256_setzero_si256();
  simde__m256i QAM_ampb_256 = simde_mm256_setzero_si256();
  simde__m256i QAM_ampc_256 = simde_mm256_setzero_si256();

  if (mod_order == 4) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM16_n1);
  } else if (mod_order == 6) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM64_n1);
    QAM_ampb_256 = simde_mm256_set1_epi16(QAM64_n2);
  } else if (mod_order == 8) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM256_n1);
    QAM_ampb_256 = simde_mm256_set1_epi16(QAM256_n2);
    QAM_ampc_256 = simde_mm256_set1_epi16(QAM256_n3);
  }

  for (int aatx = 0; aatx < nb_layers; aatx++) {
    simde__m256i *rxComp_256 = (simde__m256i *)&rxComp[aatx * nb_rx_ant][bufOffset];
    simde__m256i *rxF_ch_maga_256 = (simde__m256i *)rxF_ch_maga[aatx];
    simde__m256i *rxF_ch_magb_256 = (simde__m256i *)rxF_ch_magb[aatx];
    simde__m256i *rxF_ch_magc_256 = (simde__m256i *)rxF_ch_magc[aatx];
    for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
      simde__m256i *rxF_256 = (simde__m256i *)rxFext[aarx];
      simde__m256i *chF_256 = (simde__m256i *)chFext[aatx][aarx];

      for (int i = 0; i < numLoopCnt; i++) {
        // MRC
        simde__m256i comp = oai_mm256_cpx_mult_conj(chF_256[i], rxF_256[i], output_shift);
        rxComp_256[i] = simde_mm256_add_epi16(rxComp_256[i], comp);

        if (bIsModGreaterThanQpsk) {
          simde__m256i mag = oai_mm256_smadd(chF_256[i], chF_256[i], output_shift); // |h|^2
          // pack and duplicate
          mag = simde_mm256_packs_epi32(mag, mag);
          mag = simde_mm256_unpacklo_epi16(mag, mag);

          rxF_ch_maga_256[i] = simde_mm256_add_epi16(rxF_ch_maga_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampa_256));

          if (bIsModGreaterThan16Qam)
            rxF_ch_magb_256[i] = simde_mm256_add_epi16(rxF_ch_magb_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampb_256));

          if (bIsModGreaterThan64Qam)
            rxF_ch_magc_256[i] = simde_mm256_add_epi16(rxF_ch_magc_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampc_256));
        }
      }
      if ((rho != NULL) && (nb_layers > 1)) {
        for (int atx = 0; atx < nb_layers; atx++) {
          simde__m256i *rho_256 = (simde__m256i *)rho[aatx][atx];
          simde__m256i *chF_256 = (simde__m256i *)chFext[aatx][aarx];
          simde__m256i *chF2_256 = (simde__m256i *)chFext[atx][aarx];
          for (int i = 0; i < numLoopCnt; i++) {
            rho_256[i] = simde_mm256_adds_epi16(rho_256[i], oai_mm256_cpx_mult_conj(chF_256[i], chF2_256[i], output_shift));
          }
        }
      }
    }
  }
#endif
}


/* Zero Forcing Rx function: nr_conjch0_mult_ch1()
 *
 *
 * */
// TODO: This function is just a wrapper, can be removed.
static void nr_ulsch_conjch0_mult_ch1(c16_t *ch0, c16_t *ch1, c16_t *ch0conj_ch1, unsigned short nb_rb, unsigned char output_shift0)
{
  //This function is used to compute multiplications in H_hermitian * H matrix
  mult_cpx_conj_vector(ch0, ch1, ch0conj_ch1, 12 * nb_rb, output_shift0);
}

/* Zero Forcing Rx function: nr_construct_HhH_elements()
 *
 *
 * */
static void nr_ulsch_construct_HhH_elements(c16_t *conjch00_ch00,
                                            c16_t *conjch01_ch01,
                                            c16_t *conjch11_ch11,
                                            c16_t *conjch10_ch10, //
                                            c16_t *conjch20_ch20,
                                            c16_t *conjch21_ch21,
                                            c16_t *conjch30_ch30,
                                            c16_t *conjch31_ch31,
                                            c16_t *conjch00_ch01, // 00_01
                                            c16_t *conjch01_ch00, // 01_00
                                            c16_t *conjch10_ch11, // 10_11
                                            c16_t *conjch11_ch10, // 11_10
                                            c16_t *conjch20_ch21,
                                            c16_t *conjch21_ch20,
                                            c16_t *conjch30_ch31,
                                            c16_t *conjch31_ch30,
                                            c16_t *after_mf_00,
                                            c16_t *after_mf_01,
                                            c16_t *after_mf_10,
                                            c16_t *after_mf_11,
                                            unsigned short nb_rb,
                                            unsigned char symbol)
{
  //This function is used to construct the (H_hermitian * H matrix) matrix elements
  simde__m128i *conjch00_ch00_128 = (simde__m128i *)conjch00_ch00;
  simde__m128i *conjch01_ch01_128 = (simde__m128i *)conjch01_ch01;
  simde__m128i *conjch11_ch11_128 = (simde__m128i *)conjch11_ch11;
  simde__m128i *conjch10_ch10_128 = (simde__m128i *)conjch10_ch10;

  simde__m128i *conjch20_ch20_128 = (simde__m128i *)conjch20_ch20;
  simde__m128i *conjch21_ch21_128 = (simde__m128i *)conjch21_ch21;
  simde__m128i *conjch30_ch30_128 = (simde__m128i *)conjch30_ch30;
  simde__m128i *conjch31_ch31_128 = (simde__m128i *)conjch31_ch31;

  simde__m128i *conjch00_ch01_128 = (simde__m128i *)conjch00_ch01;
  simde__m128i *conjch01_ch00_128 = (simde__m128i *)conjch01_ch00;
  simde__m128i *conjch10_ch11_128 = (simde__m128i *)conjch10_ch11;
  simde__m128i *conjch11_ch10_128 = (simde__m128i *)conjch11_ch10;

  simde__m128i *conjch20_ch21_128 = (simde__m128i *)conjch20_ch21;
  simde__m128i *conjch21_ch20_128 = (simde__m128i *)conjch21_ch20;
  simde__m128i *conjch30_ch31_128 = (simde__m128i *)conjch30_ch31;
  simde__m128i *conjch31_ch30_128 = (simde__m128i *)conjch31_ch30;

  simde__m128i *after_mf_00_128 = (simde__m128i *)after_mf_00;
  simde__m128i *after_mf_01_128 = (simde__m128i *)after_mf_01;
  simde__m128i *after_mf_10_128 = (simde__m128i *)after_mf_10;
  simde__m128i *after_mf_11_128 = (simde__m128i *)after_mf_11;

  for (unsigned short rb=0; rb<3*nb_rb; rb++) {

    after_mf_00_128[0] = simde_mm_adds_epi16(conjch00_ch00_128[0], conjch10_ch10_128[0]); //00_00 + 10_10
    if (conjch20_ch20 != NULL) after_mf_00_128[0] = simde_mm_adds_epi16(after_mf_00_128[0], conjch20_ch20_128[0]);
    if (conjch30_ch30 != NULL) after_mf_00_128[0] = simde_mm_adds_epi16(after_mf_00_128[0], conjch30_ch30_128[0]);

    after_mf_11_128[0] = simde_mm_adds_epi16(conjch01_ch01_128[0], conjch11_ch11_128[0]); //01_01 + 11_11
    if (conjch21_ch21 != NULL) after_mf_11_128[0] = simde_mm_adds_epi16(after_mf_11_128[0], conjch21_ch21_128[0]);
    if (conjch31_ch31 != NULL) after_mf_11_128[0] = simde_mm_adds_epi16(after_mf_11_128[0], conjch31_ch31_128[0]);

    after_mf_01_128[0] = simde_mm_adds_epi16(conjch00_ch01_128[0], conjch10_ch11_128[0]); //00_01 + 10_11
    if (conjch20_ch21 != NULL) after_mf_01_128[0] = simde_mm_adds_epi16(after_mf_01_128[0], conjch20_ch21_128[0]);
    if (conjch30_ch31 != NULL) after_mf_01_128[0] = simde_mm_adds_epi16(after_mf_01_128[0], conjch30_ch31_128[0]);

    after_mf_10_128[0] = simde_mm_adds_epi16(conjch01_ch00_128[0], conjch11_ch10_128[0]); //01_00 + 11_10
    if (conjch21_ch20 != NULL) after_mf_10_128[0] = simde_mm_adds_epi16(after_mf_10_128[0], conjch21_ch20_128[0]);
    if (conjch31_ch30 != NULL) after_mf_10_128[0] = simde_mm_adds_epi16(after_mf_10_128[0], conjch31_ch30_128[0]);

#ifdef DEBUG_DLSCH_DEMOD
    if ((rb<=30))
    {
      printf(" \n construct_HhH_elements \n");
      print_shorts("after_mf_00_128:",(int16_t*)&after_mf_00_128[0]);
      print_shorts("after_mf_01_128:",(int16_t*)&after_mf_01_128[0]);
      print_shorts("after_mf_10_128:",(int16_t*)&after_mf_10_128[0]);
      print_shorts("after_mf_11_128:",(int16_t*)&after_mf_11_128[0]);
    }
#endif
    conjch00_ch00_128+=1;
    conjch10_ch10_128+=1;
    conjch01_ch01_128+=1;
    conjch11_ch11_128+=1;

    if (conjch20_ch20 != NULL) conjch20_ch20_128+=1;
    if (conjch21_ch21 != NULL) conjch21_ch21_128+=1;
    if (conjch30_ch30 != NULL) conjch30_ch30_128+=1;
    if (conjch31_ch31 != NULL) conjch31_ch31_128+=1;

    conjch00_ch01_128+=1;
    conjch01_ch00_128+=1;
    conjch10_ch11_128+=1;
    conjch11_ch10_128+=1;

    if (conjch20_ch21 != NULL) conjch20_ch21_128+=1;
    if (conjch21_ch20 != NULL) conjch21_ch20_128+=1;
    if (conjch30_ch31 != NULL) conjch30_ch31_128+=1;
    if (conjch31_ch30 != NULL) conjch31_ch30_128+=1;

    after_mf_00_128 += 1;
    after_mf_01_128 += 1;
    after_mf_10_128 += 1;
    after_mf_11_128 += 1;
  }
}

// Zero Forcing Rx function: nr_det_HhH()
static void nr_ulsch_det_HhH(c16_t *after_mf_00, // a
                             c16_t *after_mf_01, // b
                             c16_t *after_mf_10, // c
                             c16_t *after_mf_11, // d
                             uint32_t *det_fin, // 1/ad-bc
                             unsigned short nb_rb,
                             unsigned char symbol,
                             int32_t shift)
{
  simde__m128i *after_mf_00_128,*after_mf_01_128, *after_mf_10_128, *after_mf_11_128, ad_re_128, bc_re_128; //ad_im_128, bc_im_128;
  simde__m128i *det_fin_128, det_re_128; //det_im_128, tmp_det0, tmp_det1;

  after_mf_00_128 = (simde__m128i *)after_mf_00;
  after_mf_01_128 = (simde__m128i *)after_mf_01;
  after_mf_10_128 = (simde__m128i *)after_mf_10;
  after_mf_11_128 = (simde__m128i *)after_mf_11;

  det_fin_128 = (simde__m128i *)det_fin;

  for (unsigned short rb=0; rb<3*nb_rb; rb++) {

    //complex multiplication (I_a+jQ_a)(I_d+jQ_d) = (I_aI_d - Q_aQ_d) + j(Q_aI_d + I_aQ_d)
    //The imag part is often zero, we compute only the real part
    ad_re_128 = simde_mm_madd_epi16(oai_mm_conj(after_mf_00_128[0]),after_mf_11_128[0]); //Re: I_a0*I_d0 - Q_a1*Q_d1
    //ad_im_128 = simde_mm_madd_epi16(oai_mm_swap(after_mf_00_128[0]),after_mf_11_128[0]);//Im: (Q_aI_d + I_aQ_d)

    //complex multiplication (I_b+jQ_b)(I_c+jQ_c) = (I_bI_c - Q_bQ_c) + j(Q_bI_c + I_bQ_c)
    //The imag part is often zero, we compute only the real part
    bc_re_128 = simde_mm_madd_epi16(oai_mm_conj(after_mf_01_128[0]),after_mf_10_128[0]); //Re: I_b0*I_c0 - Q_b1*Q_c1
    //bc_im_128 = simde_mm_madd_epi16(oai_mm_swap(after_mf_01_128[0]),after_mf_10_128[0]);//Im: (Q_bI_c + I_bQ_c)

    det_re_128 = simde_mm_sub_epi32(ad_re_128, bc_re_128);
    //det_im_128 = simde_mm_sub_epi32(ad_im_128, bc_im_128);

    //det in Q30 format
    det_fin_128[0] = simde_mm_abs_epi32(det_re_128);


#ifdef DEBUG_DLSCH_DEMOD
     printf("\n Computing det_HhH_inv \n");
     //print_ints("det_re_128:",(int32_t*)&det_re_128);
     //print_ints("det_im_128:",(int32_t*)&det_im_128);
     print_ints("det_fin_128:",(int32_t*)&det_fin_128[0]);
#endif
    det_fin_128+=1;
    after_mf_00_128+=1;
    after_mf_01_128+=1;
    after_mf_10_128+=1;
    after_mf_11_128+=1;
  }
}

static simde__m128i nr_ulsch_comp_muli_sum(simde__m128i input_x,
                                           simde__m128i input_y,
                                           simde__m128i input_w,
                                           simde__m128i input_z,
                                           simde__m128i det)
{

  // complex multiplication (x_re + jx_im)*(y_re + jy_im) = (x_re*y_re - x_im*y_im) + j(x_im*y_re + x_re*y_im)
  // complex multiplication (w_re + jw_im)*(z_re + jz_im) = (w_re*z_re - w_im*z_im) + j(w_im*z_re + w_re*z_im)
  // the real part
  simde__m128i xy_re_128 = simde_mm_madd_epi16(oai_mm_conj(input_x), input_y); //Re: (x_re*y_re - x_im*y_im)
  simde__m128i wz_re_128 = simde_mm_madd_epi16(oai_mm_conj(input_w), input_z); //Re: (w_re*z_re - w_im*z_im)
  xy_re_128 = simde_mm_sub_epi32(xy_re_128, wz_re_128);

  // the imag part
  simde__m128i xy_im_128 = simde_mm_madd_epi16(oai_mm_swap(input_x), input_y); //Im: (x_im*y_re + x_re*y_im)
  simde__m128i wz_im_128 = simde_mm_madd_epi16(oai_mm_swap(input_w), input_z); //Im: (w_im*z_re + w_re*z_im)
  xy_im_128 = simde_mm_sub_epi32(xy_im_128, wz_im_128);

  //print_ints("rx_re:",(int32_t*)&xy_re_128[0]);
  //print_ints("rx_Img:",(int32_t*)&xy_im_128[0]);
  //divide by matrix det and convert back to Q15 before packing
  uint64_t sum_det = 0;
  for (int k = 0; k < 4; k++) {
    sum_det += (((uint32_t *)&det)[k]);
  }
  // Add bias to reduce rounding error
  sum_det = (sum_det + 2) >> 2;

  int b = log2_approx(sum_det) - 8;
  if (b > 0) {
    xy_re_128 = simde_mm_srai_epi32(xy_re_128, b);
    xy_im_128 = simde_mm_srai_epi32(xy_im_128, b);
  } else {
    xy_re_128 = simde_mm_slli_epi32(xy_re_128, -b);
    xy_im_128 = simde_mm_slli_epi32(xy_im_128, -b);
  }

  simde__m128i output = oai_mm_pack(xy_re_128, xy_im_128);

  return(output);
}


// MMSE Rx function: nr_mmse_2layers()
uint8_t nr_mmse_2layers(const c16_t** rxdataF_comp,
                        uint32_t buffer_length,
                        int nb_rx_ant,
                        c16_t ul_ch_mag[][buffer_length],
                        c16_t ul_ch_magb[][buffer_length],
                        c16_t ul_ch_magc[][buffer_length],
                        c16_t ul_ch_estimates_ext[][nb_rx_ant][buffer_length],
                        unsigned short nb_rb,
                        unsigned char mod_order,
                        int shift,
                        uint32_t bufOffset,
                        unsigned char symbol,
                        int length,
                        uint32_t noise_var)
{
  uint32_t nb_rb_0 = length/12 + ((length%12)?1:0);

  /* we need at least alignment to 16 bytes, let's put 32 to be sure
   * (maybe not necessary but doesn't hurt)
   */
  c16_t conjch00_ch01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch01_ch00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch10_ch11[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch11_ch10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch00_ch00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch01_ch01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch10_ch10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch11_ch11[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch20_ch20[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch21_ch21[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch30_ch30[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch31_ch31[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch20_ch21[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch30_ch31[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch21_ch20[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch31_ch30[12 * nb_rb] __attribute__((aligned(32)));

  c16_t af_mf_00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_11[12 * nb_rb] __attribute__((aligned(32)));
  uint32_t determ_fin[12*nb_rb] __attribute__((aligned(32)));

  c16_t *ch00, *ch01, *ch10, *ch11;
  c16_t *ch20, *ch30, *ch21, *ch31;
  switch (nb_rx_ant) {
    case 2://
      ch00 = ul_ch_estimates_ext[0][0];
      ch01 = ul_ch_estimates_ext[1][0];
      ch10 = ul_ch_estimates_ext[0][1];
      ch11 = ul_ch_estimates_ext[1][1];
      ch20 = NULL;
      ch21 = NULL;
      ch30 = NULL;
      ch31 = NULL;
      break;

    case 4://
      ch00 = ul_ch_estimates_ext[0][0];
      ch01 = ul_ch_estimates_ext[1][0];
      ch10 = ul_ch_estimates_ext[0][1];
      ch11 = ul_ch_estimates_ext[1][1];
      ch20 = ul_ch_estimates_ext[0][2];
      ch21 = ul_ch_estimates_ext[1][2];
      ch30 = ul_ch_estimates_ext[0][3];
      ch31 = ul_ch_estimates_ext[1][3];
      break;

    default:
      return -1;
      break;
  }

  /* 1- Compute the rx channel matrix after compensation: (1/2^log2_max)x(H_herm x H)
   * for n_rx = 2
   * |conj_H_00       conj_H_10|    | H_00         H_01|   |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
   * |                         |  x |                  | = |                                                                 |
   * |conj_H_01       conj_H_11|    | H_10         H_11|   |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
   *
   */

  if (nb_rx_ant >= 2) {
    // (1/2^log2_maxh)*conj_H_00xH_00: (1/(64*2))conjH_00*H_00*2^15
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch00,
                        conjch00_ch00,
                        nb_rb_0,
                        shift);
    // (1/2^log2_maxh)*conj_H_10xH_10: (1/(64*2))conjH_10*H_10*2^15
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch10,
                        conjch10_ch10,
                        nb_rb_0,
                        shift);
    // conj_H_00xH_01
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch01,
                        conjch00_ch01,
                        nb_rb_0,
                        shift); // this shift is equal to the channel level log2_maxh
    // conj_H_10xH_11
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch11,
                        conjch10_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_01
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch01,
                        conjch01_ch01,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_11
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch11,
                        conjch11_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_00
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch00,
                        conjch01_ch00,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_10
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch10,
                        conjch11_ch10,
                        nb_rb_0,
                        shift);
  }
  if (nb_rx_ant == 4) {
    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2*16))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch20,
                        conjch20_ch20,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_30xH_30: (1/(64*2*4))conjH_30*H_30*2^15
    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch30,
                        conjch30_ch30,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch21,
                        conjch20_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch31,
                        conjch30_ch31,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch21,
                        conjch21_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch31,
                        conjch31_ch31,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch20,
                        conjch21_ch20,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch30,
                        conjch31_ch30,
                        nb_rb_0,
                        shift);

    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              conjch20_ch20,
                              conjch21_ch21,
                              conjch30_ch30,
                              conjch31_ch31,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              conjch20_ch21,
                              conjch21_ch20,
                              conjch30_ch31,
                              conjch31_ch30,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0,
                              symbol);
  }
  if (nb_rx_ant == 2) {
    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0,
                              symbol);
  }

  // Add noise_var such that: H^h * H + noise_var * I
  if (noise_var != 0) {
    simde__m128i nvar_128i = simde_mm_set1_epi32(noise_var);
    simde__m128i *af_mf_00_128i = (simde__m128i *)af_mf_00;
    simde__m128i *af_mf_11_128i = (simde__m128i *)af_mf_11;
    for (int k = 0; k < 3 * nb_rb_0; k++) {
      af_mf_00_128i[0] = simde_mm_add_epi32(af_mf_00_128i[0], nvar_128i);
      af_mf_11_128i[0] = simde_mm_add_epi32(af_mf_11_128i[0], nvar_128i);
      af_mf_00_128i++;
      af_mf_11_128i++;
    }
  }

  //det_HhH = ad -bc
  nr_ulsch_det_HhH(af_mf_00,//a
             af_mf_01,//b
             af_mf_10,//c
             af_mf_11,//d
             determ_fin,
             nb_rb_0,
             symbol,
             shift);
  /* 2- Compute the channel matrix inversion **********************************
   *
     *    |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
     * A= |                                                                 |
     *    |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
     *
     *
     *
     *inv(A) =(1/det)*[d  -b
     *                 -c  a]
     *
     *
     **************************************************************************/
  simde__m128i *ul_ch_mag128_0 = NULL, *ul_ch_mag128b_0 = NULL, *ul_ch_mag128c_0 = NULL; // Layer 0
  simde__m128i *ul_ch_mag128_1 = NULL, *ul_ch_mag128b_1 = NULL, *ul_ch_mag128c_1 = NULL; // Layer 1
  simde__m128i mmtmpD0, mmtmpD1, mmtmpD2, mmtmpD3;
  simde__m128i QAM_amp128 = {0}, QAM_amp128b = {0}, QAM_amp128c = {0};

  simde__m128i *determ_fin_128 = (simde__m128i *)&determ_fin[0];

  simde__m128i *after_mf_a_128 = (simde__m128i *)af_mf_00;
  simde__m128i *after_mf_b_128 = (simde__m128i *)af_mf_01;
  simde__m128i *after_mf_c_128 = (simde__m128i *)af_mf_10;
  simde__m128i *after_mf_d_128 = (simde__m128i *)af_mf_11;
  
  simde__m128i *rxdataF_comp128_0 = (simde__m128i *)&rxdataF_comp[0][bufOffset];
  simde__m128i *rxdataF_comp128_1 = (simde__m128i *)&rxdataF_comp[nb_rx_ant][bufOffset];

  if (mod_order > 2) {
    if (mod_order == 4) {
      QAM_amp128 = simde_mm_set1_epi16(QAM16_n1); // 2/sqrt(10)
      QAM_amp128b = simde_mm_setzero_si128();
      QAM_amp128c = simde_mm_setzero_si128();
    } else if (mod_order == 6) {
      QAM_amp128 = simde_mm_set1_epi16(QAM64_n1); // 4/sqrt{42}
      QAM_amp128b = simde_mm_set1_epi16(QAM64_n2); // 2/sqrt{42}
      QAM_amp128c = simde_mm_setzero_si128();
    } else if (mod_order == 8) {
      QAM_amp128 =  simde_mm_set1_epi16(QAM256_n1);
      QAM_amp128b = simde_mm_set1_epi16(QAM256_n2);
      QAM_amp128c = simde_mm_set1_epi16(QAM256_n3);
    }
    ul_ch_mag128_0 = (simde__m128i *)&ul_ch_mag[0];
    ul_ch_mag128b_0 = (simde__m128i *)&ul_ch_magb[0];
    ul_ch_mag128c_0 = (simde__m128i *)&ul_ch_magc[0];
    ul_ch_mag128_1 = (simde__m128i *)&ul_ch_mag[1];
    ul_ch_mag128b_1 = (simde__m128i *)&ul_ch_magb[1];
    ul_ch_mag128c_1 = (simde__m128i *)&ul_ch_magc[1];
  }

  for (int rb = 0; rb < 3 * nb_rb_0; rb++) {

    // Magnitude computation
    if (mod_order > 2) {
      uint64_t sum_det = 0;
      for (int k = 0; k < 4; k++) {
        sum_det += (((uint32_t *)&determ_fin_128[0])[k]);
      }
      // Add bias to reduce rounding error
      sum_det = (sum_det + 2) >> 2;

      int b = log2_approx(sum_det) - 8;
      if (b > 0) {
        mmtmpD2 = simde_mm_srai_epi32(determ_fin_128[0], b);
      } else {
        mmtmpD2 = simde_mm_slli_epi32(determ_fin_128[0], -b);
      }
      mmtmpD3 = simde_mm_unpacklo_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = simde_mm_unpackhi_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = simde_mm_packs_epi32(mmtmpD3, mmtmpD2);

      // Layer 0
      ul_ch_mag128_0[0] = mmtmpD2;
      ul_ch_mag128b_0[0] = mmtmpD2;
      ul_ch_mag128c_0[0] = mmtmpD2;
      ul_ch_mag128_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128_0[0], QAM_amp128);
      ul_ch_mag128_0[0] = simde_mm_slli_epi16(ul_ch_mag128_0[0], 1);
      ul_ch_mag128b_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128b_0[0], QAM_amp128b);
      ul_ch_mag128b_0[0] = simde_mm_slli_epi16(ul_ch_mag128b_0[0], 1);
      ul_ch_mag128c_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128c_0[0], QAM_amp128c);
      ul_ch_mag128c_0[0] = simde_mm_slli_epi16(ul_ch_mag128c_0[0], 1);

      // Layer 1
      ul_ch_mag128_1[0] = mmtmpD2;
      ul_ch_mag128b_1[0] = mmtmpD2;
      ul_ch_mag128c_1[0] = mmtmpD2;
      ul_ch_mag128_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128_1[0], QAM_amp128);
      ul_ch_mag128_1[0] = simde_mm_slli_epi16(ul_ch_mag128_1[0], 1);
      ul_ch_mag128b_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128b_1[0], QAM_amp128b);
      ul_ch_mag128b_1[0] = simde_mm_slli_epi16(ul_ch_mag128b_1[0], 1);
      ul_ch_mag128c_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128c_1[0], QAM_amp128c);
      ul_ch_mag128c_1[0] = simde_mm_slli_epi16(ul_ch_mag128c_1[0], 1);
    }

    // multiply by channel Inv
    //rxdataF_zf128_0 = rxdataF_comp128_0*d - b*rxdataF_comp128_1
    //rxdataF_zf128_1 = rxdataF_comp128_1*a - c*rxdataF_comp128_0
    //printf("layer_1 \n");
    mmtmpD0 = nr_ulsch_comp_muli_sum(rxdataF_comp128_0[0],
                               after_mf_d_128[0],
                               rxdataF_comp128_1[0],
                               after_mf_b_128[0],
                               determ_fin_128[0]);

    //printf("layer_2 \n");
    mmtmpD1 = nr_ulsch_comp_muli_sum(rxdataF_comp128_1[0],
                               after_mf_a_128[0],
                               rxdataF_comp128_0[0],
                               after_mf_c_128[0],
                               determ_fin_128[0]);

    rxdataF_comp128_0[0] = mmtmpD0;
    rxdataF_comp128_1[0] = mmtmpD1;

#ifdef DEBUG_DLSCH_DEMOD
    printf("\n Rx signal after ZF rb%d\n",rb);
    print_shorts(" Rx layer 1:",(int16_t*)&rxdataF_comp128_0[0]);
    print_shorts(" Rx layer 2:",(int16_t*)&rxdataF_comp128_1[0]);
#endif
    determ_fin_128 += 1;
    ul_ch_mag128_0 += 1;
    ul_ch_mag128_1 += 1;
    ul_ch_mag128b_0 += 1;
    ul_ch_mag128b_1 += 1;
    ul_ch_mag128c_0 += 1;
    ul_ch_mag128c_1 += 1;
    rxdataF_comp128_0 += 1;
    rxdataF_comp128_1 += 1;
    after_mf_a_128 += 1;
    after_mf_b_128 += 1;
    after_mf_c_128 += 1;
    after_mf_d_128 += 1;
  }
   return(0);
}



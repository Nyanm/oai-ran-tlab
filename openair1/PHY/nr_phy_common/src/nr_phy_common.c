/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_phy_common.h"
#include <complex.h>
#ifdef __aarch64__
#define USE_128BIT
#endif

#define PEAK_DETECT_THRESHOLD 15
simde__m128i byte2m128i[256];
void init_byte2m128i(void)
{
  for (int s = 0; s < 256; s++) {
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * (s & 1)), 0);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 1) & 1)), 1);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 2) & 1)), 2);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 3) & 1)), 3);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 4) & 1)), 4);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 5) & 1)), 5);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 6) & 1)), 6);
    byte2m128i[s] = simde_mm_insert_epi16(byte2m128i[s], (1 - 2 * ((s >> 7) & 1)), 7);
  }
}

void init_delay_table(uint16_t ofdm_symbol_size,
                      int max_delay_comp,
                      int max_ofdm_symbol_size,
                      c16_t delay_table[][max_ofdm_symbol_size])
{
  for (int delay = -max_delay_comp; delay <= max_delay_comp; delay++) {
    for (int k = 0; k < ofdm_symbol_size; k++) {
      double complex delay_cexp = cexp(I * (2.0 * M_PI * k * delay / ofdm_symbol_size));
      delay_table[max_delay_comp + delay][k].r = (int16_t)round(256 * creal(delay_cexp));
      delay_table[max_delay_comp + delay][k].i = (int16_t)round(256 * cimag(delay_cexp));
    }
  }
}

static inline c16_t saturating_sub(c16_t a, c16_t b)
{
  c32_t tmp = {a.r - abs(b.r), a.i - abs(b.i)};
  tmp = (c32_t){min(tmp.r, INT16_MAX), min(tmp.i, INT16_MAX)};
  c16_t tmp2 = (c16_t){max(tmp.r, -INT16_MAX), max(tmp.i, -INT16_MAX)};
  return tmp2;
}

//----------------------------------------------------------------------------------------------
// QPSK
//----------------------------------------------------------------------------------------------
void nr_qpsk_llr(const c16_t *rxdataF_comp, int16_t *llr, uint32_t nb_re)
{
  const c16_t *rxF = rxdataF_comp;
  c16_t *llr32 = (c16_t *)llr;
  for (int i = 0; i < nb_re; i++) {
    llr32[i].r = rxF[i].r >> 4;
    llr32[i].i = rxF[i].i >> 4;
  }
}

//----------------------------------------------------------------------------------------------
// 16-QAM
//----------------------------------------------------------------------------------------------

void nr_16qam_llr(const c16_t *rxdataF_comp, const c16_t *ch_mag_in, int16_t *llr, uint32_t nb_re)
{
  simde__m256i *rxF_256 = (simde__m256i *)rxdataF_comp;
  simde__m256i *ch_mag256 = (simde__m256i *)ch_mag_in;
  int64_t *llr_64 = (int64_t *)llr;

#ifndef USE_128BIT
  for (int i = 0; i < (nb_re >> 3); i++) {
    // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    simde__m256i xmm0 = protected_abs256(*rxF_256);
    // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
    xmm0 = simde_mm256_subs_epi16(*ch_mag256, xmm0);

    simde__m256i xmm1 = simde_mm256_unpacklo_epi32(*rxF_256, xmm0);
    simde__m256i xmm2 = simde_mm256_unpackhi_epi32(*rxF_256, xmm0);

    // xmm1 |1st 2ed 3rd 4th  9th 10th 13rd 14th|
    // xmm2 |5th 6th 7th 8th 11st 12ed 15th 16th|

    *llr_64++ = simde_mm256_extract_epi64(xmm1, 0);
    *llr_64++ = simde_mm256_extract_epi64(xmm1, 1);
    *llr_64++ = simde_mm256_extract_epi64(xmm2, 0);
    *llr_64++ = simde_mm256_extract_epi64(xmm2, 1);
    *llr_64++ = simde_mm256_extract_epi64(xmm1, 2);
    *llr_64++ = simde_mm256_extract_epi64(xmm1, 3);
    *llr_64++ = simde_mm256_extract_epi64(xmm2, 2);
    *llr_64++ = simde_mm256_extract_epi64(xmm2, 3);
    rxF_256++;
    ch_mag256++;
  }

  nb_re &= 0x7;
#endif

  simde__m128i *rxF_128 = (simde__m128i *)rxF_256;
  simde__m128i *ch_mag_128 = (simde__m128i *)ch_mag256;
  simde__m128i *llr_128 = (simde__m128i *)llr_64;

  // Each iteration does 4 RE (gives 16 16bit-llrs)
  for (int i = 0; i < (nb_re >> 2); i++) {
    // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    simde__m128i xmm0 = protected_abs128(*rxF_128);
    // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
    xmm0 = simde_mm_subs_epi16(*ch_mag_128, xmm0);

    llr_128[0] = simde_mm_unpacklo_epi32(*rxF_128, xmm0); // llr128[0] contains the llrs of the 1st,2nd,5th and 6th REs
    llr_128[1] = simde_mm_unpackhi_epi32(*rxF_128, xmm0); // llr128[1] contains the llrs of the 3rd, 4th, 7th and 8th REs
    llr_128 += 2;
    rxF_128++;
    ch_mag_128++;
  }


  nb_re &= 0x3;
  c16_t *rxDataF = (c16_t *)rxF_128;
  c16_t *ch_mag = (c16_t *)ch_mag_128;
  c16_t *llr_tail = (c16_t *)llr_128;
  for (uint i = 0U; i < nb_re; i++) {
    c16_t tmp = *rxDataF++;
    *llr_tail++ = tmp;
    *llr_tail++ = saturating_sub(*ch_mag++, tmp);
  }
}

//----------------------------------------------------------------------------------------------
// 64-QAM
//----------------------------------------------------------------------------------------------

void nr_64qam_llr(const c16_t *rxdataF_comp, const c16_t *ch_mag, const c16_t *ch_mag2, int16_t *llr, uint32_t nb_re)
{
  simde__m256i *rxF = (simde__m256i *)rxdataF_comp;

  simde__m256i *ch_maga = (simde__m256i *)ch_mag;
  simde__m256i *ch_magb = (simde__m256i *)ch_mag2;

  int32_t *llr_32 = (int32_t *)llr;
#ifndef USE_128BIT
  for (int i = 0; i < (nb_re >> 3); i++) {
    simde__m256i xmm0 = simde_mm256_loadu_si256(rxF);
    // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    simde__m256i xmm1 = protected_abs256(xmm0);
    // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
    xmm1 = simde_mm256_subs_epi16(*ch_maga, xmm1);
    simde__m256i xmm2 = protected_abs256(xmm1);
    xmm2 = simde_mm256_subs_epi16(*ch_magb, xmm2);
    // xmm0 |1st 4th 7th 10th 13th 16th 19th 22ed|
    // xmm1 |2ed 5th 8th 11th 14th 17th 20th 23rd|
    // xmm2 |3rd 6th 9th 12th 15th 18th 21st 24th|

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 0);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 0);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 0);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 1);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 1);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 1);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 2);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 2);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 2);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 3);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 3);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 3);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 4);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 4);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 4);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 5);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 5);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 5);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 6);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 6);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 6);

    *llr_32++ = simde_mm256_extract_epi32(xmm0, 7);
    *llr_32++ = simde_mm256_extract_epi32(xmm1, 7);
    *llr_32++ = simde_mm256_extract_epi32(xmm2, 7);
    rxF++;
    ch_maga++;
    ch_magb++;
  }

  nb_re &= 0x7;
#endif

  simde__m128i *rxF_128 = (simde__m128i *)rxF;
  simde__m128i *ch_mag_128 = (simde__m128i *)ch_maga;
  simde__m128i *ch_magb_128 = (simde__m128i *)ch_magb;
  // Each iteration does 4 RE (gives 24 16bit-llrs)
  for (int i = 0; i < (nb_re >> 2); i++) {
    simde__m128i xmm0, xmm1, xmm2;
    xmm0 = *rxF_128;
    xmm1 = protected_abs128(xmm0);
    xmm1 = simde_mm_subs_epi16(*ch_mag_128, xmm1);
    xmm2 = protected_abs128(xmm1);
    xmm2 = simde_mm_subs_epi16(*ch_magb_128, xmm2);

    *llr_32++ = simde_mm_extract_epi32(xmm0, 0);
    *llr_32++ = simde_mm_extract_epi32(xmm1, 0);
    *llr_32++ = simde_mm_extract_epi32(xmm2, 0);
    *llr_32++ = simde_mm_extract_epi32(xmm0, 1);
    *llr_32++ = simde_mm_extract_epi32(xmm1, 1);
    *llr_32++ = simde_mm_extract_epi32(xmm2, 1);
    *llr_32++ = simde_mm_extract_epi32(xmm0, 2);
    *llr_32++ = simde_mm_extract_epi32(xmm1, 2);
    *llr_32++ = simde_mm_extract_epi32(xmm2, 2);
    *llr_32++ = simde_mm_extract_epi32(xmm0, 3);
    *llr_32++ = simde_mm_extract_epi32(xmm1, 3);
    *llr_32++ = simde_mm_extract_epi32(xmm2, 3);
    rxF_128++;
    ch_mag_128++;
    ch_magb_128++;
  }

  nb_re &= 0x3;

  c16_t *rxDataF = (c16_t *)rxF_128;
  c16_t *ch_mag_tail = (c16_t *)ch_mag_128;
  c16_t *ch_magb_tail = (c16_t *)ch_magb_128;
  c16_t *llr_tail = (c16_t *)llr_32;
  for (int i = 0; i < nb_re; i++) {
    *llr_tail++ = *rxDataF;
    c16_t tmp = saturating_sub(*ch_mag_tail++, *rxDataF++);
    *llr_tail++ = tmp;
    *llr_tail++ = saturating_sub(*ch_magb_tail++, tmp);
  }
}

void nr_256qam_llr(const c16_t *rxdataF_comp, const c16_t *ch_mag, const c16_t *ch_mag2, const c16_t *ch_mag3, int16_t *llr, uint32_t nb_re)
{
  simde__m256i *rxF_256 = (simde__m256i *)rxdataF_comp;
  simde__m256i *llr256 = (simde__m256i *)llr;

  simde__m256i *ch_maga = (simde__m256i *)ch_mag;
  simde__m256i *ch_magb = (simde__m256i *)ch_mag2;
  simde__m256i *ch_magc = (simde__m256i *)ch_mag3;
#ifndef USE_128BIT
  for (int i = 0; i < (nb_re >> 3); i++) {
    // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    simde__m256i xmm0 = protected_abs256(*rxF_256);
    // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
    xmm0 = simde_mm256_subs_epi16(*ch_maga, xmm0);
    //  xmmtmpD2 contains 16 LLRs
    simde__m256i xmm1 = protected_abs256(xmm0);
    xmm1 = simde_mm256_subs_epi16(*ch_magb, xmm1); // contains 16 LLRs
    simde__m256i xmm2 = protected_abs256(xmm1);
    xmm2 = simde_mm256_subs_epi16(*ch_magc, xmm2); // contains 16 LLRs
    // rxF[i] A0 A1 A2 A3 A4 A5 A6 A7 bits 7,6
    // xmm0   B0 B1 B2 B3 B4 B5 B6 B7 bits 5,4
    // xmm1   C0 C1 C2 C3 C4 C5 C6 C7 bits 3,2
    // xmm2   D0 D1 D2 D3 D4 D5 D6 D7 bits 1,0
    simde__m256i xmm3 = simde_mm256_unpacklo_epi32(*rxF_256, xmm0); // A0 B0 A1 B1 A4 B4 A5 B5
    simde__m256i xmm4 = simde_mm256_unpackhi_epi32(*rxF_256, xmm0); // A2 B2 A3 B3 A6 B6 A7 B7
    simde__m256i xmm5 = simde_mm256_unpacklo_epi32(xmm1, xmm2); // C0 D0 C1 D1 C4 D4 C5 D5
    simde__m256i xmm6 = simde_mm256_unpackhi_epi32(xmm1, xmm2); // C2 D2 C3 D3 C6 D6 C7 D7

    xmm0 = simde_mm256_unpacklo_epi64(xmm3, xmm5); // A0 B0 C0 D0 A4 B4 C4 D4
    xmm1 = simde_mm256_unpackhi_epi64(xmm3, xmm5); // A1 B1 C1 D1 A5 B5 C5 D5
    xmm2 = simde_mm256_unpacklo_epi64(xmm4, xmm6); // A2 B2 C2 D2 A6 B6 C6 D6
    xmm3 = simde_mm256_unpackhi_epi64(xmm4, xmm6); // A3 B3 C3 D3 A7 B7 C7 D7
    simde_mm256_storeu_si256(llr256, simde_mm256_permute2x128_si256(xmm0, xmm1, 0x20)); // A0 B0 C0 D0 A1 B1 C1 D1
    llr256++;
    simde_mm256_storeu_si256(llr256, simde_mm256_permute2x128_si256(xmm2, xmm3, 0x20)); // A2 B2 C2 D2 A3 B3 C3 D3
    llr256++;
    simde_mm256_storeu_si256(llr256, simde_mm256_permute2x128_si256(xmm0, xmm1, 0x31)); // A4 B4 C4 D4 A5 B5 C5 D5
    llr256++;
    simde_mm256_storeu_si256(llr256, simde_mm256_permute2x128_si256(xmm2, xmm3, 0x31)); // A6 B6 C6 D6 A7 B7 C7 D7
    llr256++;

    ch_magc++;
    ch_magb++;
    ch_maga++;
    rxF_256++;
  }

  nb_re &= 0x7;
#endif

  simde__m128i *rxF_128 = (simde__m128i *)rxF_256;
  simde__m128i *llr_128 = (simde__m128i *)llr256;

  simde__m128i *ch_maga_128 = (simde__m128i *)ch_maga;
  simde__m128i *ch_magb_128 = (simde__m128i *)ch_magb;
  simde__m128i *ch_magc_128 = (simde__m128i *)ch_magc;
  for (int i = 0; i < (nb_re >> 2); i++) {
    simde__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6;
    // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    xmm0 = protected_abs128(*rxF_128);
    // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
    xmm0 = simde_mm_subs_epi16(*ch_maga_128, xmm0);
    xmm1 = protected_abs128(xmm0);
    xmm1 = simde_mm_subs_epi16(*ch_magb_128, xmm1); // contains 8 LLRs
    xmm2 = protected_abs128(xmm1);
    xmm2 = simde_mm_subs_epi16(*ch_magc_128, xmm2); // contains 8 LLRs
    // rxF[i] A0 A1 A2 A3
    // xmm0   B0 B1 B2 B3
    // xmm1   C0 C1 C2 C3
    // xmm2   D0 D1 D2 D3
    xmm3 = simde_mm_unpacklo_epi32(*rxF_128, xmm0); // A0 B0 A1 B1
    xmm4 = simde_mm_unpackhi_epi32(*rxF_128, xmm0); // A2 B2 A3 B3
    xmm5 = simde_mm_unpacklo_epi32(xmm1, xmm2); // C0 D0 C1 D1
    xmm6 = simde_mm_unpackhi_epi32(xmm1, xmm2); // C2 D2 C3 D3

    *llr_128++ = simde_mm_unpacklo_epi64(xmm3, xmm5); // A0 B0 C0 D0
    *llr_128++ = simde_mm_unpackhi_epi64(xmm3, xmm5); // A1 B1 C1 D1
    *llr_128++ = simde_mm_unpacklo_epi64(xmm4, xmm6); // A2 B2 C2 D2
    *llr_128++ = simde_mm_unpackhi_epi64(xmm4, xmm6); // A3 B3 C3 D3

    rxF_128++;
    ch_maga_128++;
    ch_magb_128++;
    ch_magc_128++;
  }

  nb_re &= 0x3;
  c16_t *rxDataF = (c16_t *)rxF_128;
  c16_t *ch_mag_tail = (c16_t *)ch_maga_128;
  c16_t *ch_magb_tail = (c16_t *)ch_magb_128;
  c16_t *ch_magc_tail = (c16_t *)ch_magc_128;
  c16_t *llr_tail = (c16_t *)llr_128;
  for (int i = 0; i < nb_re; i++) {
    c16_t tmp = *rxDataF++;
    *llr_tail++ = tmp;
    c16_t tmp1 = saturating_sub(*ch_mag_tail++, tmp);
    *llr_tail++ = tmp1;
    c16_t tmp2 = saturating_sub(*ch_magb_tail++, tmp1);
    *llr_tail++ = tmp2;
    *llr_tail++ = saturating_sub(*ch_magc_tail++, tmp2);
  }
}

void freq2time(uint16_t ofdm_symbol_size, int16_t *freq_signal, int16_t *time_signal)
{
  const idft_size_idx_t idft_size = get_idft(ofdm_symbol_size);
  idft(idft_size, freq_signal, time_signal, 1);
}

void nr_est_delay(int ofdm_symbol_size, const c16_t *ls_est, c16_t *ch_estimates_time, delay_t *delay)
{
  idft(get_idft(ofdm_symbol_size), (int16_t *)ls_est, (int16_t *)ch_estimates_time, 1);

  int max_pos = delay->delay_max_pos;
  int max_val = delay->delay_max_val;
  const int sync_pos = 0;

  uint64_t mean_val = 0;
  for (int i = 0; i < ofdm_symbol_size; i++) {
    int temp = c16amp2(ch_estimates_time[i]) >> 1;
    mean_val += temp;
    if (temp > max_val) {
      max_pos = i;
      max_val = temp;
    }
  }
  mean_val /= ofdm_symbol_size;

  if (max_pos > ofdm_symbol_size / 2)
    max_pos = max_pos - ofdm_symbol_size;

  delay->delay_max_pos = max_pos;
  delay->delay_max_val = max_val;

  // The peak in general is quite clear. It only gives a small peak when the noise is high, generally obtaining an incorrect
  // estimated delay, and causing the delay compensation to worsen the result instead of improving it. After analyzing several
  // peaks, and doing many tests, a PEAK_DETECT_THRESHOLD = 15 is an adequate value, to apply delay compensation only when there is
  // clearly a peak
  delay->est_delay = mean_val > 0 && max_val / mean_val > PEAK_DETECT_THRESHOLD ? max_pos - sync_pos : 0;
}

unsigned int nr_get_tx_amp(int power_dBm, int power_max_dBm, int total_nb_rb, int nb_rb)
{
  // assume power at AMP is 20dBm
  // if gain = 20 (power == 40)
  int gain_dB = power_dBm - power_max_dBm;
  double gain_lin;

  gain_lin = pow(10, .1 * gain_dB);
  if ((nb_rb > 0) && (nb_rb <= total_nb_rb)) {
    return ((int)(AMP * sqrt(gain_lin * total_nb_rb / (double)nb_rb)));
  } else {
    LOG_E(PHY, "Illegal nb_rb/N_RB_UL combination (%d/%d)\n", nb_rb, total_nb_rb);
    // mac_xface->macphy_exit("");
  }
  return (0);
}

// compute average channel_level on each antenna
void nr_channel_level(const int symbol,
                      const int size_est,
                      const c16_t ch_estimates_ext[][size_est],
                      const int nb_rx,
                      const int Nl,
                      int32_t avg[nb_rx * Nl],
                      const uint32_t len)
{
  int16_t x = factor2(len);
  int16_t y = len >> x;
  for (int aarx = 0; aarx < nb_rx; aarx++) {
    for (int l = 0; l < Nl; l++) {
      simde__m128i *ch128 = (simde__m128i *)&ch_estimates_ext[l * nb_rx + aarx][symbol * len];
      //compute average level
      avg[l * nb_rx + aarx] = simde_mm_average(ch128, len, x, y);
      LOG_D(PHY, "Channel level: %d\n", avg[l * nb_rx + aarx]);
    }
  }
}

void nr_scale_channel(int size, int ch_estimates_ext[][size], int symb, uint32_t len, int nrOfLayers, int nb_rx, int shift_ch_ext)
{
  for (int l = 0; l < nrOfLayers; l++) {
    for (int aarx = 0; aarx < nb_rx; aarx++) {
      simde__m128i *ul_ch128 = (simde__m128i *)&ch_estimates_ext[l * nb_rx + aarx][symb * len];
      int loop_end = len >> 2;
      for (int i = 0; i < loop_end; i++) {
        ul_ch128[i] = simde_mm_srai_epi16(ul_ch128[i], shift_ch_ext);
      }
      // loop for the remaining elements
      int start_index = loop_end * 4;
      for (int j = start_index; j < len; j++) {
        c16_t *temp = ((c16_t *)ul_ch128) + j;
        *temp = c16Shift(*temp, shift_ch_ext);
      }
    }
  }
}

void nr_fo_compensation(double fo_Hz, int samples_per_ms, int sample_offset, const c16_t *rxdata_in, c16_t *rxdata_out, int size)
{
  const double phase_inc = -fo_Hz / (samples_per_ms * 1000);
  double phase = sample_offset * phase_inc;
  phase -= (int)phase;
#if 1
  // The bottleneck is the calculation of the complex rotation values using get_sin_cos().
  // This code path does not compute these values for the complete OFDM symbol, but only for a smaller CHUNK size.
  // After applying the rotation to a CHUNK size of the output, these rotation values are efficiently rotated further by `rot_vec`.
  // Unfortunately, this propagates small errors from one chunk to the next.
  // Therefore, there is a tradeoff between speed (better with small CHUNK sizes) and accuracy (better with large CHUNK sizes).
#define CHUNK 128
  c16_t rot[CHUNK] __attribute__((aligned(32)));
  for (int i = 0; i < CHUNK; i++) {
    rot[i] = get_sin_cos(phase);
    phase += phase_inc;
  }
  const c16_t rot_vec = get_sin_cos(CHUNK * phase_inc);
  while (size > CHUNK) {
    mult_complex_vectors(rxdata_in, rot, rxdata_out, CHUNK, 14);
    rotate_cpx_vector(rot, &rot_vec, rot, CHUNK, 14);
    rxdata_in += CHUNK;
    rxdata_out += CHUNK;
    size -= CHUNK;
  }
  mult_complex_vectors(rxdata_in, rot, rxdata_out, size, 14);
#else
  // This code path computes the complex rotation values for the complete OFDM symbol using get_sin_cos().
  // This is more accurate, but also slower than the code path above.
  c16_t rot[size] __attribute__((aligned(32)));
  for (int i = 0; i < size; i++) {
    rot[i] = get_sin_cos(phase);
    phase += phase_inc;
  }
  mult_complex_vectors(rxdata_in, rot, rxdata_out, size, 14);
#endif
}

/*!
* Setting the first subcarrier
* 3GPP TS 38.211 sections 7.4.3.1 and 4.4.4.2
* for FR1 offsetToPointA and k_SSB are expressed in terms of 15 kHz SCS
* for FR2 offsetToPointA is expressed in terms of 60 kHz SCS and k_SSB expressed in terms of the SCS provided
* by the higher-layer parameter subCarrierSpacingCommon
*/
int nr_get_ssb_start_sc(int scs, int ssb_offset_point_a, int ssb_sco, frequency_range_t freq_range)
{
  const int prb_offset =
      (freq_range == FR1) ? ssb_offset_point_a >> scs : ssb_offset_point_a >> (scs - 2);
  const int sc_offset =
      (freq_range == FR1) ? ssb_sco >> scs : ssb_sco;

  int ssb_start_subcarrier = (12 * prb_offset + sc_offset);

  LOG_D(NR_PHY, "prb_offset:%d, ssb_subcarrier_offset:%d,scs :%d, Fr:%d, ssb_start_subcarrier:%d\n",
                        prb_offset, ssb_sco, scs, freq_range, ssb_start_subcarrier);

  return ssb_start_subcarrier;

}

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



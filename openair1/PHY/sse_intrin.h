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

/*! \file PHY/sse_intrin.h
 * \brief SSE includes and compatibility functions.
 *
 * This header collects all SSE compatibility functions. To use SSE inside a source file, include only sse_intrin.h.
 * The host CPU needs to have support for SSE2 at least. SSE3 and SSE4.1 functions are emulated if the CPU lacks support for them.
 * This will slow down the softmodem, but may be valuable if only offline signal processing is required.
 *
 * 
 * Has been changed in August 2022 to rely on SIMD Everywhere (SIMDE) from MIT
 * by bruno.mongazon-cazavet@nokia-bell-labs.com
 *
 * All AVX2 code is mapped to SIMDE which transparently relies on AVX2 HW (avx2-capable host) or SIMDE emulation
 * (non-avx2-capable host).
 * To force using SIMDE emulation on avx2-capable host use the --noavx2 flag. 
 * avx512 code is not mapped to SIMDE. It depends on --noavx512 flag.
 * If the --noavx512 is set the OAI AVX512 emulation using AVX2 is used.
 * If the --noavx512 is not set, AVX512 HW is used on avx512-capable host while OAI AVX512 emulation using AVX2
 * is used on non-avx512-capable host. 
 *
 * \author S. Held, Laurent THOMAS
 * \email sebastian.held@imst.de, laurent.thomas@open-cells.com	
 * \company IMST GmbH, Open Cells Project
 * \date 2019
 * \version 0.2
*/

#ifndef SSE_INTRIN_H
#define SSE_INTRIN_H

#include <simde/simde-common.h>
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>

#if defined(__AVX512BW__) || defined(__AVX512F__)
#include <immintrin.h>
// a solution should be found to use simde package for also AVX512, but it is C++ implementation, difficult to use in OAI
typedef struct {
  union {
    __m512i v;
    int16_t i16[32];
    int8_t i8[64];
  };
} oai512_t;
#endif

//Note that the following is not needed for gcc>=13
#ifdef __aarch64__
#ifdef __ARM_FEATURE_SVE2
static inline svint16_t cast_neon_to_sve_s16(simde__m128i n) {
    svint16_t s;
// "w" refers to an FP/SIMD register.
// This tells GCC: "Take the value in the Neon register and
// just start calling it an SVE register."
    asm ("" : "=w" (s) : "0" (n));
    return s;
}
static inline simde__m128i cast_sve_to_neon_s16(svint16_t s) {
    simde__m128i n;
    asm ("" : "=w" (n) : "0" (s));
    return n;
}
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__arm__) || defined(__aarch64__)
/* ARM processors */
// note this fails on some x86 machines, with an error like:
// /usr/lib/gcc/x86_64-redhat-linux/8/include/gfniintrin.h:57:1: error: inlining failed in call to always_inline ‘_mm_gf2p8affine_epi64_epi8’: target specific option mismatch
#include <simde/x86/clmul.h>
#include <arm_neon.h>
#endif // x86_64 || i386

#include <stdbool.h>
#include "assertions.h"


#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif
/*
 * OAI specific SSE section
 */

typedef struct {
  union {
    simde__m128i v;
    int16_t i16[8];
    int8_t i8[16];
  };
} oai128_t;
typedef struct {
  union {
    simde__m256i v;
    int16_t i16[16];
    int8_t i8[32];
  };
} oai256_t;

__attribute__((always_inline)) static inline int64_t simde_mm_average_sse(simde__m128i *a, int length, int shift)
{
  // compute average level with shift (64-bit verstion)
  simde__m128i avg128 = simde_mm_setzero_si128();
  for (int i = 0; i < length >> 2; i++) {
    const simde__m128i in1 = a[i];
    avg128 = simde_mm_add_epi32(avg128, simde_mm_srai_epi32(simde_mm_madd_epi16(in1, in1), shift));
  }

  // Horizontally add pairs
  // 1st [A + C, B + D]
  simde__m128i sum_pairs = simde_mm_add_epi64(simde_mm_unpacklo_epi32(avg128, simde_mm_setzero_si128()), // [A, B] → [A, 0, B, 0]
                                              simde_mm_unpackhi_epi32(avg128, simde_mm_setzero_si128())  // [C, D] → [C, 0, D, 0]
  );

  // 2nd [A + B + C + D, ...]
  simde__m128i total_sum = simde_mm_add_epi64(sum_pairs, simde_mm_shuffle_epi32(sum_pairs, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));

  // Extract horizontal sum as a scalar int64_t result
  return simde_mm_cvtsi128_si64(total_sum);
}

__attribute__((always_inline)) static inline int64_t simde_mm_average_avx2(simde__m256i *a, int length, int shift)
{
  simde__m256i avg256 = simde_mm256_setzero_si256();
  for (int i = 0; i < length >> 3; i++) {
    const simde__m256i in1 = simde_mm256_loadu_si256(&a[i]); // unaligned load
    avg256 = simde_mm256_add_epi32(avg256, simde_mm256_srai_epi32(simde_mm256_madd_epi16(in1, in1), shift));
  }

  // Split the 256-bit vector into two 128-bit halves and convert to 64-bit
  // [A + E, B + F, C + G, D + H]
  simde__m256i sum_pairs = simde_mm256_add_epi64(
      simde_mm256_cvtepi32_epi64(simde_mm256_castsi256_si128(avg256)),     // [A, B, C, D] → [A, 0, B, 0, C, 0, D, 0]
      simde_mm256_cvtepi32_epi64(simde_mm256_extracti128_si256(avg256, 1)) // [E, F, G, H] → [E, 0, F, 0, G, 0, H, 0]
  );

  // Horizontal sum within the 256-bit vector
  // [A + E + B + F, C + G + D + H]
  simde__m128i total_sum = simde_mm_add_epi64(simde_mm256_castsi256_si128(sum_pairs), simde_mm256_extracti128_si256(sum_pairs, 1));

  // [A + E + B + F + C + G + D + H, ...]
  total_sum = simde_mm_add_epi64(total_sum, simde_mm_shuffle_epi32(total_sum, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));

  // Extract horizontal sum as a scalar int64_t result
  return simde_mm_cvtsi128_si64(total_sum);
}

__attribute__((always_inline)) static inline int32_t simde_mm_average(simde__m128i *a, int length, int shift, int16_t scale)
{
  int64_t avg = 0;

#if defined(__x86_64__) || defined(__i386__)
  if (__builtin_cpu_supports("avx2")) {
    avg += simde_mm_average_avx2((simde__m256i *)a, length, shift);

    // tail processing by SSE
    a += ((length & ~7) >> 2);
    length -= (length & ~7);
  }
#endif

  avg += simde_mm_average_sse(a, length, shift);

  return (uint32_t)(avg / scale);
}

/**
 * Perform element-wise conjugation on a 128-bit SIMD vector of 16-bit integers.
 *
 * The flips the sign of imaginary part of each complex element in the vector:
 * Input:  [r0,  i0, ..., r3,  i3]
 * Output: [r0, -i0, ..., r3, -i3]
 *
 * @param 128-bit SIMD vector of 4x complex 16-bit integers.
 * @return Complex conjugated 128-bit SIMD vector.
 */
__attribute__((always_inline)) static inline simde__m128i oai_mm_conj(simde__m128i a)
{
#ifdef __aarch64__
  const oai128_t neg_imag = {.i16 = {0, -1, 0, -1, 0, -1, 0, -1}};
  int16x8_t aneg = vnegq_s16((int16x8_t)a);
  return (simde__m128i)vbslq_s16((uint16x8_t)neg_imag.v, aneg, (int16x8_t)a);
#else
  const oai128_t neg_imag = {.i16 = {1, -1, 1, -1, 1, -1, 1, -1}};
  return simde_mm_sign_epi16(a, neg_imag.v);
#endif
}

/**
 * Perform element-wise IQ swap on a 128-bit SIMD vector of 16-bit integers.
 *
 * The swap imag and real part of each complex element in the vector:
 * Input:  [r0, i0, ..., r3, i3]
 * Output: [i0, r0, ..., i3, r3]
 *
 * @param 128-bit SIMD vector of 16-bit integers.
 * @return Swaped 128-bit SIMD vector.
 */
__attribute__((always_inline)) static inline
simde__m128i oai_mm_swap(simde__m128i a)
{
  // Shuffle mask to swap bytes for IQ swapping
#ifdef __aarch64__
  const oai128_t shuffle_mask_swap = {.i8 = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13}};
  return simde_mm_shuffle_epi8(a, shuffle_mask_swap.v);
#else
  return (simde__m128i)vrev32_s16((int16x8_t)a);
#endif
}

__attribute__((always_inline)) static inline
simde__m128i oai_mm_smadd(simde__m128i z1, simde__m128i z2, int shift)
{
  return simde_mm_srai_epi32(simde_mm_madd_epi16(z1, z2), shift);
}

__attribute__((always_inline)) static inline
simde__m128i oai_mm_pack(simde__m128i a, simde__m128i b)
{
  return simde_mm_packs_epi32(
    simde_mm_unpacklo_epi32(a, b), // real
    simde_mm_unpackhi_epi32(a, b)  // imag
  );
}

/**
 * Perform a COMPLEX MULTIPLICATION on a 128-bit SIMD vector of complex 16-bit integers.
 *
 * Input:  z1 = (a + bi) [ a0,  b0,  ...,  a3,  b3]
 * Input:  z2 = (c + di) [ c0,  d0,  ...,  c3,  d3]
 * Output: z3 = (e + fi) [ e0,  f0,  ...,  e3,  f3]
 *
 * conj(z1)              [ a0, -b0,  ...,  a3, -b3]
 * swap(z1)              [ b0,  a0,  ...,  b3,  a3] 
 * 
 * z3 = z1 * z2 = + (ac-bd) + (ad+bc)i
 *
 * @param 128-bit SIMD vector of four complex 16-bit integers.
 * @return a 128-bit SIMD vector.
 */
__attribute__((always_inline)) static inline
simde__m128i oai_mm_cpx_mult(simde__m128i z1, simde__m128i z2, int shift)
{
  simde__m128i re = oai_mm_smadd(oai_mm_conj(z1), z2, shift);
  simde__m128i im = oai_mm_smadd(oai_mm_swap(z1), z2, shift);
  return oai_mm_pack(re, im);
}

/**
 * Perform a CONJUGATE multiplication on a 128-bit SIMD vector of 16-bit integers.
 *
 * Input:  z1 = (a + bi) [ a0,  b0,  ...,  a3,  b3]
 * Input:  z2 = (c + di) [ c0,  d0,  ...,  c3,  d3]
 * Output: z3 = (e + fi) [ e0,  f0,  ...,  e3,  f3]
 * z3 = z1 * conj(z2) = + (ac+bd) + i(bc-ad)
 *
 * @param 128-bit SIMD vector of four complex 16-bit integers.
 * @return a 128-bit SIMD vector.
 */
	
__attribute__((always_inline)) static inline
simde__m128i oai_mm_cpx_mult_conj(simde__m128i a, simde__m128i b, int shift)
{
#ifdef __aarch64__
#ifdef __ARM_FEATURE_SVE2

/**
 *  * Full-vector Complex Mul for int16_t with Dynamic Shift.
 *   * a, b: Interleaved [R, I, R, I...]
 *    * shift: The calculated optimal shift for the block.
 *     */
    svbool_t pg = svptrue_b16();
    uint64_t shift64 = (uint64_t)shift;

    svint16_t asv = cast_neon_to_sve_s16(a);// with gcc 13+ svset_neonq_s16(svundef_s16(),a);
    svint16_t bsv = cast_neon_to_sve_s16(b);

// --- 1. Process Bottom Half (Lanes 0, 2, 4...) ---
//     // Calculates: (a.re*b.re - a.im*b.im) and (a.re*b.im + a.im*b.re)
    svint32_t res_re_low = svqdmlalb_s32(svdup_n_s32(0), asv,  bsv, 0);
    svint32_t res_im_low = svqdmlalb_s32(svdup_n_s32(0), asv,  bsv, 90);

// --- 2. Process Top Half (Lanes 1, 3, 5...) ---
    svint32_t res_re_high = svqdmlalt_s32(svdup_n_s32(0), asv, bsv, 0);
    svint32_t res_im_high = svqdmlalt_s32(svdup_n_s32(0), asv, bsv, 90);

// --- 3. Apply the Dynamic Shift ---
// Use svasr (Arithmetic Shift Right) for truncation.
// If you need to shift LEFT because inputs are small, use svlsl.
    svint32_t re_low_s = svasr_n_s32_z(svptrue_b32(), res_re_low, shift64);
    svint32_t im_low_s = svasr_n_s32_z(svptrue_b32(), res_im_low, shift64);
    svint32_t re_high_s = svasr_n_s32_z(svptrue_b32(), res_re_high, shift64);
    svint32_t im_high_s = svasr_n_s32_z(svptrue_b32(), res_im_high, shift64);

// --- 4. Saturate to Q15 Range ---
// svqxtn "Saturating Extract Narrow" clamps to [-32768, 32767]
    svint16_t re_low_16 = svqxtn_s32(re_low_s);
    svint16_t im_low_16 = svqxtn_s32(im_low_s);
    svint16_t re_high_16 = svqxtn_s32(re_high_s);
    svint16_t im_high_16 = svqxtn_s32(im_high_s);

// --- 5. Re-interleave Results ---
// Combine the low/high parts and interleave Real/Imaginary
    svint16_t re_full = svuzp1_s16(re_low_16, re_high_16); // Reconstructs full Real vector
    svint16_t im_full = svuzp1_s16(im_low_16, im_high_16); // Reconstructs full Imag vector

    return cast_sve_to_neon_s16(svzip1_s16(re_full, im_full));
}
#else
/*
    // 1. De-interleave real and imaginary parts
    // a_parts.val[0] = [R0, R1, R2, R3], a_parts.val[1] = [I0, I1, I2, I3]
    int16x4_t ar = vget_low_s16(vuzp1q_s16((int16x8_t)a, (int16x8_t)a)); // This gets the lower 4 complex pairs
    int16x4_t ai = vget_low_s16(vuzp2q_s16((int16x8_t)a, (int16x8_t)a)); // This gets the lower 4 complex pairs
    int16x4_t br = vget_low_s16(vuzp1q_s16((int16x8_t)b, (int16x8_t)b));
    int16x4_t bi = vget_low_s16(vuzp2q_s16((int16x8_t)b, (int16x8_t)b));
    
    // 2. Perform widening multiplication (16-bit * 16-bit -> 32-bit)
    int32x4_t re_re = vmull_s16(ar, br); // a.re * b.re
    int32x4_t im_im = vmull_s16(ai, bi); // a.im * b.im
    int32x4_t re_im = vmull_s16(ar, bi); // a.re * b.im
    int32x4_t im_re = vmull_s16(ai, br); // a.im * b.re
    
    // 3. Combine parts: Real = (re*re + im*im), Imag = (-re*im + im*re)
    int32x4_t res_re = vaddq_s32(re_re, im_im);
    int32x4_t res_im = vsubq_s32(im_re, re_im);
    
    // 4. Scale back (e.g., Q15 format) and Narrow back to 16-bit
    // vshrn_n_s32 shifts right and narrows. Use vqshrn for saturating narrow.
    int16x4_t out_re = vmovn_s32(vshlq_s32(res_re,vdupq_n_s32(-shift))); 
    int16x4_t out_im = vmovn_s32(vshlq_s32(res_im,vdupq_n_s32(-shift)));

    // 5. Re-interleave for the final result
    return (simde__m128i)vcombine_s16(vzip1_s16(out_re, out_im), 
                                      vzip2_s16(out_re, out_im));
*/

  simde__m128i re = oai_mm_smadd(a, b, shift);
  simde__m128i im = oai_mm_smadd(oai_mm_swap(oai_mm_conj(a)), b, shift);
  return oai_mm_pack(re, im);
#endif
}

/*
 * OAI specific AVX2 section
 */

/**
 * Perform element-wise conjugation on a 256-bit SIMD vector of 16-bit integers.
 *
 * The flips the sign of imaginary part of each complex element in the vector:
 * Input:  [r0,  i0, ..., r7,  i7]
 * Output: [r0, -i0, ..., r7, -i7]
 *
 * @param 256-bit SIMD vector of 8x complex 16-bit integers.
 * @return Complex conjugated 256-bit SIMD vector.
 */
__attribute__((always_inline)) static inline simde__m256i oai_mm256_conj(simde__m256i a)
{
  const oai256_t neg_imag = {.i16 = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1}};
  return simde_mm256_sign_epi16(a, neg_imag.v);
}

/**
 * Perform element-wise IQ swap on a 256-bit SIMD vector of 16-bit integers.
 *
 * This swaps the real and imaginary parts of each complex element in the vector:
 * Input:  [r0, i0, ..., r7, i7]
 * Output: [i0, r0, ..., i7, r7]
 *
 * @param 256-bit SIMD vector of 16-bit integers.
 * @return Swapped 256-bit SIMD vector.
 */
__attribute__((always_inline)) static inline simde__m256i oai_mm256_swap(simde__m256i a)
{
  // Shuffle mask to swap bytes for IQ swapping
  const oai256_t shuffle_mask_swap = {.i8 = {
                                          2,  3,  0,  1,  6,  7,  4,  5,  10, 11, 8,  9,  14, 15, 12, 13, // Low bytes
                                          18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29 // High bytes
                                      }};
  return simde_mm256_shuffle_epi8(a, shuffle_mask_swap.v);
}

__attribute__((always_inline)) static inline
simde__m256i oai_mm256_smadd(simde__m256i z1, simde__m256i z2, int shift)
{
  return simde_mm256_srai_epi32(simde_mm256_madd_epi16(z1, z2), shift);
}

__attribute__((always_inline)) static inline
simde__m256i oai_mm256_pack(simde__m256i a, simde__m256i b)
{
  return simde_mm256_packs_epi32(
    simde_mm256_unpacklo_epi32(a, b), // real
    simde_mm256_unpackhi_epi32(a, b)  // imag
  );
}

/**
 * Perform a COMPLEX MULTIPLICATION on a 256-bit SIMD vector of complex 16-bit integers.
 *
 * Input:  z1 = (a + bi) [ a0,  b0,  ...,  a7,  b7 ]
 * Input:  z2 = (c + di) [ c0,  d0,  ...,  c7,  d7 ]
 * Output: z3 = (e + fi) [ e0,  f0,  ...,  e7,  f7]
 *
 * conj(z1)              [ a0, -b0,  ...,  a7, -b7]
 * swap(z1)              [ b0,  a0,  ...,  b7,  a7] 
 * 
 * z3 = z1 * z2 = + (ac-bd) + (ad+bc)i
 *
 * @param 256-bit SIMD vector of eight complex 16-bit integers.
 * @return a 256-bit SIMD vector.
 */
__attribute__((always_inline)) static inline
simde__m256i oai_mm256_cpx_mult(simde__m256i z1, simde__m256i z2, int shift)
{
  simde__m256i re = oai_mm256_smadd(oai_mm256_conj(z1), z2, shift);
  simde__m256i im = oai_mm256_smadd(oai_mm256_swap(z1), z2, shift);
  return oai_mm256_pack(re, im);
}

/**
 * Perform a CONJUGATE multiplication on a 256-bit SIMD vector of complex 16-bit integers.
 *
 * Input:  z1 = (a + bi) [ a0,  b0,  ...,  a3,  b3]
 * Input:  z2 = (c + di) [ c0,  d0,  ...,  c3,  d3]
 * Output: z3 = (e + fi) [ e0,  f0,  ...,  e3,  f3]
 * z3 =  z1 * conj(z2) =  (ac+bd) + i(bc-ad)
 *
 * @param 256-bit SIMD vector of eight complex 16-bit integers.
 * @return a 256-bit SIMD vector.
 */
__attribute__((always_inline)) static inline
simde__m256i oai_mm256_cpx_mult_conj(simde__m256i a, simde__m256i b, int shift)
{
  simde__m256i re = oai_mm256_smadd(a, b, shift);
  simde__m256i im = oai_mm256_smadd(oai_mm256_swap(oai_mm256_conj(a)), b, shift);
  return oai_mm256_pack(re, im);
}

#ifdef __AVX512BW__
__attribute__((always_inline)) static inline __m512i oai_mm512_conj(__512i a)
{
  const oai512_t neg_imag = {.i16 = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1}};
  return _mm512_sign_epi16(a, neg_imag.v);
}

__attribute__((always_inline)) static inline __m512i oai_mm512_swap(__m512i a)

{
  // Shuffle mask to swap bytes for IQ swapping
  const oai512_t shuffle_mask_swap = {.i8 = {
                                          2,  3,  0,  1,  6,  7,  4,  5,  10, 11, 8,  9,  14, 15, 12, 13,
                                          18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29,
                                          34, 35, 32, 33, 38, 39, 36, 37, 42, 43, 40, 41, 46, 47, 44, 45,
                                          50, 51, 48, 49, 54, 55, 52, 53, 58, 59, 56, 57, 62, 63, 60, 61
                                      }};
  return _mm512_shuffle_epi8(a, shuffle_mask_swap.v);
}

__attribute__((always_inline)) static inline
__m512i oai_mm512_smadd(__m512i z1, __m512i z2, int shift)
{
  return _mm512_srai_epi32(_mm512_madd_epi16(z1, z2), shift);
}

__attribute__((always_inline)) static inline
__m512i oai_mm512_pack(__m512i a, __m512i b)
{
  return _mm512_packs_epi32(
    _mm512_unpacklo_epi32(a, b), // real
    _mm512_unpackhi_epi32(a, b)  // imag
  );
}
__attribute__((always_inline)) static inline
__m512i oai_mm512_cpx_mult_conj(__m512i a, __m512i b, int shift)
{
  __m512i re = oai_mm512_smadd(a, b, shift);
  __m512i im = oai_mm512_smadd(oai_mm512_swap(oai_mm512_conj(a)), b, shift);
  return oai_mm256_pack(re, im);
}
#endif

#ifdef __cplusplus
}
#endif

#endif // SSE_INTRIN_H

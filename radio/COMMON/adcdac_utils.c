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

#include "adcdac_utils.h"
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>
#include <simde/arm/neon.h>
#include <simde/arm/sve.h>

void int16_to_msb(int16_t *restrict source, int16_t *restrict dest, size_t n)
{
  const int shift = 4;
  size_t i = 0;
  // x86: AVX-512 + AVX2 fall-through
#if defined(__x86_64__)
#if defined(__AVX512BW__)
  size_t vecsize512 = sizeof(simde__m512i) / sizeof(int16_t);
  size_t num_iter512 = n / vecsize512;
  simde__m512i *in512 = (simde__m512i *)source;
  simde__m512i *out512 = (simde__m512i *)dest;
  if ((((uintptr_t)in512) & 0x3F) == 0 && (((uintptr_t)out512) & 0x3F) == 0) {
    for (; i < num_iter512; i++) {
      out512[i] = simde_mm512_slli_epi16(in512[i], shift);
    }
  } else if ((((uintptr_t)out512) & 0x3F) == 0) {
    for (; i < num_iter512; i++) {
      simde__m512i tmp = simde_mm512_loadu_si512(&in512[i]);
      out512[i] = simde_mm512_slli_epi16(tmp, shift);
    }
  } else {
    for (; i < num_iter512; i++) {
      simde__m512i tmp = simde_mm512_loadu_si512(&in512[i]);
      tmp = simde_mm512_slli_epi16(tmp, shift);
      simde_mm512_storeu_si512(&out512[i], tmp);
    }
  }
  i *= vecsize512;
#endif
  // AVX2 for the tail
  size_t vecsize256 = sizeof(simde__m256i) / sizeof(int16_t);
  size_t num_iter256 = (n - i) / vecsize256;
  simde__m256i *in256 = (simde__m256i *)&source[i];
  simde__m256i *out256 = (simde__m256i *)&dest[i];
  if ((((uintptr_t)in256) & 0x1F) == 0 && (((uintptr_t)out256) & 0x1F) == 0) {
    for (size_t j = 0; j < num_iter256; j++) {
      out256[j] = simde_mm256_slli_epi16(in256[j], shift);
    }
  } else if ((((uintptr_t)out256) & 0x1F) == 0) {
    for (size_t j = 0; j < num_iter256; j++) {
      simde__m256i tmp = simde_mm256_loadu_si256(&in256[j]);
      out256[j] = simde_mm256_slli_epi16(tmp, shift);
    }
  } else {
    for (size_t j = 0; j < num_iter256; j++) {
      simde__m256i tmp = simde_mm256_loadu_si256(&in256[j]);
      tmp = simde_mm256_slli_epi16(tmp, shift);
      simde_mm256_storeu_si256(&out256[j], tmp);
    }
  }
  i += num_iter256 * vecsize256;
#else
#if defined(__ARM_FEATURE_SVE2)
  while (i < n) {
    size_t vl = simde_svcnth();
    svint16_t op2 = svdup_n_s16(shift);
    if (vl > n - i)
      vl = n - i;
    svbool_t pg = svwhilelt_b16((uint64_t)0, (uint64_t)vl);
    svint16_t vin  = svqrshl_s16_x(pg, svld1_s16(pg, &source[i]), op2);
    svst1_s16(pg, &dest[i], vin);
    i += vl;
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #error NOT IMPLEMENTED
#else
#endif
#endif
  for (; i < n; i++) {
    dest[i] = source[i] << shift;
  }
}

void int16_from_msb(int16_t *restrict source, int16_t *restrict dest, size_t n)
{
  const int shift = 4;
  size_t i = 0;
  // x86: AVX-512 + AVX2 fall-through
#if defined(__x86_64__)
#if defined(__AVX512BW__)
  size_t vecsize512 = sizeof(simde__m512i) / sizeof(int16_t);
  size_t num_iter512 = n / vecsize512;
  simde__m512i *in512 = (simde__m512i *)source;
  simde__m512i *out512 = (simde__m512i *)dest;
  if ((((uintptr_t)in512) & 0x3F) == 0 && (((uintptr_t)out512) & 0x3F) == 0) {
    for (; i < num_iter512; i++) {
      out512[i] = simde_mm512_srai_epi16(in512[i], shift);
    }
  } else if ((((uintptr_t)out512) & 0x3F) == 0) {
    for (; i < num_iter512; i++) {
      simde__m512i tmp = simde_mm512_loadu_si512(&in512[i]);
      out512[i] = simde_mm512_srai_epi16(tmp, shift);
    }
  } else {
    for (; i < num_iter512; i++) {
      simde__m512i tmp = simde_mm512_loadu_si512(&in512[i]);
      tmp = simde_mm512_srai_epi16(tmp, shift);
      simde_mm512_storeu_si512(&out512[i], tmp);
    }
  }
  i *= vecsize512;
#endif
  // AVX2 for the tail
  size_t vecsize256 = sizeof(simde__m256i) / sizeof(int16_t);
  size_t num_iter256 = (n - i) / vecsize256;
  simde__m256i *in256 = (simde__m256i *)&source[i];
  simde__m256i *out256 = (simde__m256i *)&dest[i];
  if ((((uintptr_t)in256) & 0x1F) == 0 && (((uintptr_t)out256) & 0x1F) == 0) {
    for (size_t j = 0; j < num_iter256; j++) {
      out256[j] = simde_mm256_srai_epi16(in256[j], shift);
    }
  } else if ((((uintptr_t)out256) & 0x1F) == 0) {
    for (size_t j = 0; j < num_iter256; j++) {
      simde__m256i tmp = simde_mm256_loadu_si256(&in256[j]);
      out256[j] = simde_mm256_srai_epi16(tmp, shift);
    }
  } else {
    for (size_t j = 0; j < num_iter256; j++) {
      simde__m256i tmp = simde_mm256_loadu_si256(&in256[j]);
      tmp = simde_mm256_srai_epi16(tmp, shift);
      simde_mm256_storeu_si256(&out256[j], tmp);
    }
  }
  i += num_iter256 * vecsize256;
#else
#if defined(__ARM_FEATURE_SVE2)
  while (i < n) {
    size_t vl = simde_svcnth();
    svint16_t op2 = svdup_n_s16(-shift);
    if (vl > n - i)
      vl = n - i;
    svbool_t pg = svwhilelt_b16((uint64_t)0, (uint64_t)vl);
    svint16_t vin  = svqrshl_s16_x(pg, svld1_s16(pg, &source[i]), op2);
    svst1_s16(pg, &dest[i], vin);
    i += vl;
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #error NOT IMPLEMENTED
#else
#endif
#endif
  for (; i < n; i++) {
    dest[i] = source[i] >> shift;
  }
}

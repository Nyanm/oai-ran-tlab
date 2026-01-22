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

#include <stdio.h>
#include <assert.h>
#include <threads.h>
#include <stdlib.h>
#include <gtest/gtest.h>
#include "adcdac_utils.h"
#include <simde/x86/avx2.h>
#include "common/platform_types.h"
#include "openair1/PHY/TOOLS/phy_test_tools.hpp"

static void int16_to_msb_ref(int16_t *source, int16_t *dest, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    dest[i] = source[i] << 4;
  }
}

static void int16_from_msb_ref(int16_t *source, int16_t *dest, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    dest[i] = source[i] >> 4;
  }
}

void test_int16_to_msb_case(size_t num_elements, int offset_in, int offset_out) {
  AlignedVector512<int16_t> input(num_elements + 16); // + padding for offsets
  AlignedVector512<int16_t> output(num_elements + 16);
  AlignedVector512<int16_t> output_ref(num_elements + 16);
  
  std::iota(input.begin(), input.end(), 0);

  int16_t* p_in = input.data() + offset_in;
  int16_t* p_out = output.data() + offset_out;
  int16_t* p_ref = output_ref.data() + offset_out;

  int16_to_msb(p_in, p_out, num_elements);
  int16_to_msb_ref(p_in, p_ref, num_elements);

  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(p_out[i], p_ref[i]) << "Mismatch at index " << i << " with offsets " << offset_in << ", " << offset_out;
  }
}

void test_int16_from_msb_case(size_t num_elements, int offset_in, int offset_out) {
  AlignedVector512<int16_t> input(num_elements + 16); // + padding for offsets
  AlignedVector512<int16_t> output(num_elements + 16);
  AlignedVector512<int16_t> output_ref(num_elements + 16);
  
  // Create shifted input so we can recover something meaningful
  std::vector<int16_t> original(num_elements + 16);
  std::iota(original.begin(), original.end(), 0);
  for(size_t k=0; k<original.size(); k++) original[k] <<= 4;

  std::copy(original.begin(), original.end(), input.begin());

  int16_t* p_in = input.data() + offset_in;
  int16_t* p_out = output.data() + offset_out;
  int16_t* p_ref = output_ref.data() + offset_out;

  int16_from_msb(p_in, p_out, num_elements);
  int16_from_msb_ref(p_in, p_ref, num_elements);

  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(p_out[i], p_ref[i]) << "Mismatch at index " << i << " with offsets " << offset_in << ", " << offset_out;
  }
}

void test_round_trip(size_t num_elements) {
  AlignedVector512<int16_t> input(num_elements);
  AlignedVector512<int16_t> intermediate(num_elements);
  AlignedVector512<int16_t> output(num_elements);
  
  // Use range that fits in 12 bits signed (-2048 to 2047)
  for (size_t i = 0; i < num_elements; i++) {
     input[i] = (int16_t)((i % 4096) - 2048);
  }

  int16_to_msb(input.data(), intermediate.data(), num_elements);
  int16_from_msb(intermediate.data(), output.data(), num_elements);

  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(input[i], output[i]) << "Round trip mismatch at index " << i;
  }
}

TEST(verify_result, int16_to_msb_alignment_combinations) {
  const size_t n = 1024;
  test_int16_to_msb_case(n, 0, 0);
  test_int16_to_msb_case(n, 0, 1);
  test_int16_to_msb_case(n, 1, 0);
  test_int16_to_msb_case(n, 1, 1);
}

TEST(verify_result, int16_from_msb_alignment_combinations) {
  const size_t n = 1024;
  test_int16_from_msb_case(n, 0, 0);
  test_int16_from_msb_case(n, 0, 1);
  test_int16_from_msb_case(n, 1, 0);
  test_int16_from_msb_case(n, 1, 1);
}

TEST(verify_result, round_trip) {
  test_round_trip(8192);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

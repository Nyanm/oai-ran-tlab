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
#include <gtest/gtest.h>
#include "wraparound_buffer.h"
#include "log.h"
#include <unistd.h>

// Thread safety can be ensured through core affinity - if two actors
// are running on the same core they are thread safe
TEST(wraparound_buffer, basic_logic)
{
  size_t buf_size = getpagesize() * 2;
  void *buffer = malloc_wraparound_buffer(buf_size);
  ASSERT_NE(buffer, nullptr);
  uint8_t *buffptr = static_cast<uint8_t *>(buffer);
  memset(buffer, 0, buf_size);
  buffptr[0] = 1;
  EXPECT_EQ(buffptr[buf_size], 1);
  free_wraparound_buffer(buffer);
}

TEST(wraparound_buffer, full_range)
{
  size_t buf_size = getpagesize() * 2;
  void *buffer = malloc_wraparound_buffer(buf_size);
  ASSERT_NE(buffer, nullptr);
  uint8_t *buffptr = static_cast<uint8_t *>(buffer);
  memset(buffer, 0, buf_size);
  for (auto i = 0U; i < buf_size; i++) {
    buffptr[i] = (uint8_t)i;
    EXPECT_EQ(buffptr[i + buf_size], (uint8_t)i);
  }
  free_wraparound_buffer(buffer);
}

TEST(wraparound_buffer, size)
{
  size_t buf_size = getpagesize() * 2 + 1;
  void *buffer = malloc_wraparound_buffer(buf_size);
  ASSERT_EQ(buffer, nullptr);
}

int main(int argc, char **argv)
{
  logInit();
  g_log->log_component[UTIL].level = OAILOG_DEBUG;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

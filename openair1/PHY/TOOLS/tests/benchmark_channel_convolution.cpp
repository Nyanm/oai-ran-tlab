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

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <numeric>
extern "C" {
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "common/config/config_userapi.h"
struct configmodule_interface_s;
configmodule_interface_t *uniqCfg;
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (assert) {
    abort();
  } else {
    exit(EXIT_SUCCESS);
  }
}
#include "SIMULATION/TOOLS/sim.h"
}
#include <cstdio>
#include "common/utils/LOG/log.h"
#include "benchmark/benchmark.h"
#include "openair1/PHY/TOOLS/phy_test_tools.hpp"

static void BM_channel_convolution_template(benchmark::State &state, void (*func)(channel_desc_t *, c16_t **, c16_t **, uint32_t))
{
  const int nb_tx = 2;
  const int nb_rx = 2;
  const uint32_t length = state.range(0);
  const uint32_t channel_length = 5;
  const uint64_t channel_offset = 10;
  const double path_loss_dB = -3.0;

  // Setup channel descriptor
  channel_desc_t desc;
  memset(&desc, 0, sizeof(channel_desc_t));
  desc.nb_tx = nb_tx;
  desc.nb_rx = nb_rx;
  desc.channel_length = channel_length;
  desc.channel_offset = channel_offset;
  desc.path_loss_dB = path_loss_dB;

  // Allocate channel taps
  std::vector<struct complexd *> ch_ptrs(nb_tx * nb_rx);
  std::vector<std::vector<struct complexd>> ch_data(nb_tx * nb_rx, std::vector<struct complexd>(channel_length));

  desc.ch = ch_ptrs.data();
  for (int i = 0; i < nb_tx * nb_rx; i++) {
    desc.ch[i] = ch_data[i].data();
    for (uint32_t l = 0; l < channel_length; l++) {
      desc.ch[i][l].r = (double)rand() / RAND_MAX;
      desc.ch[i][l].i = (double)rand() / RAND_MAX;
    }
  }

  // Input
  std::vector<c16_t *> input_ptrs(nb_tx);
  std::vector<AlignedVector512<c16_t>> input_data(nb_tx);
  for (int i = 0; i < nb_tx; i++) {
    input_data[i] = generate_random_c16(length + 8);
    input_ptrs[i] = input_data[i].data();
  }

  // Output
  std::vector<c16_t *> output_ptrs(nb_rx);
  std::vector<AlignedVector512<c16_t>> output_data(nb_rx);
  for (int i = 0; i < nb_rx; i++) {
    output_data[i].resize(length);
    output_ptrs[i] = output_data[i].data();
  }

  for (auto _ : state) {
    func(&desc, input_ptrs.data(), output_ptrs.data(), length);
  }
}

static void BM_channel_convolution(benchmark::State &state)
{
  BM_channel_convolution_template(state, channel_convolution);
}

static void BM_channel_convolution_avx2(benchmark::State &state)
{
  BM_channel_convolution_template(state, channel_convolution_avx2);
}

static void BM_channel_convolution_avx512(benchmark::State &state)
{
  BM_channel_convolution_template(state, channel_convolution_avx512);
}

BENCHMARK(BM_channel_convolution)->RangeMultiplier(4)->Range(128, 30000);
BENCHMARK(BM_channel_convolution_avx2)->RangeMultiplier(4)->Range(128, 30000);
BENCHMARK(BM_channel_convolution_avx512)->RangeMultiplier(4)->Range(128, 30000);

BENCHMARK_MAIN();

#include <gtest/gtest.h>
extern "C" {
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "common/utils/LOG/log.h"
#include "SIMULATION/TOOLS/sim.h"
}
#include <cmath>
#include <cstdlib>
#include <vector>

void channel_convolution_ref(channel_desc_t *desc, c16_t **input, c16_t **output, uint32_t length)
{
  float path_loss = powf(10.0f, (float)desc->path_loss_dB / 20.0f);
  uint64_t dd = desc->channel_offset;

  for (int i = 0; i < ((int)length - (int)dd); i++) {
    for (int ii = 0; ii < desc->nb_rx; ii++) {
      cf_t rx_tmp = {0};
      for (int j = 0; j < desc->nb_tx; j++) {
        struct complexd *chan = desc->ch[ii + (j * desc->nb_rx)];
        for (int l = 0; l < (int)desc->channel_length; l++) {
          if ((i >= 0) && (i - l) >= 0) {
            cf_t tx;
            tx.r = input[j][i - l].r;
            tx.i = input[j][i - l].i;
            rx_tmp.r += (tx.r * (float)chan[l].r) - (tx.i * (float)chan[l].i);
            rx_tmp.i += (tx.i * (float)chan[l].r) + (tx.r * (float)chan[l].i);
          }
        } // l
      } // j

      output[ii][i + dd].r = rx_tmp.r * path_loss;
      output[ii][i + dd].i = rx_tmp.i * path_loss;
    } // ii
  } // i
}


TEST(convolve, compare_against_reference)
{
  // Parameters
  const int nb_tx = 2;
  const int nb_rx = 2;
  const uint32_t length = 1000;
  const uint32_t channel_length = 5;
  const uint64_t channel_offset = 10;
  const double path_loss_dB = -3.0;

  // Setup channel descriptor
  channel_desc_t desc;
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
  std::vector<std::vector<c16_t>> input_data(nb_tx, std::vector<c16_t>(length));
  for (int i = 0; i < nb_tx; i++) {
    input_ptrs[i] = input_data[i].data();
    for (uint32_t j = 0; j < length; j++) {
      input_data[i][j].r = rand() % 1000;
      input_data[i][j].i = rand() % 1000;
    }
  }

  // Output
  std::vector<c16_t *> output_ref_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_ref_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  std::vector<c16_t *> output_test_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_test_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  for (int i = 0; i < nb_rx; i++) {
    output_ref_ptrs[i] = output_ref_data[i].data();
    output_test_ptrs[i] = output_test_data[i].data();
  }

  // Run
  channel_convolution_ref(&desc, input_ptrs.data(), output_ref_ptrs.data(), length);
  channel_convolution(&desc, input_ptrs.data(), output_test_ptrs.data(), length);

  // Verify
  for (int i = 0; i < nb_rx; i++) {
    for (uint32_t j = 0; j < length; j++) {
      EXPECT_NEAR(output_ref_data[i][j].r, output_test_data[i][j].r, 1) << "Mismatch at rx " << i << " sample " << j;
      EXPECT_NEAR(output_ref_data[i][j].i, output_test_data[i][j].i, 1) << "Mismatch at rx " << i << " sample " << j;
    }
  }
}

TEST(convolve, compare_simd_against_reference)
{
  // Parameters
  const int nb_tx = 2;
  const int nb_rx = 2;
  const uint32_t length = 1000;
  const uint32_t channel_length = 5;
  const uint64_t channel_offset = 10;
  const double path_loss_dB = -3.0;

  // Setup channel descriptor
  channel_desc_t desc;
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

  // Input - allocate extra padding for SIMD reads
  std::vector<c16_t *> input_ptrs(nb_tx);
  std::vector<std::vector<c16_t>> input_data(nb_tx, std::vector<c16_t>(length + 8));
  for (int i = 0; i < nb_tx; i++) {
    input_ptrs[i] = input_data[i].data();
    for (uint32_t j = 0; j < length; j++) {
      input_data[i][j].r = rand() % 1000;
      input_data[i][j].i = rand() % 1000;
    }
  }

  // Output
  std::vector<c16_t *> output_ref_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_ref_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  std::vector<c16_t *> output_test_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_test_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  for (int i = 0; i < nb_rx; i++) {
    output_ref_ptrs[i] = output_ref_data[i].data();
    output_test_ptrs[i] = output_test_data[i].data();
  }

  // Run
  channel_convolution_ref(&desc, input_ptrs.data(), output_ref_ptrs.data(), length);
  channel_convolution_avx2(&desc, input_ptrs.data(), output_test_ptrs.data(), length);

  // Verify
  for (int i = 0; i < nb_rx; i++) {
    for (uint32_t j = 0; j < length; j++) {
      EXPECT_NEAR(output_ref_data[i][j].r, output_test_data[i][j].r, 1) << "Mismatch at rx " << i << " sample " << j;
      EXPECT_NEAR(output_ref_data[i][j].i, output_test_data[i][j].i, 1) << "Mismatch at rx " << i << " sample " << j;
    }
  }
}

TEST(convolve, compare_avx512_against_reference)
{
  // Parameters
  const int nb_tx = 2;
  const int nb_rx = 2;
  const uint32_t length = 1000;
  const uint32_t channel_length = 5;
  const uint64_t channel_offset = 10;
  const double path_loss_dB = -3.0;

  // Setup channel descriptor
  channel_desc_t desc;
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

  // Input - allocate extra padding for SIMD reads
  std::vector<c16_t *> input_ptrs(nb_tx);
  std::vector<std::vector<c16_t>> input_data(nb_tx, std::vector<c16_t>(length + 8));
  for (int i = 0; i < nb_tx; i++) {
    input_ptrs[i] = input_data[i].data();
    for (uint32_t j = 0; j < length; j++) {
      input_data[i][j].r = rand() % 1000;
      input_data[i][j].i = rand() % 1000;
    }
  }

  // Output
  std::vector<c16_t *> output_ref_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_ref_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  std::vector<c16_t *> output_test_ptrs(nb_rx);
  std::vector<std::vector<c16_t>> output_test_data(nb_rx, std::vector<c16_t>(length, {0, 0}));

  for (int i = 0; i < nb_rx; i++) {
    output_ref_ptrs[i] = output_ref_data[i].data();
    output_test_ptrs[i] = output_test_data[i].data();
  }

  // Run
  channel_convolution_ref(&desc, input_ptrs.data(), output_ref_ptrs.data(), length);
  channel_convolution_avx512(&desc, input_ptrs.data(), output_test_ptrs.data(), length);

  // Verify
  for (int i = 0; i < nb_rx; i++) {
    for (uint32_t j = 0; j < length; j++) {
      EXPECT_NEAR(output_ref_data[i][j].r, output_test_data[i][j].r, 1) << "Mismatch at rx " << i << " sample " << j;
      EXPECT_NEAR(output_ref_data[i][j].i, output_test_data[i][j].i, 1) << "Mismatch at rx " << i << " sample " << j;
    }
  }
}

int main(int argc, char **argv)
{
  logInit();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

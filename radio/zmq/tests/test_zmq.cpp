#include <gtest/gtest.h>
#include "common/config/config_userapi.h"
#include "radio/zmq/zmq.cpp"
#include <zmq.h>
extern "C" {
#include "common/config/config_userapi.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
}
#include "common/platform_types.h"

configmodule_interface_t *uniqCfg = NULL;

extern "C" void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  fprintf(stderr, "FATAL: %s at %s:%s:%d\n", s, file, function, line);
  exit(EXIT_FAILURE);
}

TEST(ZMQ, SendSamples)
{
  char *argv[4];
  argv[0] = strdup("--zmq.tx_channels");
  argv[1] = strdup("tcp://127.0.0.1:5555");
  argv[2] = strdup("--zmq.rx_channels");
  argv[3] = strdup("tcp://127.0.0.1:5556");
  configmodule_interface_t *cfg = load_configmodule(sizeofArray(argv), argv, CONFIG_ENABLECMDLINEONLY);
  uniqCfg = cfg;
  // 2. Initialize the ZMQ device
  openair0_device_t device1;
  openair0_config_t config;
  config.tx_num_channels = 1;
  config.rx_num_channels = 1;
  config.sample_rate = 30;
  ASSERT_EQ(device_init(&device1, &config), 0);
  ASSERT_EQ(device1.trx_start_func(&device1), 0);

  // Swap the RX with TX for second device
  char* tmp = argv[0];
  argv[0] = argv[2];
  argv[2] = tmp;
  configmodule_interface_t *cfg2 = load_configmodule(sizeofArray(argv), argv, CONFIG_ENABLECMDLINEONLY);
  uniqCfg = cfg2;
  openair0_config_t config2;
  config2.tx_num_channels = 1;
  config2.rx_num_channels = 1;
  config2.sample_rate = 30;
  openair0_device_t device2;
  ASSERT_EQ(device_init(&device2, &config2), 0);
  ASSERT_EQ(device2.trx_start_func(&device2), 0);

  c16_t rx_samples[15];
  c16_t *rx_samples_ptr[1];
  rx_samples_ptr[0] = rx_samples;

  openair0_timestamp_t rx_timestamp;
  int ret = device2.trx_read_func(&device2, &rx_timestamp, (void **)rx_samples_ptr, 15, 1);
  ASSERT_EQ(rx_timestamp, 0);
  ASSERT_EQ(ret, 15);
  printf("Read %d samples from device2, timestamp %ld\n", ret, rx_timestamp);
  for (int i = 0; i < 15; i++) {
    ASSERT_EQ(rx_samples[i].r, 0);
    ASSERT_EQ(rx_samples[i].i, 0);
  }
  ret = device1.trx_read_func(&device1, &rx_timestamp, (void **)rx_samples_ptr, 15, 1);
  ASSERT_EQ(rx_timestamp, 0);
  ASSERT_EQ(ret, 15);
  printf("Read %d samples from device1, timestamp %ld\n", ret, rx_timestamp);
  for (int i = 0; i < 15; i++) {
    ASSERT_EQ(rx_samples[i].r, 0);
    ASSERT_EQ(rx_samples[i].i, 0);
  }

  // 3. Send samples
  c16_t samples[15];
  for (int i = 0; i < 15; i++) {
    samples[i].r = i;
    samples[i].i = i + 1;
  }
  c16_t *samples_ptr[1];
  samples_ptr[0] = samples;
  device1.trx_write_func(&device1, 15, (void **)samples_ptr, 15, 1, 0);
  printf("Wrote %d samples to device1\n", 15);

  // 4. Verify received samples are equal to sent samples
  ret = device2.trx_read_func(&device2, &rx_timestamp, (void **)rx_samples_ptr, 15, 1);
  printf("Read %d samples from device2\n", ret);
  ASSERT_EQ(rx_timestamp, 15);
  ASSERT_EQ(ret, 15);
  for (int i = 0; i < 15; i++) {
    ASSERT_EQ(rx_samples[i].r, samples[i].r);
    ASSERT_EQ(rx_samples[i].i, samples[i].i);
  }

  device1.trx_end_func(&device1);
  device2.trx_end_func(&device2);

  end_configmodule(cfg);
  end_configmodule(cfg2);
}

int main(int argc, char **argv)
{
  logInit();
  g_log->log_component[HW].level = OAILOG_DEBUG;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

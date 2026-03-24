/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 * Based on zmq library in OCUDU project:
 * Copyright (c) 2021-2026 Software Radio Systems Limited.
 * Refer to OCUDU_LICENSE file
 */

#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_common.h"
#include "common/platform_types.h"
#include "softmodem-common.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <limits>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/epoll.h>
#include <netdb.h>

#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>
#include <common/config/config_userapi.h>
#include "common_lib.h"
#include <queue>
#include <mutex>
#include <vector>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <ring_buffer.h>
#include <zmq.h>

#define ZMQ_SECTION "zmq"
#define ZMQ_TX_CHANNELS "tx_channels"
#define ZMQ_RX_CHANNELS "rx_channels"

#define ZMQ_PARAMS_DESC                                                                                                           \
  {                                                                                                                               \
      STRINGLISTPARAM(ZMQ_TX_CHANNELS, "list of zmq addresses represeting tx channels\n", PARAMFLAG_MANDATORY, nullptr, nullptr), \
      STRINGLISTPARAM(ZMQ_RX_CHANNELS, "list of zmq addresses represeting rx channels\n", PARAMFLAG_MANDATORY, nullptr, nullptr), \
  };

const size_t sample_size = sizeof(cf_t);
const size_t rx_buffer_size = sample_size * 300000;
const float c16_t_to_cf_t_factor = std::numeric_limits<int16_t>::max();
static constexpr std::chrono::milliseconds TRANSMIT_TS_ALIGN_TIMEOUT = std::chrono::milliseconds(0);
static constexpr std::chrono::milliseconds RECEIVE_TS_ALIGN_TIMEOUT = std::chrono::milliseconds(100);

class zmq_tx_channel {
 public:
  void *socket;
  overflow_buffer buffer;
  std::atomic<uint64_t> sample_count = 0;
  std::atomic<bool> is_tx_enabled = false;
  std::mutex transmit_alignment_mutex;
  std::condition_variable transmit_alignment_cvar;

  zmq_tx_channel(void *s, uint64_t buffer_size) : socket(s), buffer(buffer_size)
  {
  }

  void transmit(c16_t *samples, size_t nsamps, uint64_t timestamp)
  {
    std::scoped_lock lock(transmit_alignment_mutex);
    size_t overflow = 0;
    if (timestamp > sample_count) {
      overflow += buffer.push_zeros(timestamp - sample_count);
      sample_count = timestamp;
    }
    cf_t samples_float[nsamps];
    for (size_t i = 0; i < nsamps; i++) {
      samples_float[i].r = samples[i].r / c16_t_to_cf_t_factor;
      samples_float[i].i = samples[i].i / c16_t_to_cf_t_factor;
    }
    overflow += buffer.push_samples(samples_float, nsamps);
    sample_count += nsamps;
    if (overflow) {
      LOG_W(HW, "Overflow on ZMQ channel by %lu samples\n", overflow);
    }
    is_tx_enabled = true;
    transmit_alignment_cvar.notify_all();
  }

  void start(uint64_t init_time)
  {
    sample_count = init_time;
  }

  bool align(uint64_t timestamp, std::chrono::milliseconds timeout)
  {
    if (sample_count >= timestamp) {
      return sample_count > timestamp;
    }
    std::unique_lock<std::mutex> lock(transmit_alignment_mutex);
    if (is_tx_enabled && (timeout.count() != 0)) {
      bool is_not_timeout =
          transmit_alignment_cvar.wait_for(lock, timeout, [this, timestamp]() { return sample_count >= timestamp; });
      if (is_not_timeout) {
        return sample_count > timestamp;
      }
      LOG_W(HW, "Timeout waiting for TX path to align samples\n");
      is_tx_enabled = false;
    }
    if (sample_count < timestamp) {
      buffer.push_zeros(timestamp - sample_count);
      sample_count = timestamp;
    }
    return false;
  }
};

class zmq_rx_channel {
 public:
  void *socket;
  overflow_buffer buffer;
  bool request_sent;
  std::atomic<bool> stopped;
  zmq_rx_channel(void *s, uint64_t buffer_size) : socket(s), buffer(buffer_size), stopped(false)
  {
  }
  void receive(c16_t *samples, size_t nsamps)
  {
    size_t samples_popped = 0;
    cf_t samples_float[nsamps];
    while (samples_popped < (size_t)nsamps && !stopped) {
      size_t popped_now = buffer.pop_samples(samples_float + samples_popped, nsamps - samples_popped);
      samples_popped += popped_now;
      if (popped_now == 0) {
        usleep(100); // wait for more samples to arrive
      }
    }
    for (size_t i = 0; i < nsamps; i++) {
      samples[i].r = samples_float[i].r * c16_t_to_cf_t_factor + 0.5;
      samples[i].i = samples_float[i].i * c16_t_to_cf_t_factor + 0.5;
    }
  }
  void stop()
  {
    stopped = true;
  }
};

class zmq_tx_stream {
 public:
  std::vector<zmq_tx_channel *> channels;
  void start(uint64_t init_time)
  {
    for (auto &chan : channels) {
      chan->start(init_time);
    }
  }
  bool align(uint64_t timestamp, std::chrono::milliseconds timeout)
  {
    bool timestamp_passed = false;
    for (auto &chan : channels) {
      timestamp_passed = timestamp_passed || chan->align(timestamp, timeout);
    }
    return timestamp_passed;
  }
  void transmit(c16_t **samples, size_t nsamps, uint64_t timestamp)
  {
    bool timestamp_passed = align(timestamp, TRANSMIT_TS_ALIGN_TIMEOUT);
    if (timestamp_passed) {
      LOG_W(HW, "Error, channel timeout\n");
      return;
    }
    int i = 0;
    for (auto chan : channels) {
      chan->transmit(samples[i++], nsamps, timestamp);
    }
  }
};

class zmq_rx_stream {
 public:
  std::vector<zmq_rx_channel *> channels;
  zmq_tx_stream *tx_stream;
  uint64_t sample_count = 0;
  zmq_rx_stream() : sample_count(0)
  {
  }
  void start(uint64_t init_time)
  {
    sample_count = init_time;
  }
  void stop()
  {
    for (auto &chan : channels) {
      chan->stop();
    }
  }
  void receive(c16_t **samples, size_t nsamps, uint64_t *timestamp)
  {
    *timestamp = sample_count;
    uint64_t passed_timestamp = sample_count + nsamps;
    tx_stream->align(passed_timestamp, RECEIVE_TS_ALIGN_TIMEOUT);
    int i = 0;
    for (auto chan : channels) {
      chan->receive(samples[i++], nsamps);
    }
    sample_count += nsamps;
  }
};

typedef struct {
  void *context;
  zmq_tx_stream tx_stream;
  zmq_rx_stream rx_stream;
  std::thread poll_thread;
  std::atomic<bool> poll_thread_running;
  bool stopped = false;
  double sample_rate;
} zmq_state_t;

static void poll_thread(zmq_state_t *s)
{
  s->poll_thread_running = true;
  unsigned char *rx_buffer = static_cast<unsigned char *>(malloc(rx_buffer_size));
  const auto num_tx_channels = s->tx_stream.channels.size();
  const auto num_rx_channels = s->rx_stream.channels.size();
  std::vector<zmq_pollitem_t> items(num_tx_channels + num_rx_channels);
  std::vector<bool> reply_requested(num_tx_channels);
  for (size_t i = 0; i < num_tx_channels; ++i) {
    items[i] = {s->tx_stream.channels[i]->socket, 0, ZMQ_POLLIN, 0};
    // wait for REQ
    reply_requested[i] = false;
  }
  for (size_t i = 0; i < num_rx_channels; i++) {
    items[i + num_tx_channels] = {s->rx_stream.channels[i]->socket, 0, ZMQ_POLLIN, 0};
  }

  const auto num_channels = num_tx_channels + num_rx_channels;
  while (s->poll_thread_running) {
    for (size_t i = 0; i < num_tx_channels; i++) {
      auto chan = s->tx_stream.channels[i];
      if (!reply_requested[i]) {
        continue;
      }
      std::vector<cf_t> samples(1024);
      size_t num_popped = chan->buffer.pop_samples(samples.data(), 1024);
      if (num_popped == 0) {
        continue;
      }
      int rc = zmq_send(chan->socket, samples.data(), num_popped * sizeof(cf_t), 0);
      if (rc < 0) {
        LOG_E(HW, "[ZMQ] poll_thread zmq_send for TX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
      }
      reply_requested[i] = false;
    }

    int rc = zmq_poll(items.data(), num_channels, 10); // 10ms timeout
    if (rc < 0) {
      if (errno == EINTR)
        continue;
      LOG_E(HW, "[ZMQ] poll_thread zmq_poll failed: %s\n", zmq_strerror(errno));
      break;
    }
    if (rc == 0) {
      continue; // timeout
    }

    // --- TX Sockets (ZMQ_REP) ---
    for (size_t i = 0; i < num_tx_channels; i++) {
      if (items[i].revents & ZMQ_POLLIN) {
        auto chan = s->tx_stream.channels[i];
        char dummy;
        rc = zmq_recv(chan->socket, &dummy, 1, 0);
        if (rc < 0) {
          LOG_E(HW, "[ZMQ] poll_thread zmq_recv for TX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
          continue;
        }
        if (reply_requested[i]) {
          LOG_E(HW, "[ZQM] Error, unexpected REQ before REP on TX antenna %d\n", (int)i);
        }
        reply_requested[i] = true;
      }
    }

    // --- RX Sockets (ZMQ_REQ) ---
    for (size_t i = 0; i < num_rx_channels; i++) {
      if (items[i + num_tx_channels].revents & ZMQ_POLLIN) {
        auto chan = s->rx_stream.channels[i];
        rc = zmq_recv(chan->socket, rx_buffer, rx_buffer_size, 0);
        if (rc < 0) {
          LOG_E(HW, "[ZMQ] poll_thread zmq_recv for RX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
        } else {
          size_t received_bytes = rc;
          if (rx_buffer_size < received_bytes) {
            LOG_W(HW,
                  "[ZMQ] the RX buffer is too small! The received message size is %lu while the buffer is %lu. Message truncated\n",
                  received_bytes,
                  rx_buffer_size);
          }
          size_t num_samples_received = std::min(received_bytes, rx_buffer_size) / sizeof(cf_t);
          cf_t *samples = reinterpret_cast<cf_t *>(rx_buffer);
          size_t overflow = chan->buffer.push_samples(samples, num_samples_received);
          if (rx_buffer_size < received_bytes) {
            overflow += chan->buffer.push_zeros((received_bytes - rx_buffer_size) / sizeof(cf_t));
          }
          if (overflow) {
            LOG_W(HW, "Overflow on receive\n");
          }
          // After receiving, send next request to keep the stream flowing
          char dummy = 0;
          if (zmq_send(chan->socket, &dummy, 1, 0) != 1) {
            LOG_E(HW, "[ZMQ] poll_thread zmq_send for RX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
          }
        }
      }
    }
  }
  free(rx_buffer);
}

static int zmq_write(openair0_device_t *device, openair0_timestamp_t timestamp, void **buff, int nsamps, int cc, int flags)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  AssertFatal((uint)cc == s->tx_stream.channels.size(),
              "Request to write on more antennas (%d) than configured (%d)",
              cc,
              (int)s->tx_stream.channels.size());

  s->tx_stream.transmit((c16_t **)buff, nsamps, timestamp);

  return nsamps;
}

static int zmq_read(openair0_device_t *device, openair0_timestamp_t *ptimestamp, void **samplesVoid, int nsamps, int nbAnt)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  AssertFatal((uint)nbAnt == s->rx_stream.channels.size(),
              "Request to read on more antennas (%d) than configured (%d)",
              nbAnt,
              (int)s->rx_stream.channels.size());
  uint64_t timestamp;
  s->rx_stream.receive((c16_t **)samplesVoid, nsamps, &timestamp);
  *ptimestamp = timestamp;
  return nsamps;
}

static int zmq_get_stats(openair0_device_t *device)
{
  return 0;
}
static int zmq_reset_stats(openair0_device_t *device)
{
  return 0;
}
static void zmq_end(openair0_device_t *device)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  if (s) {
    if (s->poll_thread_running) {
      s->poll_thread_running = false;
      if (s->poll_thread.joinable()) {
        s->poll_thread.join();
      }
    }
    for (auto &chan : s->tx_stream.channels) {
      if (chan->socket)
        zmq_close(chan->socket);
      delete chan;
    }
    s->tx_stream.channels.clear();

    for (auto &chan : s->rx_stream.channels) {
      if (chan->socket)
        zmq_close(chan->socket);
      delete chan;
    }
    s->rx_stream.channels.clear();

    if (s->context)
      zmq_ctx_destroy(s->context);
    delete s;
  }
}

static int zmq_start(openair0_device_t *device)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  s->rx_stream.start(s->sample_rate / 100);
  s->tx_stream.start(s->sample_rate / 100);
  for (size_t i = 0; i < s->rx_stream.channels.size(); i++) {
    auto channel = s->rx_stream.channels[i];
    // Send initial request to start data flow
    char dummy = 0;
    if (zmq_send(channel->socket, &dummy, 1, 0) != 1) {
      LOG_E(HW, "[ZMQ] zmq_send for initial RX request failed for antenna %lu: %s\n", i, zmq_strerror(errno));
      return -1;
    }
  }
  s->poll_thread = std::thread(poll_thread, s);
  return 0;
}

static int zmq_stop(openair0_device_t *device)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  s->rx_stream.stop();
  return 0;
}

static int zmq_set_freq(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  return 0;
}
static int zmq_set_gains(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  return 0;
}
static int zmq_write_init(openair0_device_t *device)
{
  return 0;
}

extern "C" __attribute__((__visibility__("default"))) int device_init(openair0_device_t *device, openair0_config_t *openair0_cfg)
{
  auto *zmq_state = new zmq_state_t();
  zmq_state->context = zmq_ctx_new();
  AssertFatal(zmq_state->context != NULL, "zmq_ctx_new failed");

  LOG_I(HW, "[ZMQ] tx_antennas: %d, rx_antennas: %d\n", openair0_cfg->tx_num_channels, openair0_cfg->rx_num_channels);
  configmodule_interface_t *cfg = config_get_if();
  paramdef_t param_desc[] = ZMQ_PARAMS_DESC;
  std::string zmq_section = std::string(ZMQ_SECTION);
  bool is_list = config_is_list(cfg, zmq_section.c_str());
  int ret;
  if (!is_list) {
    ret = config_get(cfg, param_desc, sizeofArray(param_desc), zmq_section.c_str());
  } else {
    int ru_id = openair0_cfg->ru_id;
    std::string zmq_array_section = std::string(ZMQ_SECTION) + ".[" + std::to_string(ru_id) + "]";
    ret = config_get(cfg, param_desc, sizeofArray(param_desc), zmq_array_section.c_str());
  }
  AssertFatal(ret >= 0, "configuration couldn't be performed\n");
  AssertFatal(gpd(param_desc, sizeofArray(param_desc), ZMQ_TX_CHANNELS)->numelt == openair0_cfg->tx_num_channels,
              "Incorrect configuration: Number of tx channels expected to be %d\n",
              openair0_cfg->tx_num_channels);
  AssertFatal(gpd(param_desc, sizeofArray(param_desc), ZMQ_RX_CHANNELS)->numelt == openair0_cfg->rx_num_channels,
              "Incorrect configuration: Number of rx channels expected to be %d\n",
              openair0_cfg->rx_num_channels);
  char **tx_channels = gpd(param_desc, sizeofArray(param_desc), ZMQ_TX_CHANNELS)->strlistptr;
  char **rx_channels = gpd(param_desc, sizeofArray(param_desc), ZMQ_RX_CHANNELS)->strlistptr;

  // Setup TX sockets (one per antenna)
  if (openair0_cfg->tx_num_channels > 0) {
    zmq_state->tx_stream.channels.resize(openair0_cfg->tx_num_channels);
    for (int i = 0; i < openair0_cfg->tx_num_channels; i++) {
      void *socket = zmq_socket(zmq_state->context, ZMQ_REP);
      AssertFatal(socket != NULL, "zmq_socket(ZMQ_REP) for TX antenna %d failed", i);
      int linger = 0;
      zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
      AssertFatal(zmq_bind(socket, tx_channels[i]) == 0, "zmq_bind for TX antenna %d failed on %s", i, tx_channels[i]);
      auto channel = new zmq_tx_channel(socket, openair0_cfg->sample_rate);
      LOG_I(HW, "[ZMQ] TX socket for antenna %d bound to %s\n", i, tx_channels[i]);
      zmq_state->tx_stream.channels[i] = channel;
    }
  }
  zmq_state->sample_rate = openair0_cfg->sample_rate;

  // Setup RX sockets (one per antenna)
  if (openair0_cfg->rx_num_channels > 0) {
    zmq_state->rx_stream.channels.resize(openair0_cfg->rx_num_channels);
    for (int i = 0; i < openair0_cfg->rx_num_channels; i++) {
      void *socket = zmq_socket(zmq_state->context, ZMQ_REQ);
      AssertFatal(socket != NULL, "zmq_socket(ZMQ_REQ) for RX antenna %d failed", i);
      int linger = 0;
      zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
      AssertFatal(zmq_connect(socket, rx_channels[i]) == 0, "zmq_connect for RX antenna %d failed on %s", i, rx_channels[i]);
      auto channel = new zmq_rx_channel(socket, openair0_cfg->sample_rate);
      LOG_I(HW, "[ZMQ] RX socket for antenna %d connected to %s\n", i, rx_channels[i]);
      zmq_state->rx_stream.channels[i] = channel;
    }
    zmq_state->rx_stream.tx_stream = &zmq_state->tx_stream;
  }

  device->trx_start_func = zmq_start;
  device->trx_get_stats_func = zmq_get_stats;
  device->trx_reset_stats_func = zmq_reset_stats;
  device->trx_end_func = zmq_end;
  device->trx_stop_func = zmq_stop;
  device->trx_set_freq_func = zmq_set_freq;
  device->trx_set_gains_func = zmq_set_gains;
  device->trx_write_func = zmq_write;
  device->trx_read_func = zmq_read;
  device->type = RFSIMULATOR;
  IS_SOFTMODEM_RFSIM = 1U;
  openair0_cfg->rx_gain[0] = 0;
  device->openair0_cfg = openair0_cfg;
  device->priv = zmq_state;
  device->trx_write_init = zmq_write_init;

  return 0;
}

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
 * Author and copyright: Laurent Thomas, open-cells.com
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

/*
 * Open issues and limitations
 * The read and write should be called in the same thread, that is not new USRP UHD design
 * When the opposite side switch from passive reading to active R+Write, the synchro is not fully deterministic
 */

#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_common.h"
#include "utils.h"
#include <cstdint>
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
#include <circular_buffer.h>
#include <zmq.h>

#define ZMQ_SECTION "zmq"
#define ZMQ_TX_CHANNELS "tx_channels"
#define ZMQ_RX_CHANNELS "rx_channels"

#define ZMQ_PARAMS_DESC                                                                                                           \
  {                                                                                                                               \
      STRINGLISTPARAM(ZMQ_TX_CHANNELS, "list of zmq addresses represeting tx channels\n", PARAMFLAG_MANDATORY, nullptr, nullptr), \
      STRINGLISTPARAM(ZMQ_RX_CHANNELS, "list of zmq addresses represeting rx channels\n", PARAMFLAG_MANDATORY, nullptr, nullptr), \
  };

typedef c16_t sample_t; // 2*16 bits complex number

class zmq_channel {
 public:
  void *socket;
  overflow_buffer buffer;
  zmq_channel(void *s, uint64_t buffer_size, uint64_t prefill_size) : socket(s), buffer(buffer_size, prefill_size)
  {
  }
};

typedef struct {
  std::vector<zmq_channel *> channels;
} zmq_streamer;

typedef struct {
  void *context;
  zmq_streamer tx_streamer;
  zmq_streamer rx_streamer;
  std::thread poll_thread;
  std::atomic<bool> poll_thread_running;
  int64_t last_tx_sample;
  int64_t last_rx_sample;
} zmq_state_t;

static void poll_thread(zmq_state_t *s)
{
  s->poll_thread_running = true;
  const auto num_tx_channels = s->tx_streamer.channels.size();
  const auto num_rx_channels = s->rx_streamer.channels.size();
  std::vector<zmq_pollitem_t> items(num_tx_channels + num_rx_channels);
  std::vector<bool> reply_requested(num_tx_channels);
  for (size_t i = 0; i < num_tx_channels; ++i) {
    items[i] = {s->tx_streamer.channels[i]->socket, 0, ZMQ_POLLIN, 0};
    // wait for REQ
    reply_requested[i] = false;
  }
  for (size_t i = 0; i < num_rx_channels; i++) {
    items[i + num_tx_channels] = {s->rx_streamer.channels[i]->socket, 0, ZMQ_POLLIN, 0};
  }

  const auto num_channels = num_tx_channels + num_rx_channels;
  while (s->poll_thread_running) {
    for (size_t i = 0; i < num_tx_channels; i++) {
      auto chan = s->tx_streamer.channels[i];
      if (!reply_requested[i]) {
        continue;
      }
      std::vector<c16_t> samples(1024);
      size_t num_popped = chan->buffer.pop_samples(samples.data(), 1024);
      if (num_popped == 0) {
        continue;
      }
      std::vector<cf_t> samples_float(num_popped);
      for (size_t j = 0; j < num_popped; j++) {
        samples_float[j] = cf_t{(float)samples[j].r, (float)samples[j].i};
      }
      int rc = zmq_send(chan->socket, samples_float.data(), num_popped * sizeof(cf_t), 0);
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
        auto chan = s->tx_streamer.channels[i];
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
        auto chan = s->rx_streamer.channels[i];
        unsigned char buffer[8192];
        rc = zmq_recv(chan->socket, buffer, sizeof(buffer), 0);
        if (rc < 0) {
          LOG_E(HW, "[ZMQ] poll_thread zmq_recv for RX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
        } else {
          size_t num_samples_received = rc / sizeof(cf_t);
          cf_t *samples_float = reinterpret_cast<cf_t *>(buffer);
          std::vector<c16_t> samples(num_samples_received);
          for (size_t j = 0; j < num_samples_received; j++) {
            samples[j] = c16_t{(int16_t)samples_float[j].r, (int16_t)samples_float[j].i};
          }
          chan->buffer.push_samples(samples.data(), num_samples_received);
          // After receiving, send next request to keep the stream flowing
          char dummy = 0;
          if (zmq_send(chan->socket, &dummy, 1, 0) != 1) {
            LOG_E(HW, "[ZMQ] poll_thread zmq_send for RX antenna %d failed: %s\n", (int)i, zmq_strerror(errno));
          }
        }
      }
    }
  }
}

static int zmq_write(openair0_device_t *device, openair0_timestamp_t timestamp, void **buff, int nsamps, int cc, int flags)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  AssertFatal((uint)cc <= s->tx_streamer.channels.size(),
              "Request to write on more antennas (%d) than configured (%d)",
              cc,
              (int)s->tx_streamer.channels.size());

  if (timestamp < s->last_tx_sample) {
    LOG_W(HW,
          "Detected out-of-order transmission, abort IQ transfer last_tx_sample %ld, timestamp %ld\n",
          s->last_tx_sample,
          timestamp);
    return nsamps;
  }
  int64_t gap = timestamp - s->last_tx_sample;
  if (gap > 1) {
    LOG_I(HW, "[ZMQ] Timestamp gap %ld on TX\n", gap);
    for (int i = 0; i < cc; i++) {
      auto chan = s->tx_streamer.channels[i];
      chan->buffer.push_zeros(gap);
    }
  }
  for (int i = 0; i < cc; i++) {
    auto chan = s->tx_streamer.channels[i];
    chan->buffer.push_samples((c16_t *)buff[i], nsamps);
  }

  s->last_tx_sample = timestamp + nsamps;

  return nsamps;
}

static int zmq_read(openair0_device_t *device, openair0_timestamp_t *ptimestamp, void **samplesVoid, int nsamps, int nbAnt)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  AssertFatal((uint)nbAnt <= s->rx_streamer.channels.size(),
              "Request to read on more antennas (%d) than configured (%d)",
              nbAnt,
              (int)s->rx_streamer.channels.size());

  for (int i = 0; i < nbAnt; i++) {
    auto chan = s->rx_streamer.channels[i];
    size_t samples_popped = 0;
    while (samples_popped < (size_t)nsamps) {
      size_t popped_now = chan->buffer.pop_samples(&((c16_t *)samplesVoid[i])[samples_popped], nsamps - samples_popped);
      samples_popped += popped_now;
      if (popped_now == 0 && samples_popped < (size_t)nsamps) {
        usleep(100); // wait for more samples to arrive
      }
    }
  }

  *ptimestamp = s->last_rx_sample;
  s->last_rx_sample += nsamps;

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
    for (auto &chan : s->tx_streamer.channels) {
      if (chan->socket)
        zmq_close(chan->socket);
      delete chan;
    }
    s->tx_streamer.channels.clear();

    for (auto &chan : s->rx_streamer.channels) {
      if (chan->socket)
        zmq_close(chan->socket);
      delete chan;
    }
    s->rx_streamer.channels.clear();

    if (s->context)
      zmq_ctx_destroy(s->context);
    delete s;
  }
}

static int zmq_start(openair0_device_t *device)
{
  zmq_state_t *s = static_cast<zmq_state_t *>(device->priv);
  for (size_t i = 0; i < s->rx_streamer.channels.size(); i++) {
    auto channel = s->rx_streamer.channels[i];
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

  LOG_I(HW, "[ZMQ] Initializing ZMQ device for per-antenna sockets\n");
  zmq_state->context = zmq_ctx_new();
  AssertFatal(zmq_state->context != NULL, "zmq_ctx_new failed");

  LOG_I(HW, "[ZMQ] tx_antennas: %d, rx_antennas: %d\n", openair0_cfg->tx_num_channels, openair0_cfg->rx_num_channels);
  configmodule_interface_t *cfg = config_get_if();
  paramdef_t param_desc[] = ZMQ_PARAMS_DESC;
  int ret = config_get(cfg, param_desc, sizeofArray(param_desc), ZMQ_SECTION);
  AssertFatal(ret >= 0, "configuration couldn't be performed\n");
  AssertFatal(gpd(param_desc, sizeofArray(param_desc), ZMQ_TX_CHANNELS)->numelt == openair0_cfg->tx_num_channels,
              "Incorrect configuration: Number of tx channels expected to be %d\n",
              openair0_cfg->tx_num_channels);
  AssertFatal(gpd(param_desc, sizeofArray(param_desc), ZMQ_RX_CHANNELS)->numelt == openair0_cfg->rx_num_channels,
              "Incorrect configuration: Number of rx channels expected to be %d\n",
              openair0_cfg->tx_num_channels);
  char **tx_channels = gpd(param_desc, sizeofArray(param_desc), ZMQ_TX_CHANNELS)->strlistptr;
  char **rx_channels = gpd(param_desc, sizeofArray(param_desc), ZMQ_RX_CHANNELS)->strlistptr;

  // Setup TX sockets (one per antenna)
  if (openair0_cfg->tx_num_channels > 0) {
    zmq_state->tx_streamer.channels.resize(openair0_cfg->tx_num_channels);
    for (int i = 0; i < openair0_cfg->tx_num_channels; i++) {
      void *socket = zmq_socket(zmq_state->context, ZMQ_REP);
      AssertFatal(socket != NULL, "zmq_socket(ZMQ_REP) for TX antenna %d failed", i);
      AssertFatal(zmq_bind(socket, tx_channels[i]) == 0, "zmq_bind for TX antenna %d failed on %s", i, tx_channels[i]);
      auto channel = new zmq_channel(socket, openair0_cfg->sample_rate, openair0_cfg->sample_rate / 2);
      LOG_I(HW, "[ZMQ] TX socket for antenna %d bound to %s\n", i, tx_channels[i]);
      zmq_state->tx_streamer.channels[i] = channel;
    }
  }
  zmq_state->last_tx_sample = openair0_cfg->sample_rate / 2;

  // Setup RX sockets (one per antenna)
  if (openair0_cfg->rx_num_channels > 0) {
    zmq_state->rx_streamer.channels.resize(openair0_cfg->rx_num_channels);
    for (int i = 0; i < openair0_cfg->rx_num_channels; i++) {
      void *socket = zmq_socket(zmq_state->context, ZMQ_REQ);
      AssertFatal(socket != NULL, "zmq_socket(ZMQ_REQ) for RX antenna %d failed", i);
      AssertFatal(zmq_connect(socket, rx_channels[i]) == 0, "zmq_connect for RX antenna %d failed on %s", i, rx_channels[i]);
      auto channel = new zmq_channel(socket, openair0_cfg->sample_rate, 0);
      LOG_I(HW, "[ZMQ] RX socket for antenna %d connected to %s\n", i, rx_channels[i]);
      zmq_state->rx_streamer.channels[i] = channel;
    }
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
  openair0_cfg->rx_gain[0] = 0;
  device->openair0_cfg = openair0_cfg;
  device->priv = zmq_state;
  device->trx_write_init = zmq_write_init;

  return 0;
}

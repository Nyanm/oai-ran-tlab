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

#include "common_lib.h"
#include "common/utils/LOG/log.h"

#include "zmq_threads.h"

typedef struct {
  int samplerate;
  zmq_thread_t *rx_thread;
  zmq_thread_t *tx_thread;
  circular_buffer_t *rx_cb;
  circular_buffer_t *tx_cb;
  int rx_ts;
} zmq_t;

static int trx_zmq_get_stats(openair0_device *device)
{
  LOG_I(HW, "trx_zmq_get_stats() called, not implemented\n");
  return 0;
}

static int trx_zmq_reset_stats(openair0_device *device)
{
  LOG_I(HW, "trx_zmq_reset_stats() called, not implemented\n");
  return 0;
}

static int trx_zmq_stop(openair0_device *device)
{
  LOG_I(HW, "trx_zmq_stop() called, not implemented\n");
  return 0;
}

static int trx_zmq_set_freq(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "trx_zmq_set_freq() called, not implemented\n");
  return 0;
}

static int trx_zmq_set_gains(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "trx_zmq_set_gains() called, not implemented\n");
  return 0;
}

static int trx_zmq_start(openair0_device *device)
{
  return 0;
}

static void trx_zmq_end(openair0_device *device)
{
}

static int trx_zmq_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  DevAssert(cc == 1);
  zmq_t *z = device->priv;
  short *in = buff[0];
//printf("writing %d samples\n", nsamps); fflush(stdout);
  circular_buffer_write(z->rx_cb, in, nsamps, timestamp);
//printf("writing of %d samples done\n", nsamps);
  return nsamps;
}

static int trx_zmq_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
  DevAssert(cc == 1);
  zmq_t *z = device->priv;
  short *out = buff[0];
//printf("wanting %d samples\n", nsamps); fflush(stdout);
  circular_buffer_read(z->rx_cb, out, nsamps, z->rx_ts);
  *ptimestamp = z->rx_ts;
  z->rx_ts += nsamps;
//printf("got %d samples\n", nsamps); fflush(stdout);
  return nsamps;
}

int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {
  device->openair0_cfg = openair0_cfg;
  device->trx_start_func = trx_zmq_start;
  device->trx_get_stats_func = trx_zmq_get_stats;
  device->trx_reset_stats_func = trx_zmq_reset_stats;
  device->trx_end_func = trx_zmq_end;
  device->trx_stop_func = trx_zmq_stop;
  device->trx_set_freq_func = trx_zmq_set_freq;
  device->trx_set_gains_func = trx_zmq_set_gains;
  device->trx_write_func = trx_zmq_write;
  device->trx_read_func  = trx_zmq_read;
  device->type = 3; /* USRP N300 */

  zmq_t *z = calloc_or_fail(1, sizeof(*z));
  device->priv = z;

  zmq_configuration_t *s = openair0_cfg->zmq_conf;

  z->samplerate = openair0_cfg->sample_rate;

  /* todo: config */
  int buffer_size = 1024 * 1024;
  double speed = 0.1;

  z->rx_cb = new_circular_buffer(buffer_size, z->samplerate, speed);
  z->tx_cb = new_circular_buffer(buffer_size, z->samplerate, speed);
  z->rx_thread = zmq_start_rx_thread(s->rx_address, s->rx_mode, z->rx_cb);
  z->tx_thread = zmq_start_tx_thread(s->tx_address, s->tx_mode, z->tx_cb);

  return 0;
}

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

#include "shm_td_iq_channel.h"
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "assertions.h"
#include "utils.h"
#include "common/utils/threadPool/pthread_utils.h"

#define CIRCULAR_BUFFER_SIZE (30720 * 14 * 20)
// Buffer prefix is a copy of the ending of the buffer to the beginning to
// allow continuous read up to this size without wrapping when doing channel modelling
#define BUFFER_PREFIX_SIZE 30720

typedef struct {
  uint64_t timestamp;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} sync_data_t;

typedef struct {
  int magic;
  int num_antennas_tx;
  int num_antennas_rx;
  sync_data_t sync_data;

  bool client_sync;
  sync_data_t client_sync_data;
} ShmTDIQChannelData;

typedef struct ShmTDIQChannel_s {
  IQChannelType type;
  ShmTDIQChannelData *data;
  char name[256];
  sample_t *tx_iq_data;
  sample_t *rx_iq_data;
  int num_antennas_tx;
  int num_antennas_rx;
  bool abort;
} ShmTDIQChannel;

static void init_sync_data(sync_data_t *sync_data)
{
  sync_data->timestamp = 0;
  pthread_mutexattr_t mutex_attr;
  pthread_condattr_t cond_attr;
  int ret = pthread_mutexattr_init(&mutex_attr);
  AssertFatal(ret == 0, "pthread_mutexattr_init() failed: errno %d, %s\n", errno, strerror(errno));

  ret = pthread_condattr_init(&cond_attr);
  AssertFatal(ret == 0, "pthread_condattr_init() failed: errno %d, %s\n", errno, strerror(errno));

  ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
  AssertFatal(ret == 0, "pthread_mutexattr_setpshared() failed: errno %d, %s\n", errno, strerror(errno));

  ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
  AssertFatal(ret == 0, "pthread_condattr_setpshared() failed: errno %d, %s\n", errno, strerror(errno));

  ret = pthread_mutex_init(&sync_data->mutex, &mutex_attr);
  AssertFatal(ret == 0, "pthread_mutex_init() failed: errno %d, %s\n", errno, strerror(errno));

  ret = pthread_cond_init(&sync_data->cond, &cond_attr);
  AssertFatal(ret == 0, "pthread_cond_init() failed: errno %d, %s\n", errno, strerror(errno));
}

static void wait_for_sync(sync_data_t *sync_data, uint64_t timestamp, bool *should_abort)
{
  mutexlock(sync_data->mutex);
  while (sync_data->timestamp < timestamp && !*should_abort) {
    condwait(sync_data->cond, sync_data->mutex);
  }
  mutexunlock(sync_data->mutex);
}

static int wait_for_sync_with_timeout(sync_data_t *sync_data, uint64_t timestamp, uint64_t timeout_uS, bool *should_abort)
{
  struct timespec ts = {.tv_sec = 0, .tv_nsec = 0};
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    fprintf(stderr, "Error: clock_gettime failed: %s\n", strerror(errno));
    return 1;
  }

  ts.tv_sec += timeout_uS / 1000000; // Convert microseconds to seconds
  ts.tv_nsec += (timeout_uS % 1000000) * 1000; // Convert remaining microseconds to nanoseconds

  mutexlock(sync_data->mutex);
  while (sync_data->timestamp < timestamp && !*should_abort) {
    int ret = pthread_cond_timedwait(&sync_data->cond, &sync_data->mutex, &ts);
    if (ret == ETIMEDOUT) {
      fprintf(stderr, "Error: Timed out waiting for samples.\n");
      mutexunlock(sync_data->mutex);
      return 1;
    } else if (ret != 0) {
      fprintf(stderr, "Error: pthread_cond_timedwait failed: %s\n", strerror(ret));
      mutexunlock(sync_data->mutex);
      return 1;
    }
  }
  mutexunlock(sync_data->mutex);
  return 0;
}

static void update_timestamp(sync_data_t *sync_data, uint64_t timestamp)
{
  mutexlock(sync_data->mutex);
  sync_data->timestamp = timestamp;
  condbroadcast(sync_data->cond);
  mutexunlock(sync_data->mutex);
}

ShmTDIQChannel *shm_td_iq_channel_create(const char *name, int num_tx_ant, int num_rx_ant, bool client_sync)
{
  AssertFatal(num_tx_ant > 0, "Number of TX antennas must be greater than 0\n");
  AssertFatal(num_rx_ant > 0, "Number of RX antennas must be greater than 0\n");
  // Create shared memory segment
  int fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  AssertFatal(fd != -1, "shm_open failed: %s\n", strerror(errno));
  size_t tx_buffer_size = (CIRCULAR_BUFFER_SIZE + BUFFER_PREFIX_SIZE) * sizeof(sample_t) * num_tx_ant;
  size_t rx_buffer_size = (CIRCULAR_BUFFER_SIZE + BUFFER_PREFIX_SIZE) * sizeof(sample_t) * num_rx_ant;
  size_t total_size = sizeof(ShmTDIQChannelData) + tx_buffer_size + rx_buffer_size;

  // Set the size of the shared memory segment
  int res = ftruncate(fd, total_size);
  AssertFatal(res != -1, "ftruncate failed: %s\n", strerror(errno));

  // Map shared memory segment to address space
  ShmTDIQChannelData *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  AssertFatal(shm_ptr != MAP_FAILED, "mmap failed: %s\n", strerror(errno));

  // Initialize shared memory
  memset(shm_ptr, 0, total_size);
  shm_ptr->num_antennas_tx = num_tx_ant;
  shm_ptr->num_antennas_rx = num_rx_ant;
  ShmTDIQChannel *channel = calloc_or_fail(1, sizeof(ShmTDIQChannel));
  strncpy(channel->name, name, sizeof(channel->name) - 1);
  channel->tx_iq_data = (sample_t *)(shm_ptr + 1);
  channel->rx_iq_data = channel->tx_iq_data + tx_buffer_size / sizeof(sample_t);
  channel->num_antennas_tx = num_tx_ant;
  channel->num_antennas_rx = num_rx_ant;
  channel->data = shm_ptr;
  channel->type = IQ_CHANNEL_TYPE_SERVER;

  init_sync_data(&shm_ptr->sync_data);

  shm_ptr->magic = SHM_MAGIC_NUMBER;

  if (client_sync) {
    shm_ptr->client_sync = true;
    init_sync_data(&shm_ptr->client_sync_data);
  }
  close(fd);
  return channel;
}

ShmTDIQChannel *shm_td_iq_channel_connect(const char *name, int timeout_in_seconds)
{
  // Create shared memory segment
  int fd = -1;
  while (timeout_in_seconds > 0 && fd == -1) {
    fd = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    timeout_in_seconds--;
    printf("Waiting for server to create shared memory segment\n");
    sleep(1);
  }
  AssertFatal(fd != -1, "shm_open() failed: errno %d, %s", errno, strerror(errno));

  struct stat buf;
  fstat(fd, &buf);
  size_t total_size = buf.st_size;

  // Map shared memory segment to address space
  ShmTDIQChannelData *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shm_ptr == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }

  ShmTDIQChannel *channel = calloc_or_fail(1, sizeof(ShmTDIQChannel));
  channel->data = shm_ptr;
  channel->rx_iq_data = (sample_t *)(shm_ptr + 1);
  size_t tx_buffer_size = (CIRCULAR_BUFFER_SIZE + BUFFER_PREFIX_SIZE) * sizeof(sample_t) * channel->data->num_antennas_tx;
  channel->tx_iq_data = channel->rx_iq_data + tx_buffer_size / sizeof(sample_t);
  channel->type = IQ_CHANNEL_TYPE_CLIENT;
  channel->num_antennas_rx = shm_ptr->num_antennas_tx;
  channel->num_antennas_tx = shm_ptr->num_antennas_rx;
  while (shm_ptr->magic != SHM_MAGIC_NUMBER) {
    printf("Waiting for server to initialize shared memory\n");
    sleep(1);
  }
  close(fd);
  return channel;
}

int shm_td_iq_channel_get_nb_antennas_tx(ShmTDIQChannel *channel)
{
  return channel->num_antennas_tx;
}

int shm_td_iq_channel_get_nb_antennas_rx(ShmTDIQChannel *channel)
{
  return channel->num_antennas_rx;
}

static sample_t *get_prefix_buffer_ptr(sample_t *base_ptr, int antenna)
{
  return base_ptr + antenna * (CIRCULAR_BUFFER_SIZE + BUFFER_PREFIX_SIZE);
}

static sample_t *get_main_buffer_ptr(sample_t *base_ptr, int antenna)
{
  return get_prefix_buffer_ptr(base_ptr, antenna) + BUFFER_PREFIX_SIZE;
}

IQChannelErrorType shm_td_iq_channel_tx(ShmTDIQChannel *channel,
                                        uint64_t timestamp,
                                        uint64_t num_samples,
                                        int antenna,
                                        const sample_t *tx_iq_data)
{
  AssertFatal(antenna < channel->num_antennas_tx,
              "Antenna index %d out of range (num antennas: %d)\n",
              antenna,
              channel->num_antennas_tx);
  ShmTDIQChannelData *data = channel->data;
  // timestamp in the past
  uint64_t current_time = data->sync_data.timestamp;
  if (timestamp < current_time) {
    return CHANNEL_ERROR_TOO_LATE;
  }
  // timestamp is too far in the future
  if (timestamp - current_time + num_samples >= CIRCULAR_BUFFER_SIZE) {
    return CHANNEL_ERROR_TOO_EARLY;
  }

  sample_t *base_ptr = get_main_buffer_ptr(channel->tx_iq_data, antenna);
  uint64_t first_sample = timestamp % CIRCULAR_BUFFER_SIZE;
  uint64_t last_sample = first_sample + num_samples - 1;
  if (last_sample >= CIRCULAR_BUFFER_SIZE) {
    size_t num_samples_first_copy = CIRCULAR_BUFFER_SIZE - first_sample;
    memcpy(base_ptr + first_sample, tx_iq_data, num_samples_first_copy * sizeof(sample_t));
    memcpy(base_ptr, tx_iq_data + num_samples_first_copy, (num_samples - num_samples_first_copy) * sizeof(sample_t));
  } else {
    memcpy(base_ptr + first_sample, tx_iq_data, num_samples * sizeof(sample_t));
  }

  // Mirror end of buffer to prefix buffer for continuous zero copy reading
  int64_t mirror_start = CIRCULAR_BUFFER_SIZE - BUFFER_PREFIX_SIZE;
  int64_t mirror_end = CIRCULAR_BUFFER_SIZE;
  int64_t tx_end = (first_sample + num_samples);
  int64_t tx_start = first_sample;
  int64_t overlap_start = (tx_start > mirror_start) ? tx_start : mirror_start;
  int64_t overlap_end = (tx_end < mirror_end) ? tx_end : mirror_end;
  if (overlap_end - overlap_start > 0) {
    sample_t *prefix_ptr = get_prefix_buffer_ptr(channel->tx_iq_data, antenna);
    memcpy(prefix_ptr + (overlap_start - mirror_start), base_ptr + overlap_start, (overlap_end - overlap_start) * sizeof(sample_t));
  }

  if (channel->type == IQ_CHANNEL_TYPE_CLIENT && data->client_sync) {
    update_timestamp(&data->client_sync_data, timestamp + num_samples);
  }

  return CHANNEL_NO_ERROR;
}

IQChannelErrorType shm_td_iq_channel_rx(ShmTDIQChannel *channel,
                                        uint64_t timestamp,
                                        uint64_t num_samples,
                                        int antenna,
                                        sample_t *rx_iq_data)
{
  AssertFatal(antenna < channel->num_antennas_rx,
              "Antenna index %d out of range (num antennas: %d)\n",
              antenna,
              channel->num_antennas_rx);
  ShmTDIQChannelData *data = channel->data;
  // timestamp in the future
  uint64_t current_time = data->sync_data.timestamp;
  if (timestamp > current_time) {
    return CHANNEL_ERROR_TOO_EARLY;
  }
  // timestamp is too far in the past
  if (current_time - timestamp >= CIRCULAR_BUFFER_SIZE) {
    return CHANNEL_ERROR_TOO_LATE;
  }

  sample_t *base_ptr = get_main_buffer_ptr(channel->rx_iq_data, antenna);

  uint64_t first_sample = timestamp % CIRCULAR_BUFFER_SIZE;
  uint64_t last_sample = first_sample + num_samples - 1;
  if (last_sample >= CIRCULAR_BUFFER_SIZE) {
    size_t num_samples_first_copy = CIRCULAR_BUFFER_SIZE - first_sample;
    memcpy(rx_iq_data, base_ptr + first_sample, num_samples_first_copy * sizeof(sample_t));
    memcpy(rx_iq_data + num_samples_first_copy, base_ptr, (num_samples - num_samples_first_copy) * sizeof(sample_t));
  } else {
    memcpy(rx_iq_data, base_ptr + first_sample, num_samples * sizeof(sample_t));
  }
  return CHANNEL_NO_ERROR;
}

IQChannelErrorType shm_td_iq_channel_zc_rx(ShmTDIQChannel *channel,
                                           uint64_t timestamp,
                                           uint64_t num_samples,
                                           int antenna,
                                           sample_t **rx_iq_data)
{
  AssertFatal(num_samples <= BUFFER_PREFIX_SIZE,
              "Number of samples %lu exceeds buffer prefix size %d for zero-copy RX\n",
              num_samples,
              BUFFER_PREFIX_SIZE);
  AssertFatal(antenna < channel->num_antennas_rx,
              "Antenna index %d out of range (num antennas: %d)\n",
              antenna,
              channel->num_antennas_rx);
  ShmTDIQChannelData *data = channel->data;
  // timestamp in the future
  uint64_t current_time = data->client_sync_data.timestamp;
  if (timestamp > current_time) {
    *rx_iq_data = NULL;
    return CHANNEL_ERROR_TOO_EARLY;
  }
  // timestamp is too far in the past
  if (current_time - timestamp >= CIRCULAR_BUFFER_SIZE) {
    *rx_iq_data = NULL;
    return CHANNEL_ERROR_TOO_LATE;
  }

  sample_t *base_ptr = get_prefix_buffer_ptr(channel->rx_iq_data, antenna);

  uint64_t first_sample_index = timestamp % CIRCULAR_BUFFER_SIZE;
  if (first_sample_index + num_samples > CIRCULAR_BUFFER_SIZE) {
    uint64_t num_samples_in_prefix_buffer = CIRCULAR_BUFFER_SIZE - first_sample_index;
    // Handle wrap around inside the prefix buffer
    *rx_iq_data = base_ptr + (BUFFER_PREFIX_SIZE - num_samples_in_prefix_buffer);
  } else {
    *rx_iq_data = base_ptr + BUFFER_PREFIX_SIZE + first_sample_index;
  }
  return CHANNEL_NO_ERROR;
}

void shm_td_iq_channel_produce_samples(ShmTDIQChannel *channel, size_t num_samples)
{
  ShmTDIQChannelData *data = channel->data;
  update_timestamp(&data->sync_data, data->sync_data.timestamp + num_samples);
}

int shm_td_iq_channel_wait(ShmTDIQChannel *channel, uint64_t timestamp, uint64_t timeout_uS)
{
  ShmTDIQChannelData *data = channel->data;
  size_t current_timestamp = data->sync_data.timestamp;
  if (current_timestamp >= timestamp) {
    return 0;
  }
  if (timeout_uS == 0) {
    wait_for_sync(&data->sync_data, timestamp, &channel->abort);
    return 0;
  } else {
    return wait_for_sync_with_timeout(&data->sync_data, timestamp, timeout_uS, &channel->abort);
  }
}

int shm_td_iq_channel_wait_for_client(ShmTDIQChannel *channel, uint64_t timestamp, uint64_t timeout_uS)
{
  ShmTDIQChannelData *data = channel->data;
  AssertFatal(data->client_sync, "Client sync not enabled on this channel\n");
  size_t current_timestamp = data->client_sync_data.timestamp;
  if (current_timestamp >= timestamp) {
    return 0;
  }
  if (timeout_uS == 0) {
    wait_for_sync(&data->client_sync_data, timestamp, &channel->abort);
    return 0;
  } else {
    return wait_for_sync_with_timeout(&data->client_sync_data, timestamp, timeout_uS, &channel->abort);
  }
}

uint64_t shm_td_iq_channel_get_current_sample(const ShmTDIQChannel *channel)
{
  ShmTDIQChannelData *data = channel->data;
  return data->sync_data.timestamp;
}

uint64_t shm_td_iq_channel_get_current_client_sample(const ShmTDIQChannel *channel)
{
  ShmTDIQChannelData *data = channel->data;
  return data->client_sync_data.timestamp;
}

void shm_td_iq_channel_abort(ShmTDIQChannel *channel)
{
  ShmTDIQChannelData *data = channel->data;
  mutexlock(data->sync_data.mutex);
  channel->abort = true;
  condbroadcast(data->sync_data.cond);
  mutexunlock(data->sync_data.mutex);
  if (data->client_sync) {
    mutexlock(data->client_sync_data.mutex);
    condbroadcast(data->client_sync_data.cond);
    mutexunlock(data->client_sync_data.mutex);
  }
}

bool shm_td_iq_channel_is_aborted(const ShmTDIQChannel *channel)
{
  return channel->abort;
}

void shm_td_iq_channel_destroy(ShmTDIQChannel *channel)
{
  ShmTDIQChannelData *data = channel->data;
  size_t tx_buffer_size = CIRCULAR_BUFFER_SIZE * sizeof(sample_t) * data->num_antennas_tx;
  size_t rx_buffer_size = CIRCULAR_BUFFER_SIZE * sizeof(sample_t) * data->num_antennas_rx;
  size_t total_size = sizeof(ShmTDIQChannelData) + tx_buffer_size + rx_buffer_size;
  if (channel->type == IQ_CHANNEL_TYPE_SERVER) {
    munmap(data, total_size);
    shm_unlink(channel->name);
  } else {
    munmap(data, total_size);
  }
  free(channel);
}

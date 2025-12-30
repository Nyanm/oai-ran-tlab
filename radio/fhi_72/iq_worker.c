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

#include "iq_worker.h"
#include "system.h"
#include "xran_compression.h"
#include "xran_fh_o_du.h"
#include "xran_up_api.h"
#include <asm-generic/errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <rte_mbuf.h>
#include <rte_ring.h>
#include <rte_ring_core.h>
#include <rte_cycles.h>
#include <rte_errno.h>
#include <rte_eal.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include "assertions.h"
#include <stdint.h>
#include <log.h>

#define DECOMPRESSION_RING_SIZE 1024
#define DECOMPRESSION_BURST_SIZE 32

// Static buffer for decompression tasks
typedef struct {
  int comp_method;
  int iq_width;
  int num_iq;
  void *input_buffer;
  void *output_buffer;
  void *mbuf;
} iq_task_t;

typedef struct {
  struct rte_ring *ring;
  iq_task_t iq_task_buffer[DECOMPRESSION_RING_SIZE];
  pthread_t thread;
  uint64_t buffer_idx;
  bool stop;
} iq_woker_context_t;

static iq_woker_context_t iq_worker_context;

static void *iq_worker_thread(void *arg)
{
  iq_woker_context_t *context = (iq_woker_context_t *)arg;
  iq_task_t *tasks[DECOMPRESSION_BURST_SIZE];
  uint32_t num_dequeued;

  while (!context->stop) {
    num_dequeued = rte_ring_dequeue_burst(context->ring, (void **)tasks, DECOMPRESSION_BURST_SIZE, NULL);
    if (num_dequeued != 0) {
        LOG_D(HW, "Dequeued %d tasks\n", num_dequeued);
    }

    if (num_dequeued == 0) {
      rte_pause();
      continue;
    }

    for (uint32_t i = 0; i < num_dequeued; i++) {
      AssertFatal(tasks[i]->comp_method == XRAN_COMPMETHOD_NONE, "Unsupported compression method: %d", tasks[i]->comp_method);
      uint16_t *source = (uint16_t *)tasks[i]->input_buffer;
      int16_t *destination = (int16_t *)tasks[i]->output_buffer;
      for (int j = 0; j < tasks[i]->num_iq * 2; j++) {
        destination[j] = (int16_t)ntohs(source[j]);
      }
      rte_pktmbuf_free(tasks[i]->mbuf);
    }
  }
  return 0;
}

int iq_worker_init(void)
{
  memset(&iq_worker_context, 0, sizeof(iq_worker_context));
  iq_worker_context.ring = rte_ring_create("iq_worker_ring", DECOMPRESSION_RING_SIZE, SOCKET_ID_ANY, RING_F_SP_ENQ | RING_F_SC_DEQ);

  if (iq_worker_context.ring == NULL) {
    rte_exit(EXIT_FAILURE, "Cannot create decompression ring: %s\n", rte_strerror(rte_errno));
  }

  threadCreate(&iq_worker_context.thread, iq_worker_thread, &iq_worker_context, "oran_iq_worker", -1, OAI_PRIORITY_RT_MAX);

  return 0;
}

void iq_worker_destroy(void)
{
  iq_worker_context.stop = true;
  rte_ring_free(iq_worker_context.ring);
  int ret = pthread_join(iq_worker_context.thread, NULL);
  AssertFatal(ret == 0, "pthread_join failed: %d", ret);
}

void iq_worker_enqueue(int comp_method, int iq_width, int num_iq, void *input_buffer, void *output_buffer, void *mbuf)
{
  int buffer_index = iq_worker_context.buffer_idx++ % DECOMPRESSION_RING_SIZE;
  iq_task_t *task = &iq_worker_context.iq_task_buffer[buffer_index];
  task->comp_method = comp_method;
  task->iq_width = iq_width;
  task->num_iq = num_iq;
  task->input_buffer = input_buffer;
  task->output_buffer = output_buffer;
  task->mbuf = mbuf;
  int ret = rte_ring_enqueue(iq_worker_context.ring, task);
  if (ret == ENOBUFS) {
    LOG_W(HW, "iq_worker is too slow\n");
    rte_pktmbuf_free(mbuf);
  }
}

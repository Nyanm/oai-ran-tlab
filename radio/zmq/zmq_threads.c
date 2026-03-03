#include "zmq_threads.h"

#include <zmq.h>

#include "common_lib.h"
#include "common/utils/system.h"

static void *zmq_rx_thread(void *_t)
{
  zmq_thread_t *t = _t;
  uint64_t ts = 0;

  while (true) {
    char buffer[4] = { 0 };
    zmq_send(t->zmq_socket, buffer, 4, 0);
    int l = zmq_recv(t->zmq_socket, t->data_buffer, t->max_data_buffer_size, 0);
    AssertFatal(l % 4 == 0, "bad data received from ZeroMQ socket, not multiple of 4\n");
    int sample_count = l / 4;
    circular_buffer_write(t->circular_buffer, t->data_buffer, sample_count, ts);
    ts += sample_count;
  }

  return 0;
}

static void *zmq_tx_thread(void *_t)
{
  zmq_thread_t *t = _t;
  uint64_t ts = 0;

  while (true) {
    char buffer[4] = { 0 };
    int l = zmq_recv(t->zmq_socket, buffer, 4, 0);
    DevAssert(l > 0 && l <= 4);
    int sample_count = 4096;
    circular_buffer_read(t->circular_buffer, t->data_buffer, sample_count, ts);
    zmq_send(t->zmq_socket, t->data_buffer, sample_count, 0);
    ts += sample_count;
  }

  return 0;
}

static zmq_thread_t *create_zmq_struct(char *url, zmq_connection_mode_t st, circular_buffer_t *cb, int type)
{
  zmq_thread_t *ret = calloc_or_fail(1, sizeof(*ret));
  ret->zmq_context = zmq_ctx_new();
  DevAssert(ret->zmq_context);
  ret->zmq_socket = zmq_socket(ret->zmq_context, type);
  DevAssert(ret->zmq_socket);
  int rc;
  if (st == ZMQ_CONNECT)
    rc = zmq_connect(ret->zmq_socket, url);
  else
    rc = zmq_bind(ret->zmq_socket, url);
//printf("rc %d %s\n", rc, strerror(errno)); fflush(stdout);
  DevAssert(!rc);
  ret->circular_buffer = cb;
  ret->max_data_buffer_size = 1024*1024;
  ret->data_buffer = calloc_or_fail(1, ret->max_data_buffer_size);
  return ret;
}

zmq_thread_t *zmq_start_rx_thread(char *url, zmq_connection_mode_t st, circular_buffer_t *cb)
{
  zmq_thread_t *ret = create_zmq_struct(url, st, cb, ZMQ_REQ);
  threadCreate(&ret->thread_id, zmq_rx_thread, ret, "ZeroMQ RX thread", -1, SCHED_OAI);
  return ret;
}

zmq_thread_t *zmq_start_tx_thread(char *url, zmq_connection_mode_t st, circular_buffer_t *cb)
{
  zmq_thread_t *ret = create_zmq_struct(url, st, cb, ZMQ_REP);
  threadCreate(&ret->thread_id, zmq_tx_thread, ret, "ZeroMQ TX thread", -1, SCHED_OAI);
  return ret;
}

void zmq_kill_thread(zmq_thread_t *t)
{
  int ret = pthread_kill(t->thread_id, SIGKILL);
  DevAssert(!ret);
  void *retval;
  ret = pthread_join(t->thread_id, &retval);
  DevAssert(!ret);
  /* todo: cleanup ZeroMQ */
  free(t->data_buffer);
  free(t);
}

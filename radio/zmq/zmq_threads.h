#ifndef RADIO_ZMQ_THREADS_H
#define RADIO_ZMQ_THREADS_H

#include <pthread.h>
#include <stdbool.h>

#include "circular_buffer.h"
#include "zmq_configuration.h"

typedef struct {
  circular_buffer_t *circular_buffer;
  int max_data_buffer_size;
  void *data_buffer;
  pthread_t thread_id;
  void *zmq_context;
  void *zmq_socket;
} zmq_thread_t;

zmq_thread_t *zmq_start_rx_thread(char *url, zmq_connection_mode_t st, circular_buffer_t *cb);
zmq_thread_t *zmq_start_tx_thread(char *url, zmq_connection_mode_t st, circular_buffer_t *cb);

void zmq_kill_thread(zmq_thread_t *t);

#endif /* RADIO_ZMQ_THREADS_H */

#ifndef RADIO_ZMQ_CIRCULAR_BUFFER_H
#define RADIO_ZMQ_CIRCULAR_BUFFER_H

#include <stdint.h>
#include <pthread.h>

/* design (code may differ a bit)
 * +-----------------------------------------------------------------+
 * |          |                                       |              |
 * |        start                                 end=start+length-1 |
 * |          |                                       |              |
 * +-----------------------------------------------------------------+
 * ts: increased each time 'start' advances
 * used to detect overflows
 *
 * read(n, ts):
 *             if (ts < buffer.ts)
 *                 overflow detected (? done in write)
 *                 l = feed 0s up to n
 *                 n -= l
 *             if (!n)
 *                 return
 *             lock
 *             while (n)
 *                 l = min(n, data in buffer)
 *                 if (l == 0) (no data available)
 *                     timeout = n * sample_timeout_duration
 *                     wait data with timeout
 *                     if (timeout)
 *                         advance ts
 *                         //put available samples (or not?) (there is no)
 *                         pad with 0s
 *                         underflow detected
 *                         unlock
 *                         return
 *                 copy l samples
 *                 updata start/length
 *                 n -= l
 *                 advance ts
 *                 cond signal
 *             unlock
 *             return
 *             
 * write(n, ts):
 *             if (ts < buffer.ts)
 *                 late detected
 *                 l = min(n, buffer.ts - ts)
 *                 n -= l
 *             lock
 *             while (n)
 *                 l = min(n, free space in buffer)
 *                 if (l == 0) (no space)
 *                     timeout = n * sample_timeout_duration
 *                     wait for some free space with timeout
 *                     if (timeout)
 *                         overwrite old at start of buffer
 *                         overflow detected
 *                         increase start
 *                         increase ts
 *                         cond signal
 *                         unlock
 *                         return
 *                 put l samples
 *                 increase length
 *                 cond signal
 *             unlock
 *             return
 *
 */

typedef struct {
  uint32_t *buffer;
  int buffer_size;
  pthread_mutex_t m;
  pthread_cond_t c;
  int samples_per_second;
  double speed;
  volatile int start;
  volatile int length;
  uint64_t ts;           /* timestamp */
  /* statistics - unit: sample */
  uint64_t late;         /* writing with old ts */
  uint64_t underflow;    /* reading but timeout and no samples available */
  uint64_t overflow;     /* writing but timeout and buffer gets overwritten */
} circular_buffer_t;

circular_buffer_t *new_circular_buffer(int buffer_size, int samples_per_second, double speed);

void circular_buffer_read(circular_buffer_t *cb, void *samples, int count, uint64_t ts);
void circular_buffer_write(circular_buffer_t *cb, void *samples, int count, uint64_t ts);

#endif /* RADIO_ZMQ_CIRCULAR_BUFFER_H */

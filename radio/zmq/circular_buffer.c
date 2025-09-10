#include "circular_buffer.h"

#include <stdint.h>
#include <pthread.h>
#include <time.h>

#include "common_lib.h"

/* internal functions */

static void lock(circular_buffer_t *cb)
{
  int ret = pthread_mutex_lock(&cb->m);
  DevAssert(ret == 0);
}

static void unlock(circular_buffer_t *cb)
{
  int ret = pthread_mutex_unlock(&cb->m);
  DevAssert(ret == 0);
}

/* given a timeout, returns the absolute time of this timeout starting
 * from now (the time when abs_timeout() is called)
 */
static struct timespec abs_timeout(double delay_nanosecond)
{
printf("abs_timeout %f\n", delay_nanosecond);
  struct timespec t;

  int ret = clock_gettime(CLOCK_REALTIME, &t);
  DevAssert(ret == 0);

printf("now %ld %ld\n", t.tv_sec, t.tv_nsec);
  uint64_t nsec = t.tv_nsec + delay_nanosecond;
  t.tv_sec += nsec / 1000000000;
  t.tv_nsec = nsec % 1000000000;

printf("wait limit %ld %ld\n", t.tv_sec, t.tv_nsec);
  return t;
}

/* return false if timeout */
static bool cond_wait_timeout(circular_buffer_t *cb, struct timespec abs_time)
{
  int ret = pthread_cond_timedwait(&cb->c, &cb->m, &abs_time);
  DevAssert(ret == 0 || ret == ETIMEDOUT);

  return ret == 0;
}

static void cond_signal(circular_buffer_t *cb)
{
  int ret = pthread_cond_broadcast(&cb->c);
  DevAssert(ret == 0);
}

/* copy data from start of circular buffer to 'samples' */
static void copy_samples(uint32_t *samples, circular_buffer_t *cb, int count)
{
  int tail_size = cb->buffer_size - cb->start;
  int l = min(count, tail_size);
  memcpy(samples, cb->buffer + cb->start, l * sizeof(uint32_t));
  count -= l;
  samples += l;
  memcpy(samples, cb->buffer, count * sizeof(uint32_t));
}

/* copy data from 'samples' to end of circular buffer */
static void copy_to_buffer(circular_buffer_t *cb, uint32_t *samples, int count)
{
  int insert_point = (cb->start + cb->length) % cb->buffer_size;
  int tail_size = cb->buffer_size - insert_point;
  int l = min(count, tail_size);
  memcpy(cb->buffer + insert_point, samples, l * sizeof(uint32_t));
  count -= l;
  samples += l;
  memcpy(cb->buffer, samples, count * sizeof(uint32_t));
}

/* public interface */

circular_buffer_t *new_circular_buffer(int buffer_size, int samples_per_second, double speed)
{
  circular_buffer_t *ret = calloc_or_fail(1, sizeof(*ret));

  ret->buffer = calloc_or_fail(buffer_size, sizeof(uint32_t));
  ret->buffer_size = buffer_size;
  ret->samples_per_second = samples_per_second;
  ret->speed = speed;

  pthread_mutex_init(&ret->m, NULL);
  pthread_cond_init(&ret->c, NULL);

  return ret;
}

void circular_buffer_read(circular_buffer_t *cb, void *_samples, int count, uint64_t ts)
{
  DevAssert(count <= cb->buffer_size);

  uint32_t *samples = _samples;
  if (ts < cb->ts) {
    /* overflow detected */
    uint64_t overflow_count = cb->ts - ts;
    cb->overflow += overflow_count;
    /* put some zeros */
    int zero_count = min(overflow_count, count);
    memset(samples, 0, zero_count * sizeof(uint32_t));
    count -= zero_count;
    samples += zero_count;
  }
  if (!count)
    return;

  lock(cb);

  while (count) {
    int l = min(count, cb->length);
    if (l == 0) {
      /* no data available - wait (with timeout) */
      double sample_timeout_duration = 1000000000. / cb->samples_per_second /  cb->speed;
      double timeout_nanosecond = count * sample_timeout_duration;
      struct timespec t = abs_timeout(timeout_nanosecond);
      bool timeout = false;
      /* wait for timeout or some data in the buffer */
      while (!(timeout || cb->length))
        timeout = !cond_wait_timeout(cb, t);
      if (timeout) {
        /* advance ts */
        cb->ts += l;
        memset(samples, 0, l * sizeof(uint32_t));
        /* underflow detected */
        cb->underflow += l;
        unlock(cb);
        return;
      }
    } else {
      /* data available - copy */
      copy_samples(samples, cb, l);
      cb->start += l;
      cb->start %= cb->buffer_size;
      cb->length -= l;
      count -= l;
      samples += l;
      cb->ts += l;
      cond_signal(cb);
    }
  }

  unlock(cb);
}

void circular_buffer_write(circular_buffer_t *cb, void *samples, int count, uint64_t ts)
{
  DevAssert(count <= cb->buffer_size);

  if (ts < cb->ts) {
    /* late detected */
    uint64_t late_count = min(count, cb->ts - ts);
    cb->late += late_count;
    count -= late_count;
  }

  lock(cb);

  while (count) {
    int free_space = cb->buffer_size - cb->length;
    int l = min(count, free_space);
    if (l == 0) {
      /* no space */
      double sample_timeout_duration = 1000000000. / cb->samples_per_second /  cb->speed;
      double timeout_nanosecond = count * sample_timeout_duration;
      struct timespec t = abs_timeout(timeout_nanosecond);
      bool timeout = false;
      /* wait until timeout or there is some free space in the buffer */
      while (!(timeout || cb->length < cb->buffer_size))
        timeout = !cond_wait_timeout(cb, t);
      if (timeout) {
        /* advance buffer - "forget" old data */
        cb->start += count;
        cb->length -= count;
        /* put new data */
        copy_to_buffer(cb, samples, count);
        /* increase ts */
        cb->ts += count;
        cond_signal(cb);
        unlock(cb);
        return;
      }
    } else {
      /* space available - copy data */
      copy_to_buffer(cb, samples, l);
      cb->length += l;
      count -= l;
      samples += l;
      cond_signal(cb);
    }
  }

  unlock(cb);
}

#include "circular_buffer.h"
#include <cstring>
#include <iostream>
#include <algorithm>

circular_buffer::circular_buffer(size_t max_size) : _max_size(max_size)
{
  _buffer = std::make_unique<c16_t[]>(max_size);
}

size_t circular_buffer::push_samples(const c16_t *samples, const size_t nsamps)
{
  size_t overflow = 0;
  // if nsamps > max_size skip nsamps - max_size samples
  size_t nsamps_left = nsamps;
  if (nsamps > _max_size) {
    samples += nsamps - _max_size;
    nsamps_left = _max_size;
    overflow += nsamps - _max_size;
  }

  // Detect overflow
  if (_size + nsamps_left > _max_size) {
    size_t new_tail_pos = (_head + nsamps_left) % _max_size;
    overflow += (_size + nsamps_left) - _max_size;
    _tail = new_tail_pos;
  }

  size_t first_chunk = std::min(nsamps_left, _max_size - _head);
  memcpy(&_buffer[_head], samples, first_chunk * sizeof(c16_t));
  _head = (_head + first_chunk) % _max_size;
  samples += first_chunk;
  nsamps_left -= first_chunk;
  if (nsamps_left > 0) {
    memcpy(&_buffer[0], samples, nsamps_left * sizeof(c16_t));
    _head = nsamps_left;
  }

  _size = std::min(_size + nsamps, _max_size);

  return overflow;
}

size_t circular_buffer::push_zeros(const size_t num_zeros)
{
  size_t overflow = 0;
  // if nsamps > max_size skip nsamps - max_size samples
  size_t nsamps_left = num_zeros;
  if (num_zeros > _max_size) {
    nsamps_left = _max_size;
    overflow += num_zeros - _max_size;
  }

  // Detect overflow
  if (_size + nsamps_left > _max_size) {
    size_t new_tail_pos = (_head + nsamps_left) % _max_size;
    overflow += (_size + nsamps_left) - _max_size;
    _tail = new_tail_pos;
  }

  size_t first_chunk = std::min(nsamps_left, _max_size - _head);
  memset(&_buffer[_head], 0, first_chunk * sizeof(c16_t));
  _head = (_head + first_chunk) % _max_size;
  nsamps_left -= first_chunk;
  if (nsamps_left > 0) {
    memset(&_buffer[0], 0, nsamps_left * sizeof(c16_t));
    _head = nsamps_left;
  }

  _size = std::min(_size + num_zeros, _max_size);

  return overflow;
}

size_t circular_buffer::pop_samples(c16_t *samples, size_t num_samples)
{
  size_t samples_to_pop = std::min(_size, num_samples);
  if (samples_to_pop > 0) {
    if (_tail + samples_to_pop > _max_size) {
      size_t first_chunk = _max_size - _tail;
      memcpy(samples, &_buffer[_tail], first_chunk * sizeof(c16_t));
      memcpy(samples + first_chunk, &_buffer[0], (samples_to_pop - first_chunk) * sizeof(c16_t));
    } else {
      memcpy(samples, &_buffer[_tail], samples_to_pop * sizeof(c16_t));
    }
    _tail = (_tail + samples_to_pop) % _max_size;
    _size -= samples_to_pop;
    return samples_to_pop;
  }
  return 0;
}

void circular_buffer::clear_samples()
{
  _head = 0;
  _tail = 0;
  _size = 0;
}

void circular_buffer::reset()
{
  clear_samples();
}

size_t circular_buffer::size() const
{
  return _size;
}

void overflow_buffer::push_samples(const c16_t *samples, size_t nsamps)
{
  std::lock_guard<std::mutex> lock(_mutex);
  size_t overflow = buffer.push_samples(samples, nsamps);
  _zeros_to_send += overflow;
}

void overflow_buffer::push_zeros(size_t num_zeros)
{
  std::lock_guard<std::mutex> lock(_mutex);
  size_t overflow = buffer.push_zeros(num_zeros);
  _zeros_to_send += overflow;
}

size_t overflow_buffer::pop_samples(c16_t *samples, size_t num_samples)
{
  std::lock_guard<std::mutex> lock(_mutex);
  size_t samples_popped = 0;
  if (_zeros_to_send > 0) {
    size_t num_zeros = std::min(_zeros_to_send, num_samples);
    memset(samples, 0, num_zeros * sizeof(c16_t));
    _zeros_to_send -= num_zeros;
    samples += num_zeros;
    num_samples -= num_zeros;
    samples_popped += num_zeros;
  }

  if (num_samples > 0) {
    samples_popped += buffer.pop_samples(samples, num_samples);
  }
  return samples_popped;
}

void overflow_buffer::reset()
{
  std::lock_guard<std::mutex> lock(_mutex);
  buffer.reset();
  _zeros_to_send = buffer.size() / 2;
}

void overflow_buffer::clear_samples()
{
  std::lock_guard<std::mutex> lock(_mutex);
  buffer.clear_samples();
  _zeros_to_send = 0;
}

size_t overflow_buffer::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return buffer.size() + _zeros_to_send;
}


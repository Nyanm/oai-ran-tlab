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
#pragma once
#include <vector>
#include <mutex>
#include "common/platform_types.h"

#pragma once
#include <mutex>
#include <memory>
#include "common/platform_types.h"

// A basic cirular sample buffer class
class circular_buffer {
 private:
  std::unique_ptr<c16_t[]> _buffer;
  size_t _head = 0;
  size_t _tail = 0;
  size_t _size = 0;
  size_t _max_size;

 public:
  circular_buffer(size_t max_size = 614400);
  size_t push_samples(const c16_t *samples, size_t nsamps);
  size_t push_zeros(size_t num_zeros);
  size_t pop_samples(c16_t *samples, size_t num_samples);
  void reset();
  void clear_samples();
  size_t size() const;
};

// A wrapper around circular_buffer that counts overflows
class overflow_buffer {
  circular_buffer buffer;
  size_t _zeros_to_send;
  std::mutex _mutex;

 public:
  overflow_buffer(size_t max_size = 614400, size_t zeros_to_send = 0) : buffer(max_size), _zeros_to_send(zeros_to_send)
  {
  }
  void push_samples(const c16_t *samples, size_t nsamps);
  void push_zeros(size_t num_zeros);
  size_t pop_samples(c16_t *samples, size_t num_samples);
  void reset();
  void clear_samples();
  size_t size();
};

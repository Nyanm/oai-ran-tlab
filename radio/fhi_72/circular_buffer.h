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
#ifndef __CIRCULAR_BUFFER_H__
#define __CIRCULAR_BUFFER_H__
#include <stdint.h>

typedef struct {
  int num_antennas;
  int num_slots;
  int num_symbols;
  int max_iq_samples;
  uint32_t ****data;
} circular_buffer_t;

void circular_buffer_init(circular_buffer_t *cb, int num_antennas, int num_slots, int num_symbols, int max_iq_samples);
uint32_t *circular_buffer_get_data(circular_buffer_t *cb, int antenna, int slot, int symbol);
void circular_buffer_destroy(circular_buffer_t *cb);

#endif

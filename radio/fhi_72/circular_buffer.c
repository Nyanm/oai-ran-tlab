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

#include "circular_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <malloc.h> // For memalign or _mm_malloc
#include "assertions.h"

void circular_buffer_init(circular_buffer_t *cb, int num_antennas, int num_slots, int num_symbols, int max_iq_samples)
{
  cb->num_antennas = num_antennas;
  cb->num_slots = num_slots;
  cb->num_symbols = num_symbols;
  cb->max_iq_samples = max_iq_samples;

  cb->data = (uint32_t ****)malloc(num_antennas * sizeof(uint32_t ***));
  for (int i = 0; i < num_antennas; i++) {
    cb->data[i] = (uint32_t ***)malloc(num_slots * sizeof(uint32_t **));
    for (int j = 0; j < num_slots; j++) {
      cb->data[i][j] = (uint32_t **)malloc(num_symbols * sizeof(uint32_t *));
      for (int k = 0; k < num_symbols; k++) {
        cb->data[i][j][k] = (uint32_t *)memalign(64, max_iq_samples * sizeof(uint32_t));
        AssertFatal(cb->data[i][j][k] != NULL, "Failed to allocate memory for circular buffer data\n");
      }
    }
  }
}

uint32_t *circular_buffer_get_data(circular_buffer_t *cb, int antenna, int slot, int symbol)
{
  AssertFatal(antenna < cb->num_antennas, "Antenna index out of bounds\n");
  AssertFatal(slot < cb->num_slots, "Slot index out of bounds\n");
  AssertFatal(symbol < cb->num_symbols, "Symbol index out of bounds\n");
  return cb->data[antenna][slot][symbol];
}

void circular_buffer_destroy(circular_buffer_t *cb)
{
  for (int i = 0; i < cb->num_antennas; i++) {
    for (int j = 0; j < cb->num_slots; j++) {
      for (int k = 0; k < cb->num_symbols; k++) {
        free(cb->data[i][j][k]);
      }
      free(cb->data[i][j]);
    }
    free(cb->data[i]);
  }
}

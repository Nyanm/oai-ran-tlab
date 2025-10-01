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
#ifndef NR_ORU_H
#define NR_ORU_H
#include "nr-oru.h"
#include <executables/softmodem-common.h>
#include "openair1/PHY/defs_RU.h"

typedef struct {
  openair0_timestamp sample;
  int slot;
  int frame;
  int symbol;
} initial_sync_t;

typedef struct {
  pthread_t north_read_thread;
  pthread_t south_read_thread;
  RU_t *ru;
  notifiedFIFO_t sync_fifo;
} ORU_t;

void *oru_north_read_thread(void *arg);
void *oru_south_read_thread(void *arg);

#endif

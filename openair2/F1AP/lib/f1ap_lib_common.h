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

#ifndef F1AP_LIB_COMMON_H_
#define F1AP_LIB_COMMON_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "openair3/UTILS/conversions.h"
#include "common/5g_platform_types.h"
#include "common/utils/ds/byte_array.h"

#include "F1AP_Cause.h"
#include "F1AP_SNSSAI.h"
#include "f1ap_messages_types.h"
#include "common/utils/eq_check.h"

#define _F1_CHECK_EXP(EXP)                 \
  do {                                     \
    if (!(EXP)) {                          \
      PRINT_ERROR("Failed at " #EXP "\n"); \
      return false;                        \
    }                                      \
  } while (0)

#define CP_OPT_BYTE_ARRAY(dst, src)          \
  do {                                       \
    if (src) {                               \
      dst = calloc_or_fail(1, sizeof(*dst)); \
      *(dst) = copy_byte_array(*(src));      \
    }                                        \
  } while (0)

#define FREE_OPT_BYTE_ARRAY(a) \
  do {                         \
    if (a) {                   \
      free_byte_array(*(a));   \
    } \
    free(a);                   \
  } while (0)

/* similar to asn1cCallocOne(), duplicated to not confuse with asn.1 types */
#define _F1_MALLOC(VaR, VaLue)          \
  do {                                  \
    VaR = malloc_or_fail(sizeof(*VaR)); \
    *VaR = VaLue;                       \
  } while (0)

bool eq_f1ap_plmn(const plmn_id_t *a, const plmn_id_t *b);
bool eq_f1ap_cell_info(const f1ap_served_cell_info_t *a, const f1ap_served_cell_info_t *b);
bool eq_f1ap_sys_info(const f1ap_gnb_du_system_info_t *a, const f1ap_gnb_du_system_info_t *b);
bool eq_f1ap_freq_info(const f1ap_nr_frequency_info_t *a, const f1ap_nr_frequency_info_t *b);
bool eq_f1ap_tx_bandwidth(const f1ap_transmission_bandwidth_t *a, const f1ap_transmission_bandwidth_t *b);

struct OCTET_STRING;
uint8_t *cp_octet_string(const struct OCTET_STRING *os, int *len);

F1AP_SNSSAI_t encode_nssai(const nssai_t *nssai);
nssai_t decode_nssai(const F1AP_SNSSAI_t *nssai);

F1AP_Cause_t encode_f1ap_cause(f1ap_Cause_t cause, long cause_value);
bool decode_f1ap_cause(F1AP_Cause_t f1_cause, f1ap_Cause_t *cause, long *cause_value);

#endif /* F1AP_LIB_COMMON_H_ */

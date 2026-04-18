/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NRPPA_COMMON_H_
#define NRPPA_COMMON_H_

#include "oai_asn1.h"
#include "common/utils/LOG/log.h"
#include "ds/byte_array.h"
#include "nrppa_includes.h"

// gnb and ue related info in NRPPA message
typedef struct nrppa_gnb_ue_info_s {
  instance_t instance;
  int32_t gNB_ue_ngap_id;
  int64_t amf_ue_ngap_id;
  byte_array_t routing_id;
} nrppa_gnb_ue_info_t;

typedef int (*nrppa_message_decoded_callback)(nrppa_gnb_ue_info_t *nrppa_msg_info, const NRPPA_NRPPA_PDU_t *pdu);

#endif /* NRPPA_COMMON_H_ */

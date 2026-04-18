/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef XNAP_COMMON_H_
#define XNAP_COMMON_H_

#include "XNAP_XnAP-PDU.h"
#include "common/openairinterface5g_limits.h"
#include "oai_asn1.h"
#include "XNAP_ProtocolIE-Field.h"
#include "XNAP_InitiatingMessage.h"
#include "XNAP_ProtocolIE-ContainerPair.h"
#include "XNAP_ProtocolExtensionField.h"
#include "XNAP_ProtocolExtensionContainer.h"
#include "XNAP_asn_constant.h"

#ifndef XNAP_PORT
#define XNAP_PORT 38422
#endif

ssize_t xnap_generate_initiating_message(uint8_t **buffer,
                                         uint32_t *length,
                                         XNAP_ProcedureCode_t procedureCode,
                                         XNAP_Criticality_t criticality,
                                         asn_TYPE_descriptor_t *td,
                                         void *sptr);

#endif /* XNAP_COMMON_H_ */

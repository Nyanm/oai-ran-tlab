/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef FIVEG_PLATFORM_TYPES_H__
#define FIVEG_PLATFORM_TYPES_H__

#include <stdint.h>

typedef struct plmn_id_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t mnc_digit_length;
} plmn_id_t;

typedef struct nssai_s {
  uint8_t sst;
  uint32_t sd;
} nssai_t;

// Globally Unique AMF Identifier
typedef struct nr_guami_s {
  plmn_id_t plmn;
  uint8_t amf_region_id;
  uint16_t amf_set_id;
  uint8_t amf_pointer;
} nr_guami_t;

typedef enum {
  PDUSessionType_ipv4 = 0,
  PDUSessionType_ipv6 = 1,
  PDUSessionType_ipv4v6 = 2,
  PDUSessionType_ethernet = 3,
  PDUSessionType_unstructured = 4
} pdu_session_type_t;

typedef enum { NON_DYNAMIC, DYNAMIC } fiveQI_t;

/* 5QI (5G QoS Identifier) - 3GPP TS 23.501 §5.7.2.1
 * Range: 0..255
 * - Standardized 5QI values: have one-to-one mapping to standardized 5G QoS characteristics (Table 5.7.4-1)
 * - Pre-configured 5QI values: pre-configured in the AN
 * - Dynamically assigned 5QI values: require signaling of QoS characteristics as part of QoS profile */
#define MIN_FIVEQI 0
#define MAX_FIVEQI 255

/* ARP Priority Level - 3GPP TS 23.501 §5.7.2.2
 * The ARP priority level defines the relative importance of a QoS Flow.
 * Range: 1 to 15, with 1 as the highest priority.
 * ARP priority levels 1-8: authorized by serving network (prioritized treatment)
 * ARP priority levels 9-15: authorized by home network (roaming scenarios) */
typedef uint8_t qos_arp_priority_level_t;
#define MIN_QOS_ARP_PRIORITY_LEVEL 1 // highest priority
#define MAX_QOS_ARP_PRIORITY_LEVEL 15 // lowest priority

/* Pre-emption Capability */
typedef enum {
  PEC_SHALL_NOT_TRIGGER_PREEMPTION = 0,
  PEC_MAY_TRIGGER_PREEMPTION,
  PEC_MAX,
} qos_pec_t;

/* Pre-emption Vulnerability */
typedef enum {
  PEV_NOT_PREEMPTABLE = 0,
  PEV_PREEMPTABLE = 1,
  PEV_MAX,
} qos_pev_t;

/* Allocation and Retention Priority (ARP) - 3GPP TS 23.501 §5.7.2.2
 * Contains information about priority level, pre-emption capability and vulnerability.
 * Used for admission control of GBR traffic and pre-emption decisions. */
typedef struct {
  // ARP priority level (1-15, 1 = highest)
  qos_arp_priority_level_t priority_level;
  qos_pec_t pre_emp_capability;
  qos_pev_t pre_emp_vulnerability;
} qos_arp_t;

/* QoS Priority Level - 3GPP TS 23.501 §5.7.3.3
 * The Priority Level associated with 5G QoS characteristics indicates a priority
 * in scheduling resources among QoS Flows. The lowest Priority Level value
 * corresponds to the highest priority.
 * Range: 1 to 127, with 1 as the highest priority and 127 as the lowest priority.
 * Used for scheduling resources among QoS Flows (different from ARP priority level
 * which is used for admission control/preemption).
 * Every standardized 5QI is associated with a default Priority Level value. */
typedef uint8_t qos_priority_level_t;
#define MIN_QOS_PRIORITY_LEVEL 1 // highest priority
#define MAX_QOS_PRIORITY_LEVEL 127 // lowest priority

typedef struct pdusession_level_qos_parameter_s {
  uint8_t qfi;
  uint64_t fiveQI;
  qos_priority_level_t qos_priority;
  fiveQI_t fiveQI_type;
  qos_arp_t arp;
} pdusession_level_qos_parameter_t;

#endif

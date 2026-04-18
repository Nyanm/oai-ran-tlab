/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef E1AP_LIB_COMMON_H_
#define E1AP_LIB_COMMON_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "openair3/UTILS/conversions.h"
#include "e1ap_messages_types.h"
#include "common/utils/eq_check.h"

#define CHECK_E1AP_DEC(exp)                                                             \
  do {                                                                                  \
    if (!(exp)) {                                                                       \
        PRINT_ERROR("Failed executing " #exp " in %s() line %d\n", __func__, __LINE__); \
        return false;                                                                   \
    }                                                                                   \
  } while (0)

/* deep copy of optional E1AP IE */
#define _E1_CP_OPTIONAL_IE(dest, src, field)                  \
  do {                                                        \
    if ((src)->field) {                                       \
      (dest)->field = malloc_or_fail(sizeof(*(dest)->field)); \
      *(dest)->field = *(src)->field;                         \
    }                                                         \
  } while (0)

struct E1AP_Cause;
struct E1AP_Cause e1_encode_cause_ie(const e1ap_cause_t *cause);
e1ap_cause_t e1_decode_cause_ie(const struct E1AP_Cause *ie);

#endif /* E1AP_LIB_COMMON_H_ */

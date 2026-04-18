/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef NGAP_UTILS_H_
#define NGAP_UTILS_H_

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "common/utils/LOG/log.h"
#include "assertions.h"
#include "common/utils/eq_check.h"

#define NGAP_UE_ID_FMT "0x%06" PRIX32

#define NGAP_ERROR(x, args...) LOG_E(NGAP, x, ##args)
#define NGAP_WARN(x, args...) LOG_W(NGAP, x, ##args)
#define NGAP_TRAF(x, args...) LOG_I(NGAP, x, ##args)
#define NGAP_INFO(x, args...) LOG_I(NGAP, x, ##args)
#define NGAP_DEBUG(x, args...) LOG_D(NGAP, x, ##args)

#endif /* NGAP_UTILS_H_ */

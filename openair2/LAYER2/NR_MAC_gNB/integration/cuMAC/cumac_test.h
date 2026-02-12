//
// Created by user on 20 Nov 2025.
//

#ifndef OPENAIRINTERFACE_CUMAC_TEST_H
#define OPENAIRINTERFACE_CUMAC_TEST_H
#include "nv_ipc.h"
#include <bits/types/struct_timeval.h>
#include "nv_ipc_utils.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include "cumac_msg.h"
#include "cumac_msg_funcs.h"
#include <sys/epoll.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../../../../common/utils/system.h"
#include "../../../../../common/platform_types.h"

int nvIPC_Init();
#endif // OPENAIRINTERFACE_CUMAC_TEST_H

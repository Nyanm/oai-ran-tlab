//
// Created by user on 27 Jan 2026.
//

#ifndef OPENAIRINTERFACE_CUMAC_NVIPC_H
#define OPENAIRINTERFACE_CUMAC_NVIPC_H
#include "cumac_msg.h"
#include "cumac_msg_funcs.h"

#include <nv_ipc.h>
#include <stdbool.h>
bool cumac_nvipc_init();
int cumac_recv_msg(nv_ipc_msg_t* recv_msg);
bool cumac_can_schedule();
void cumac_set_can_schedule(bool val);
bool cumac_send_msg(cumac_msg_t type,build_cumac_msg_fn_v_t fn,  void* args);
void cumac_handle_rx_msg(nv_ipc_msg_t* recv_msg);

#endif // OPENAIRINTERFACE_CUMAC_NVIPC_H

//
// Created by user on 27 Jan 2026.
//

#ifndef OPENAIRINTERFACE_CUMAC_NVIPC_H
#define OPENAIRINTERFACE_CUMAC_NVIPC_H
#include "cumac_msg.h"
#include "cumac_msg_funcs.h"
#include "PHY/defs_common.h"

#include <nv_ipc.h>
#include <stdbool.h>

extern slot_data_entry_t *slot_data[MAX_FRAME_NUMBER];
extern pthread_mutex_t cumac_can_send_mutex;
extern pthread_mutex_t cumac_can_process_mutex;

bool cumac_nvipc_init();
int cumac_recv_msg(nv_ipc_msg_t* recv_msg);
bool cumac_can_schedule();
void cumac_set_can_schedule(bool val);
bool cumac_send_msg(cumac_msg_t type,build_cumac_msg_fn_v_t fn,  void* args);
void cumac_handle_rx_msg(nv_ipc_msg_t* recv_msg);
void cumac_wait_to_send();
void cumac_allow_send();
void cumac_wait_to_process();
void cumac_allow_process();
#endif // OPENAIRINTERFACE_CUMAC_NVIPC_H

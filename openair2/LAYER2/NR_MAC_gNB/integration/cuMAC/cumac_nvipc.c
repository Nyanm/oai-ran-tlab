//
// Created by user on 27 Jan 2026.
//

#include "cumac_nvipc.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <unistd.h>
#include <termios.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/epoll.h>
#include <time.h>
#include <math.h>

#include "nv_ipc_utils.h"
#include "cu_mac_api.h"
#include "cumac_msg.h"
#include "cumac_msg_funcs.h"
#include "LOG/log.h"

#ifdef CUMAC_TIMING_DEBUG

#endif

bool recv_task_running = false;
static nv_ipc_config_t nv_ipc_config;
static bool cumac_start = false;
nv_ipc_t* ipc;
#define MAX_EVENTS 10
#define RECV_BUF_LEN 8192
char cpu_buf_recv[RECV_BUF_LEN];
/* Log TAG configured in nvlog */
static int TAG = (NVLOG_TAG_BASE_NVIPC + 0);
static bool waiting_tti_resp = false;
static bool has_sent_tti_end = false;
static bool has_sent_conf_req = false;
slot_data_entry_t *slot_data[MAX_FRAME_NUMBER];
pthread_mutex_t cumac_can_send_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t cumac_can_process_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint16_t nMaxSchUePerCell;
static uint8_t allocType;
static uint16_t nPrbPerPrg;

uint16_t cumac_nMax_schUePerCell() {
  return nMaxSchUePerCell;
}

uint16_t cumac_nPrbPerPrg() {
  return nPrbPerPrg;
}

int cumac_get_UE_ID_by_RNTI(const uint16_t frame, const uint16_t slot, const uint16_t rnti) {
  const slot_data_entry_t *slot_data_entry = get_slot_data(frame,slot);
  AssertFatal(slot_data_entry, "Entry should not be null!");
  for (int i = 0; i < slot_data_entry->nActiveUe; ++i) {
    if (slot_data_entry->c_rnti[i] == rnti) {
      return i;
    }
  }
  return -1;
}

slot_data_entry_t* get_slot_data(const uint16_t frame, const uint16_t slot) {
  return &slot_data[frame][slot];
}


bool cumac_can_schedule() {
  return !waiting_tti_resp;
}
void cumac_set_can_schedule(bool val) {
   waiting_tti_resp = !val;
}
bool cumac_send_msg(cumac_msg_t type, build_cumac_msg_fn_v_t fn, void* args)
{
  if ( has_sent_tti_end  || (type == CUMAC_CONFIG_REQUEST && has_sent_conf_req) || (type == CUMAC_SCH_TTI_REQUEST && waiting_tti_resp)) {
    return false;
  }
  if (type == CUMAC_SCH_TTI_REQUEST && !waiting_tti_resp) {
    waiting_tti_resp = true;
    cumac_sch_tti_req_args_t *sch_tti_args =  args;
    slot_data_entry_t *slot_data_entry = &slot_data[sch_tti_args->frame][sch_tti_args->slot];
    slot_data_entry->nActiveUe = sch_tti_args->payload.nActiveUe;
    slot_data_entry->allocSolSize = sch_tti_args->payload.nActiveUe * 2; // for Type 1 allocation
    slot_data_entry->nMaxSchUePerCell = nMaxSchUePerCell;
    slot_data_entry->taskBitMask = sch_tti_args->payload.taskBitMask;
  }
  if (type == CUMAC_TTI_END) {
    has_sent_tti_end = true;
  }
  if (type == CUMAC_CONFIG_REQUEST) {
    has_sent_conf_req = true;
    printf("Got CONFIG.request, initializing slot data matrix\n");
    // Determine number of slots per frame from SCS
    const cumac_config_req_payload_t* conf_req_payload = args;
    // Save nMax UE and allocation type for later
    nMaxSchUePerCell = conf_req_payload->nMaxSchUePerCell;
    allocType = conf_req_payload->allocType;
    nPrbPerPrg = conf_req_payload->nPrbPerPrg;
    const uint8_t num_slots = 10 << conf_req_payload->scSpacing;
    printf("Number of slots per frame: %d\n", num_slots);
    // Init slot data structure for later use
    for (int i = 0; i < MAX_FRAME_NUMBER; ++i) {
      printf("Initializing frame %d\n", i);
      slot_data[i] = calloc(num_slots, sizeof(slot_data_entry_t));
    }
  }

#ifdef CUMAC_TIMING_DEBUG
  printf("message 0x%02x\n", type);
  struct timespec t1, t2, t3, t4;
  clock_gettime(CLOCK_REALTIME, &t1);
#endif
  nv_ipc_msg_t send_msg;
  send_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
  if (ipc->tx_allocate(ipc, &send_msg, 0) != 0) {
    NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "NVIPC memory pool full");
    return NULL;
  }
#ifdef CUMAC_TIMING_DEBUG
  clock_gettime(CLOCK_REALTIME, &t2);
#endif
  if (fn(type,&send_msg, args) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return NULL;
  }

#ifdef CUMAC_TIMING_DEBUG
  clock_gettime(CLOCK_REALTIME, &t3);
#endif
  if (ipc->tx_send_msg(ipc, &send_msg) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return NULL;
  }
  ipc->tx_tti_sem_post(ipc);

  if (type == CUMAC_SCH_TTI_REQUEST) {
    const cumac_sch_tti_req_args_t *sch_tti_args =  args;
    slot_data_entry_t *slot_data_entry = &slot_data[sch_tti_args->frame][sch_tti_args->slot];
    clock_gettime(CLOCK_REALTIME, &slot_data_entry->tti_req_timestamp);
  }
  if (type == CUMAC_TTI_END) {
    const cumac_sch_tti_end_args_t *tti_end_args =  args;
    slot_data_entry_t *slot_data_entry = &slot_data[tti_end_args->frame][tti_end_args->slot];
    clock_gettime(CLOCK_REALTIME, &slot_data_entry->tti_end_timestamp);
  }

#ifdef CUMAC_TIMING_DEBUG
  clock_gettime(CLOCK_REALTIME, &t4);
  printf("message 0x%02x\n", type);
  printf("alloc: %ld ns\n", nvlog_timespec_interval(&t1, &t2));
  printf("build: %ld ns\n", nvlog_timespec_interval(&t2, &t3));
  printf("send : %ld ns\n", nvlog_timespec_interval(&t3, &t4));
#endif
  return true;
}

void* epoll_recv_task(void* arg) {
  struct epoll_event ev, events[MAX_EVENTS];

  printf("Aerial cuMAC recv task start \n");

  int epoll_fd = epoll_create1(0);
  if (epoll_fd == -1) {
    printf("%s epoll_create failed\n", __func__);
    return NULL;
  }

  int ipc_rx_event_fd = ipc->get_fd(ipc);
  ev.events = EPOLLIN;
  ev.data.fd = ipc_rx_event_fd;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ev.data.fd, &ev) == -1) {
    printf("%s epoll_ctl failed\n", __func__);
    return NULL;
  }
  while (1) {
    if (!recv_task_running) {
      recv_task_running = true;
      printf("%s: epoll_recv_task started\n", __func__);
    }
    printf("%s: epoll_wait fd_rx=%d ...\n", __func__, ipc_rx_event_fd);

    int nfds;
    do {
      // epoll_wait() may return EINTR when get unexpected signal SIGSTOP from system
      nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
    } while (nfds == -1 && errno == EINTR);

    if (nfds < 0) {
      printf("epoll_wait failed: epoll_fd=%d nfds=%d err=%d - %s\n", epoll_fd, nfds, errno, strerror(errno));
    }

    int n = 0;
    for (n = 0; n < nfds; ++n) {
      if (events[n].data.fd == ipc_rx_event_fd) {
        ipc->get_value(ipc);
        nv_ipc_msg_t recv_msg;
        while (cumac_recv_msg(&recv_msg) == 0)
          ;
      }
    }
  }
  close(epoll_fd);
  return NULL;
}

bool cumac_nvipc_init() {
  if (ipc != NULL) {
    return false;
  }
  // Create configuration
  nv_ipc_config.ipc_transport = NV_IPC_TRANSPORT_SHM;
  if (set_nv_ipc_default_config(&nv_ipc_config, NV_IPC_MODULE_SECONDARY) < 0) {
    printf("%s: set configuration failed\n", __func__);
    return false;
  }

  int cuda_device_id = -1;
  printf("CUDA device ID configured : %d \n", cuda_device_id);
  nv_ipc_config.transport_config.shm.cuda_device_id = cuda_device_id;
  // Create nv_ipc_t instance
  printf("%s: creating IPC interface with prefix %s\n", __func__, "cumac");
  strcpy(nv_ipc_config.transport_config.shm.prefix, "cumac");
  if ((ipc = create_nv_ipc_interface(&nv_ipc_config)) == NULL) {
    printf("%s: create IPC interface failed\n", __func__);
    return false;
  }
  printf("%s: create IPC interface successful\n", __func__);
  // setup receiver thread
  sleep(1);

  mutexinit(cumac_can_send_mutex);
  mutexinit(cumac_can_process_mutex);
  // Set initial lock state
  pthread_mutex_unlock(&cumac_can_send_mutex);
  pthread_mutex_lock(&cumac_can_process_mutex);

  pthread_t thread_id;
  pthread_create(&thread_id, NULL, epoll_recv_task, NULL);
return true;
}

int cumac_recv_msg(nv_ipc_msg_t* recv_msg)
{
  if (ipc == NULL) {
    return -1;
  }
  recv_msg->msg_buf = NULL;
  recv_msg->data_buf = NULL;

  // Allocate buffer for TX message
  if (ipc->rx_recv_msg(ipc, recv_msg) < 0) {
    printf("%s: no more message available\n", __func__);
    return -1;
  }

  printf("recv: cell_id=%d msg_id=0x%02X msg_len=%d data_len=%d data_pool=%d\n",
         recv_msg->cell_id,
         recv_msg->msg_id,
         recv_msg->msg_len,
         recv_msg->data_len,
         recv_msg->data_pool);

  cumac_handle_rx_msg(recv_msg);


  // Release buffer of RX message
  int release_retval = ipc->rx_release(ipc, recv_msg);
  if (release_retval != 0) {
    printf("%s error: release RX buffer failed Error: %d\n", __FUNCTION__, release_retval);
    return release_retval;
  }
  return 0;
}

void cumac_handle_rx_msg(nv_ipc_msg_t* recv_msg) {
  switch (recv_msg->msg_id) {
    case CUMAC_CONFIG_RESPONSE:
      // Send Start request
      cumac_send_msg(CUMAC_START_REQUEST,l2_build_start_request, NULL);
    break;
    case CUMAC_START_RESPONSE:
      cumac_start = true;
    break;
    case CUMAC_SCH_TTI_RESPONSE: {
      // Handle SCH_TTI.request
      cumac_sch_tti_resp_t* resp = recv_msg->msg_buf;
      cumac_handle_sch_tti_response(CUMAC_SCH_TTI_RESPONSE, recv_msg, &slot_data[resp->sfn][resp->slot]);
      waiting_tti_resp = false;
      has_sent_tti_end = false;
      cumac_allow_send();
      cumac_allow_process();
      break;
    }
    default:
    break;
  }
}

void cumac_wait_to_send() {
  pthread_mutex_lock(&cumac_can_send_mutex);
}

void cumac_allow_send() {
  pthread_mutex_unlock(&cumac_can_send_mutex);
}

void cumac_wait_to_process() {
  pthread_mutex_lock(&cumac_can_process_mutex);
}

void cumac_allow_process() {
  pthread_mutex_unlock(&cumac_can_process_mutex);
}

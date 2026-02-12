

#include "cumac_test.h"
#include <stdio.h>
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

#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "cu_mac_api.h"
#include "cumac_msg_funcs.h"

/* YAML config path */
#define YAML_CONFIG_PATH "./cuMAC-CP/examples/L2IntegrationExample/l2_nvipc.yaml"
#define MAX_PATH_LEN (1024)

nv_ipc_config_t nv_ipc_config;

bool can_start = false;
/* C replacement for std::pair<uint16_t,uint8_t> */
typedef struct {
  uint16_t ue;
  uint8_t lc;
} scheduled_ue_lc;

/* Log TAG configured in nvlog */
static int TAG = (NVLOG_TAG_BASE_NVIPC + 0);

nv_ipc_t* ipc = NULL;

#define SENDER_THREAD_CORE (8)
#define RECV_THREAD_CORE (9)

////////////////////////////////////////////////////////////////////////
// Build a TX message
static int l2_build_tx_msg(nv_ipc_msg_t* nvipc_buf,
                           uint32_t slot,
                           uint32_t sfn,
                           uint32_t num_ue,
                           struct PFM_UE_INFO* test_data)
{
  if (ipc == NULL || nvipc_buf == NULL) {
    NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: ipc or msg buffer is NULL", __func__);
    return -1;
  }

  struct cumac_pfm_tti_req_t* req = (struct cumac_pfm_tti_req_t*)nvipc_buf->msg_buf;
  uint8_t* data_buf = (uint8_t*)nvipc_buf->data_buf;

  req->header.type_id = CUMAC_SCH_TTI_REQUEST;
  req->sfn = sfn;
  req->slot = slot;
  req->num_ue = num_ue;

  for (int i = 0; i < 10; i++) {
    req->num_output_sorted_lc[i] = MAX_NUM_OUTPUT_SORTED_LC_PER_QOS;
  }

  uint32_t offset = 0;
  uint32_t data_size = sizeof(struct PFM_UE_INFO) * num_ue;

  req->offset_ue_info_arr = offset;
  memcpy(data_buf + offset, test_data, data_size);
  offset += data_size;

  nvipc_buf->msg_id = CUMAC_SCH_TTI_REQUEST;
  nvipc_buf->cell_id = 0;
  nvipc_buf->msg_len = sizeof(struct cumac_pfm_tti_req_t);
  nvipc_buf->data_len = data_size;

  return 0;
}

////////////////////////////////////////////////////////////////////////

static int test_l2_send_slot(uint32_t slot, uint32_t sfn, uint32_t num_ue, struct PFM_UE_INFO* test_data)
{
  struct timespec t1, t2, t3, t4, t5, t6;

  clock_gettime(CLOCK_REALTIME, &t1);

  nv_ipc_msg_t send_msg;
  send_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

  if (ipc->tx_allocate(ipc, &send_msg, 0) != 0) {
    NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "NVIPC memory pool full");
    return -1;
  }

  clock_gettime(CLOCK_REALTIME, &t2);

  clock_gettime(CLOCK_REALTIME, &t3);
  if (l2_build_tx_msg(&send_msg, slot, sfn, num_ue, test_data) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return -1;
  }
  clock_gettime(CLOCK_REALTIME, &t4);

  clock_gettime(CLOCK_REALTIME, &t5);
  if (ipc->tx_send_msg(ipc, &send_msg) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return -1;
  }

  ipc->tx_tti_sem_post(ipc);
  clock_gettime(CLOCK_REALTIME, &t6);

  printf("alloc: %ld ns\n", nvlog_timespec_interval(&t1, &t2));
  printf("build: %ld ns\n", nvlog_timespec_interval(&t3, &t4));
  printf("send : %ld ns\n", nvlog_timespec_interval(&t5, &t6));

  return 0;
}

static void get_next_slot_timespec(struct timespec* ts, uint64_t interval_nsec)
{
  ts->tv_nsec += interval_nsec;
  while (ts->tv_nsec >= 1000000000L) {
    ts->tv_nsec -= 1000000000L;
    ts->tv_sec++;
  }
}

////////////////////////////////////////////////////////////////////////
// C versions of prepare_slot_data_dl / ul

void prepare_slot_data_dl(struct PFM_UE_INFO* test_data, scheduled_ue_lc* sched_list, int slot_idx)
{
  if (slot_idx == 0) {
    for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
      sched_list[i].ue = i;
      sched_list[i].lc = rand() % MAX_NUM_LC;
    }

    for (int u = 0; u < MAX_NUM_UE; u++) {
      for (int lc = 0; lc < MAX_NUM_LC; lc++) {
        struct PFM_DL_LC_INFO* p = &test_data[u].dl_lc_info[lc];
        p->pfm = 0;
        p->ravg = 1;
        p->flags = 0x03;
        p->qos_type = rand() % 5;
        p->tbs_scheduled = 0;
      }
      test_data[u].num_dl_lcs = MAX_NUM_LC;
      test_data[u].ambr = 50000000;
      test_data[u].rcurrent_dl = 1000000 + rand() % 20000000;
      test_data[u].rnti = u;
      test_data[u].num_layers_dl = rand() % 2 + 1;
      test_data[u].flags = 0x07;
      test_data[u].carrier_id = 0;
    }
  } else {
    for (int u = 0; u < MAX_NUM_UE; u++)
      for (int lc = 0; lc < MAX_NUM_LC; lc++)
        test_data[u].dl_lc_info[lc].tbs_scheduled = 0;

    for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
      uint16_t ue = sched_list[i].ue;
      uint8_t lc = sched_list[i].lc;

      test_data[ue].dl_lc_info[lc].tbs_scheduled = (uint32_t)floorf((float)test_data[ue].rcurrent_dl * SLOT_DURATION);

      sched_list[i].ue = (ue + MAX_NUM_SCHEDULED_UE) % MAX_NUM_UE;
      sched_list[i].lc = rand() % MAX_NUM_LC;
    }

    for (int u = 0; u < MAX_NUM_UE; u++) {
      test_data[u].rcurrent_dl = 1000000 + rand() % 20000000;
      test_data[u].num_layers_dl = rand() % 2 + 1;
      for (int lc = 0; lc < MAX_NUM_LC; lc++)
        test_data[u].dl_lc_info[lc].flags = 0x01;
    }
  }
}

////////////////////////////////////////////////////////////////////////

void prepare_slot_data_ul(struct PFM_UE_INFO* test_data, scheduled_ue_lc* sched_list, int slot_idx)
{
  if (slot_idx == 0) {
    for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
      sched_list[i].ue = i;
      sched_list[i].lc = rand() % MAX_NUM_LCG;
    }

    for (int u = 0; u < MAX_NUM_UE; u++) {
      for (int g = 0; g < MAX_NUM_LCG; g++) {
        struct PFM_UL_LCG_INFO* p = &test_data[u].ul_lcg_info[g];
        p->pfm = 0;
        p->ravg = 1;
        p->flags = 0x03;
        p->qos_type = rand() % 5;
        p->tbs_scheduled = 0;
      }
      test_data[u].num_ul_lcgs = MAX_NUM_LCG;
      test_data[u].ambr = 20000000;
      test_data[u].rcurrent_ul = 500000 + rand() % 10000000;
      test_data[u].rnti = u;
      test_data[u].num_layers_ul = rand() % 2 + 1;
      test_data[u].flags = 0x07;
      test_data[u].carrier_id = 0;
    }
  } else {
    for (int u = 0; u < MAX_NUM_UE; u++)
      for (int g = 0; g < MAX_NUM_LCG; g++)
        test_data[u].ul_lcg_info[g].tbs_scheduled = 0;

    for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
      uint16_t ue = sched_list[i].ue;
      uint8_t g = sched_list[i].lc;

      test_data[ue].ul_lcg_info[g].tbs_scheduled = (uint32_t)floorf((float)test_data[ue].rcurrent_ul * SLOT_DURATION);

      sched_list[i].ue = (ue + MAX_NUM_SCHEDULED_UE) % MAX_NUM_UE;
      sched_list[i].lc = rand() % MAX_NUM_LCG;
    }

    for (int u = 0; u < MAX_NUM_UE; u++) {
      test_data[u].rcurrent_ul = 500000 + rand() % 10000000;
      test_data[u].num_layers_ul = rand() % 2 + 1;
      for (int g = 0; g < MAX_NUM_LCG; g++)
        test_data[u].ul_lcg_info[g].flags = 0x01;
    }
  }
}

////////////////////////////////////////////////////////////////////////

int load_hard_code_config(nv_ipc_config_t* config, int module_type, nv_ipc_transport_t _transport)
{
  // Create configuration
  config->ipc_transport = _transport;
  if (set_nv_ipc_default_config(config, module_type) < 0) {
    printf("%s: set configuration failed\n", __func__);
    return -1;
  }

  int test_cuda_device_id = -1;
  printf("CUDA device ID configured : %d \n", test_cuda_device_id);
  config->transport_config.shm.cuda_device_id = test_cuda_device_id;
  if (test_cuda_device_id >= 0) {
    config->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = 128;
    config->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_DATA].pool_len = 1024;
    config->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].pool_len = 4096;
  }

  return 0;
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  nvIPC_Init();
  /*// Want to use transport SHM, type epoll, module secondary (reads the created shm from cuphycontroller)
  load_hard_code_config(&nv_ipc_config, NV_IPC_MODULE_SECONDARY, NV_IPC_TRANSPORT_SHM);
  // Create nv_ipc_t instance
  printf("%s: creating IPC interface with prefix %s\n", __func__, "cumac");
  strcpy(nv_ipc_config.transport_config.shm.prefix, "cumac");
  if ((ipc = create_nv_ipc_interface(&nv_ipc_config)) == NULL) {
    printf("%s: create IPC interface failed\n", __func__);
    return -1;
  }
  printf("%s: create IPC interface successful\n", __func__);


  pthread_t tid;
  pthread_create(&tid, NULL, l2_blocking_recv_task, NULL);
 */
  sleep(1);

  usleep(1000000);
  while (!can_start) {
    sleep(1);
  }
  static struct PFM_UE_INFO test_data[MAX_NUM_UE];
  static scheduled_ue_lc sched_dl[MAX_NUM_SCHEDULED_UE];
  static scheduled_ue_lc sched_ul[MAX_NUM_SCHEDULED_UE];

  uint16_t sfn = 0, slot = 0;

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_nsec = 0;
  ts.tv_sec++;

  // pthread_setname_np(pthread_self(), "l2_sender");
  //nv_set_sched_fifo_priority(80);
  //nv_assign_thread_cpu_core(SENDER_THREAD_CORE);

  for (int i = 0; i < NUM_TIME_SLOTS; i++) {
    clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, NULL);

    prepare_slot_data_dl(test_data, sched_dl, i);
    prepare_slot_data_ul(test_data, sched_ul, i);

    test_l2_send_slot(slot, sfn, MAX_NUM_UE, test_data);

    slot++;
    if (slot >= 20) {
      slot = 0;
      sfn = (sfn + 1) % 1024;
    }

    get_next_slot_timespec(&ts, SLOT_INTERVAL_NS);
  }

  return 0;
}


bool recv_task_running = false;
nv_ipc_config_t nv_ipc_config;
#define MAX_EVENTS 10
#define RECV_BUF_LEN 8192
char cpu_buf_recv[RECV_BUF_LEN];
//nv_ipc_t* ipc;
typedef int (*build_cumac_msg_fn_v_t)(cumac_msg_t type,nv_ipc_msg_t *nvipc_buf,  void * args);
bool cumac_test_send_msg(cumac_msg_t type,build_cumac_msg_fn_v_t fn,  void* args)
{
  struct timespec t1, t2, t3, t4;
  clock_gettime(CLOCK_REALTIME, &t1);
  nv_ipc_msg_t send_msg;
  send_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
  if (ipc->tx_allocate(ipc, &send_msg, 0) != 0) {
    NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "NVIPC memory pool full");
    return NULL;
  }
  clock_gettime(CLOCK_REALTIME, &t2);
  if (fn(type,&send_msg, args) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return NULL;
  }
  clock_gettime(CLOCK_REALTIME, &t3);
  if (ipc->tx_send_msg(ipc, &send_msg) < 0) {
    ipc->tx_release(ipc, &send_msg);
    return NULL;
  }
  ipc->tx_tti_sem_post(ipc);
  clock_gettime(CLOCK_REALTIME, &t4);
  printf("alloc: %ld ns\n", nvlog_timespec_interval(&t1, &t2));
  printf("build: %ld ns\n", nvlog_timespec_interval(&t2, &t3));
  printf("send : %ld ns\n", nvlog_timespec_interval(&t3, &t4));

  return true;
}

void ipc_handle_rx_msg(nv_ipc_msg_t* recv_msg) {
  switch (recv_msg->msg_id) {
    case CUMAC_CONFIG_RESPONSE:
    // Send Start request
      cumac_test_send_msg(CUMAC_START_REQUEST,l2_build_start_request, NULL);
    case CUMAC_START_RESPONSE:
    can_start = true;

    default:;
  }
}
// Always allocate message buffer, but allocate data buffer only when data_len > 0
static int cumac_recv_msg(nv_ipc_msg_t* recv_msg)
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

   ipc_handle_rx_msg(recv_msg);


  // Release buffer of RX message
  int release_retval = ipc->rx_release(ipc, recv_msg);
  if (release_retval != 0) {
    printf("%s error: release RX buffer failed Error: %d\n", __FUNCTION__, release_retval);
    return release_retval;
  }
  return 0;
}



void* epoll_recv_task(void* arg)
{
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
      cumac_config_req_payload_t conf_req_payload;
      conf_req_payload.harqEnabledInd = 0;
      conf_req_payload.mcsSelCqi = 0;
      conf_req_payload.nMaxCell = 1;
      conf_req_payload.nMaxActUePerCell = 16;
      conf_req_payload.nMaxSchUePerCell = 1;
      conf_req_payload.nMaxPrg = 1;
      conf_req_payload.nPrbPerPrg = 255;
      conf_req_payload.nMaxBsAnt = 4;
      conf_req_payload.nMaxUeAnt = 4;
      conf_req_payload.scSpacing = 1;
      conf_req_payload.allocType = 1;
      conf_req_payload.precoderType = 0;
      conf_req_payload.receiverType = 0;
      conf_req_payload.colMajChanAccess = 0;
      conf_req_payload.betaCoeff = 1;
      conf_req_payload.sinValThr = 0.1f;
      conf_req_payload.corrThr = 0.1f;
      conf_req_payload.mcsSelSinrCapThr = 0;
      conf_req_payload.mcsSelLutType = 1;
      conf_req_payload.prioWeightStep = 1;
      cumac_test_send_msg(CUMAC_CONFIG_REQUEST,l2_build_config_request, &conf_req_payload);
      printf("%s: cumac_send_msg called\n", __func__);
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

int nvIPC_Init()
{
  // Want to use transport SHM, type epoll, module secondary (reads the created shm from cuphycontroller)
  load_hard_code_config(&nv_ipc_config, NV_IPC_MODULE_SECONDARY, NV_IPC_TRANSPORT_SHM);
  // Create nv_ipc_t instance
  printf("%s: creating IPC interface with prefix %s\n", __func__, "cumac");
  strcpy(nv_ipc_config.transport_config.shm.prefix, "cumac");
  if ((ipc = create_nv_ipc_interface(&nv_ipc_config)) == NULL) {
    printf("%s: create IPC interface failed\n", __func__);
    return -1;
  }
  printf("%s: create IPC interface successful\n", __func__);
  sleep(1);
  pthread_t thread_id;
  pthread_create(&thread_id, NULL, epoll_recv_task, NULL);
  //threadCreate(&thread_id, epoll_recv_task, NULL, "nvipc_cumac", 13, OAI_PRIORITY_RT);

  while (!recv_task_running) {
    usleep(100000);
  }

  return 0;
} /*
 void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
 {
   exit(1);
 }*/
// int main(int argc, char* argv[])
// {
//   nvIPC_Init();
//   sleep(300);
//   return 0;
// }

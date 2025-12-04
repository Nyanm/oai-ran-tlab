#/*
# * Licensed to the EPYSYS SCIENCE (EpiSci) under one or more
# * contributor license agreements.
# * The EPYSYS SCIENCE (EpiSci) licenses this file to You under
# * the Episys Science (EpiSci) Public License (Version 1.1) (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      https://github.com/EpiSci/oai-lte-5g-multi-ue-proxy/blob/master/LICENSE
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about EPYSYS SCIENCE (EpiSci):
# *      bo.ryu@episci.com
# *-------------------------------------------------------------------------------
# * Modified by Russell Ford, Samsung Research America, russell.ford@samsung.com
# */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/sctp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <string>
#include <cstring>
#include <mutex>
#include <assert.h>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <non_blocking_queue.h>
#include "proxy.h"
#include "nfapi_pnf.h"
// #include "../../openair2/NR_UE_PHY_INTERFACE/NR_Packet_Drop.h"

#ifdef __cplusplus
extern "C" {
#endif
    #include "queue.h"
#ifdef __cplusplus
}
#endif

typedef struct 
{
    uint16_t msg_id;
    int msg_len;
    void *msg;
    void *msg_buf;
    std::atomic<int> ref_count;
} downlink_nfapi_msg_wrapper_t;

class Multi_UE_NR_Proxy
{
public:
    Multi_UE_NR_Proxy(std::vector<std::string> gnb_ips, std::string proxy_ip, std::vector<std::string> ue_ips, const char* ch_trace_path);
    ~Multi_UE_NR_Proxy() = default;
    void configure(std::vector<std::string> gnb_ip, std::string proxy_ip, std::vector<std::string> ue_ip);
    int init_oai_socket(const char *addr, int tx_port, int rx_port, int ue_idx);
    void oai_gnb_downlink_nfapi_task(void *msg, uint16_t msg_id);
    void oai_gnb_downlink_nfapi_task_par(int ue_idx);
    // void enqueue_downlink_nfapi_msg(void *msg, uint16_t msg_id);
    void enqueue_downlink_nfapi_msg(void *msg_buf, int msg_len, uint16_t msg_id);
    void testcode_tx_packet_to_UE( int ue_tx_socket_);
    void pack_and_send_downlink_sfn_slot_msg(uint16_t sfn_slot);
    void pack_and_send_downlink_sfn_slot_ch_info_msg(nr_phy_channel_params_t *ch_info, int ue_idx);
    void receive_message_from_nr_ue(int ue_id);
    void send_nr_ue_to_gnb_msg(void *buffer, size_t buflen);
    void send_received_msg_to_proxy_queue(void *buffer, size_t buflen);
    void send_uplink_oai_msg_to_proxy_queue(void *buffer, size_t buflen);
    void start(softmodem_mode_t softmodem_mode);

    bool stop_threads;

private:
    std::vector<std::string> oai_ue_ipaddr;
    std::vector<std::string> vnf_ipaddr;
    std::string pnf_ipaddr;
    int vnf_p5port = -1;
    int vnf_p7port = -1;
    int pnf_p7port = -1;

    eth_params_t gnb_conn_info;
    eth_params_t proxy_conn_info;

    std::uint16_t u16SequenceNumber_ = 0;
    // struct sockaddr_in address_tx_;
    struct sockaddr_in address_tx_[MAX_UES];
    struct sockaddr_in address_rx_;
    int ue_tx_socket_ = -1;
    int ue_rx_socket_ = -1;
    int ue_rx_socket[MAX_UES];
    int ue_tx_socket[MAX_UES];
    std::uint16_t id;
    std::recursive_mutex mutex;
    using lock_guard_t = std::lock_guard<std::recursive_mutex>;
    std::vector<std::thread> threads;
    bool stop_thread = false;
    int port_delta = 2;

    // queue_t dl_msg_queue[MAX_UES];
    // downlink_nfapi_msg_wrapper_t *msg_wrapper;
    NonBlockingQueue<downlink_nfapi_msg_wrapper_t*> dl_msg_queue[MAX_UES];
    downlink_nfapi_msg_wrapper_t curr_msg_wrapper[MAX_UES];
};

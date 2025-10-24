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

#include <sys/stat.h>
#include <sstream>
#include "../../common/platform_types.h"
#include "nr_proxy.h"
#include "nfapi_pnf.h"

#ifdef __cplusplus
extern "C" {
#endif
    #include "nr_fapi_p7_utils.h"
    #include "queue.h"
#ifdef __cplusplus
}
#endif

namespace
{
    Multi_UE_NR_Proxy *instance;
}

Multi_UE_NR_Proxy::Multi_UE_NR_Proxy(std::vector<std::string> gnb_ips, std::string proxy_ip, std::vector<std::string> ue_ips, const char* ch_trace_path)
{
    assert(instance == NULL);
    instance = this;
    num_ues = ue_ips.size();
    stop_threads = false;
    // msg_wrapper = (downlink_nfapi_msg_wrapper_t*)malloc(sizeof(downlink_nfapi_msg_wrapper_t));

    configure(gnb_ips, proxy_ip, ue_ips);

    oai_slot_init();

    pthread_t ch_trace_thread;
    if (ch_trace_path && strcmp(ch_trace_path, "") != 0) {
        if (pthread_create(&ch_trace_thread, NULL, &ch_trace_task, (void *)ch_trace_path) != 0) {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "pthread_create failed for calling oai_slot_task");
        }
    }
}

void Multi_UE_NR_Proxy::start(softmodem_mode_t softmodem_mode)
{
    pthread_t thread;
    
    // configure_nr_nfapi_pnf(vnf_ipaddr.c_str(), vnf_p5port, pnf_ipaddr.c_str(), pnf_p7port, vnf_p7port);

    for (unsigned int gnb_idx = 0; gnb_idx < vnf_ipaddr.size(); gnb_idx++) {
        configure_nr_nfapi_pnf(vnf_ipaddr[gnb_idx].c_str(), vnf_p5port, pnf_ipaddr.c_str(), pnf_p7port, vnf_p7port, gnb_idx);
    }

    if (pthread_create(&thread, NULL, &oai_slot_task, (void *)softmodem_mode) != 0)
    {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "pthread_create failed for calling oai_slot_task");
    }
    
    for (int i = 0; i < num_ues; i++)
    {
        // init_queue(&dl_msg_queue[i]);
        threads.push_back(std::thread(&Multi_UE_NR_Proxy::receive_message_from_nr_ue, this, i));
        threads.push_back(std::thread(&Multi_UE_NR_Proxy::oai_gnb_downlink_nfapi_task_par, this, i));
    }
    threads.push_back(std::thread(&Multi_UE_NR_Proxy::oai_gnb_downlink_nfapi_task_par, this, num_ues));  // Additional thread to handle deallocation

    // for (int i = 0; i < 5; i++) {
    //     threads.push_back(std::thread(&Multi_UE_NR_Proxy::oai_gnb_downlink_nfapi_task_par, this, i));
    // }
    for (auto &th : threads)
    {
        if(th.joinable())
        {
            th.join();
        }
    }
}

void Multi_UE_NR_Proxy::configure(std::vector<std::string> gnb_ips, std::string proxy_ip, std::vector<std::string> ue_ips)
{
    oai_ue_ipaddr = ue_ips;
    vnf_ipaddr = gnb_ips;
    pnf_ipaddr = proxy_ip;
    vnf_p5port = 50601;
    vnf_p7port = 50611;
    pnf_p7port = 50810;

    std::cout<<"PNF is on IP Address "<<pnf_ipaddr<<std::endl;
    
    for (unsigned int gnb_idx = 0; gnb_idx < vnf_ipaddr.size(); gnb_idx++) {
        std::cout<<"VNF "<<gnb_idx<<" is on IP Address "<<vnf_ipaddr[gnb_idx]<<std::endl;
    }
    
    for (int ue_idx = 0; ue_idx < num_ues; ue_idx++)
    {
        std::cout<<"OAI-UE "<<ue_idx<<" is on IP Address "<<oai_ue_ipaddr[ue_idx]<<std::endl;
        int oai_rx_ue_port = 3611 + ue_idx * port_delta;
        int oai_tx_ue_port = 3612 + ue_idx * port_delta;
        init_oai_socket(oai_ue_ipaddr[ue_idx].c_str(), oai_tx_ue_port, oai_rx_ue_port, ue_idx);
    }
}

int Multi_UE_NR_Proxy::init_oai_socket(const char *addr, int tx_port, int rx_port, int ue_idx)
{
    {   //Setup Rx Socket
        memset(&address_rx_, 0, sizeof(address_rx_));
        address_rx_.sin_family = AF_INET;
        address_rx_.sin_addr.s_addr = INADDR_ANY;
        address_rx_.sin_port = htons(rx_port);

        ue_rx_socket_ = socket(address_rx_.sin_family, SOCK_DGRAM, 0);
        ue_rx_socket[ue_idx] = ue_rx_socket_;
        if (ue_rx_socket_ < 0)
        {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "socket: %s", ERR);
            return -1;
        }
        if (bind(ue_rx_socket_, (struct sockaddr *)&address_rx_, sizeof(address_rx_)) < 0)
        {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "bind failed in init_oai_socket: %s\n", strerror(errno));
            close(ue_rx_socket_);
            ue_rx_socket_ = -1;
            return -1;
        }
        NFAPI_TRACE(NFAPI_TRACE_DEBUG, "rx addr: %s, rx port: %d", addr, rx_port);
    }
    {   //Setup Tx Socket
        memset(&address_tx_[ue_idx], 0, sizeof(address_tx_[ue_idx]));
        address_tx_[ue_idx].sin_family = AF_INET;
        address_tx_[ue_idx].sin_port = htons(tx_port);

        if (inet_aton(addr, &address_tx_[ue_idx].sin_addr) == 0)
        {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "addr no good %s", addr);
            return -1;
        }

        ue_tx_socket_ = socket(address_tx_[ue_idx].sin_family, SOCK_DGRAM, 0);
        ue_tx_socket[ue_idx] = ue_tx_socket_;
        if (ue_tx_socket_ < 0)
        {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "socket: %s", ERR);
            return -1;
        }

        if (connect(ue_tx_socket_, (struct sockaddr *)&address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
        {
          NFAPI_TRACE(NFAPI_TRACE_ERROR, "tx connection failed in init_oai_socket: %s\n", strerror(errno));
          close(ue_tx_socket_);
          return -1;
        }
        NFAPI_TRACE(NFAPI_TRACE_DEBUG, "tx addr: %s, tx port: %d", addr, tx_port);
    }
    return 0;
}

void Multi_UE_NR_Proxy::receive_message_from_nr_ue(int ue_idx)
{
    char buffer[NFAPI_MAX_PACKED_MESSAGE_SIZE];
    socklen_t addr_len = sizeof(address_rx_);

    while(true)
    {
        int buflen = recvfrom(ue_rx_socket[ue_idx], buffer, sizeof(buffer), 0, (sockaddr *)&address_rx_, &addr_len);
        if (buflen == -1)
        {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "Recvfrom failed %s", strerror(errno));
            return ;
        }
        if (buflen == 4)
        {
            //NFAPI_TRACE(NFAPI_TRACE_INFO , "Dummy frame");
            continue;
        }
        else
        {
            nfapi_nr_p7_message_header_t header;
            if (nfapi_nr_p7_message_header_unpack(buffer, buflen, &header, sizeof(header), NULL) < 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Header unpack failed for standalone pnf");
                return ;
            }
            uint16_t sfn_slot = nfapi_get_sfnslot(MU, buffer, buflen);
            NFAPI_TRACE(NFAPI_TRACE_INFO , "(Proxy) Proxy has received %d uplink message from OAI UE at socket. Frame: %d, Slot: %d",
                    header.message_id, NFAPI_SFNSLOTDEC2SFN(MU, sfn_slot), NFAPI_SFNSLOTDEC2SLOT(MU, sfn_slot));
        }
        oai_slot_handle_msg_from_ue(buffer, buflen, ue_idx + 2);
    }
}

void Multi_UE_NR_Proxy::oai_gnb_downlink_nfapi_task(void *msg, uint16_t msg_id)
{
    // lock_guard_t lock(mutex);

    char buffer[NFAPI_MAX_PACKED_MESSAGE_SIZE];
    // int encoded_size = nfapi_nr_p7_message_pack(msg, buffer, sizeof(buffer), nullptr);
    // if (encoded_size <= 0)
    // {
    //     NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
    //     return;
    // }


    for(int ue_idx = 0; ue_idx < num_ues; ue_idx++)
    {
        inet_aton(oai_ue_ipaddr[ue_idx].c_str(), &address_tx_[ue_idx].sin_addr);
        address_tx_[ue_idx].sin_port = htons(3612 + ue_idx * port_delta);

        ue_slot_info_t *ue_info = &ue_slot_info[ue_idx];
        // memset (ue_info->valid_dci_inds, -1, sizeof(ue_info->valid_dci_inds));
        int encoded_size = -1;

        switch (msg_id)
        {

        case NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST:
        {
            const nfapi_nr_dl_tti_request_t *dl_tti_request = (nfapi_nr_dl_tti_request_t*)(msg);
            
            int dl_sfn = dl_tti_request->SFN;
            int dl_slot = dl_tti_request->Slot;
            uint16_t dl_numPDU = dl_tti_request->dl_tti_request_body.nPDUs;

            // if (ue_info->rnti != 0xFFFF) {
                // UE has been assigned C-RNTI
                int *valid_dci_ptr = ue_info->valid_dci_inds[dl_tti_request->Slot];
                nfapi_nr_dl_tti_request_t dl_tti_request_ue_spec;
                copy_dl_tti_request_for_rnti(dl_tti_request, &dl_tti_request_ue_spec, ue_info->rnti, valid_dci_ptr);
                if (dl_tti_request_ue_spec.dl_tti_request_body.nPDUs == 0) { // No PDUs for this RNTI
                    continue;
                }
                encoded_size = nfapi_nr_p7_message_pack(&dl_tti_request_ue_spec, buffer, sizeof(buffer), nullptr);
                if (encoded_size <= 0)
                {
                    NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                    return;
                }
            // }

            NFAPI_TRACE(NFAPI_TRACE_INFO , "(Proxy UE) Prior to sending dl_tti_req to OAI UE. Frame: %d,"
                    " Slot: %d, Number of PDUs: %u",
                    dl_sfn, dl_slot, dl_numPDU);
            assert(ue_tx_socket[ue_idx] > 2);
            if (sendto(ue_tx_socket[ue_idx], buffer, encoded_size, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST to OAI UE failed");
            }
            else
            {
                NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST forwarded from Proxy to UE");
            }
            break;
        }

        case NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST:
        {
            assert(ue_tx_socket[ue_idx] > 2);
            // if (ue_info->rnti != 0xFFFF) {
                const nfapi_nr_tx_data_request_t *tx_data_req = (nfapi_nr_tx_data_request_t*)(msg);
                int *valid_dci_ptr = ue_info->valid_dci_inds[tx_data_req->Slot];
                nfapi_nr_tx_data_request_t tx_data_req_ue_spec;
                copy_tx_data_request_for_rnti(tx_data_req, &tx_data_req_ue_spec, valid_dci_ptr);
                if (tx_data_req_ue_spec.Number_of_PDUs == 0) { // No PDUs for this RNTI
                    continue;
                }
                encoded_size = nfapi_nr_p7_message_pack(&tx_data_req_ue_spec, buffer, sizeof(buffer), nullptr);
                if (encoded_size <= 0)
                {
                    NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                    return;
                }
            // }

            if (sendto(ue_tx_socket[ue_idx], buffer, encoded_size, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST to OAI UE failed");
            }
            else
            {
                NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST forwarded from Proxy to UE");
            }
            break;
        }

        case NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST:
            assert(ue_tx_socket[ue_idx] > 2);
            encoded_size = nfapi_nr_p7_message_pack(msg, buffer, sizeof(buffer), nullptr);
            if (encoded_size <= 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                return;
            }
            if (sendto(ue_tx_socket[ue_idx], buffer, encoded_size, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST to OAI UE failed");
            }
            else
            {
                NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST forwarded from Proxy to UE");
            }
            break;

        case NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST:
            assert(ue_tx_socket[ue_idx] > 2);
            encoded_size = nfapi_nr_p7_message_pack(msg, buffer, sizeof(buffer), nullptr);
            if (encoded_size <= 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                return;
            }
            if (sendto(ue_tx_socket[ue_idx], buffer, encoded_size, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
            {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST to OAI UE failed");
            }
            else
            {
                NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST forwarded from Proxy to UE");
            }
            break;

        default:
            NFAPI_TRACE(NFAPI_TRACE_INFO , "Unhandled message at Proxy message_id: %u", msg_id);
            break;
        }
    }
}

void Multi_UE_NR_Proxy::enqueue_downlink_nfapi_msg(void *msg_buf, int msg_len, uint16_t msg_id)
{
    void *msg;
    switch (msg_id) {
        case NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST:
        {
            msg = malloc(sizeof(nfapi_nr_dl_tti_request_t));
            // msg = get_queue(&recv_buf_pool.free_pool);
            if (!msg) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s Could not allocate NFAPI message.\n", __FUNCTION__);
                return;
            }
            int unpack_result = nfapi_nr_p7_message_unpack(msg_buf, msg_len, msg, sizeof(nfapi_nr_dl_tti_request_t), nullptr);
            if (unpack_result != 0) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Failed to unpack dl_tti_req");
                return;
            }
            break;
        }
        case NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST:
        {
            msg = malloc(sizeof(nfapi_nr_tx_data_request_t));
            // msg = get_queue(&recv_buf_pool.free_pool);
            if (!msg) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s Could not allocate NFAPI message.\n", __FUNCTION__);
                return;
            }
            int unpack_result = nfapi_nr_p7_message_unpack(msg_buf, msg_len, msg, sizeof(nfapi_nr_tx_data_request_t), nullptr);
            if (unpack_result != 0) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Failed to unpack tx_data_req");
                return;
            }
            break;
        }
        case NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST:
        {
            msg = malloc(sizeof(nfapi_nr_ul_tti_request_t));
            // msg = get_queue(&recv_buf_pool.free_pool);
            if (!msg) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s Could not allocate NFAPI message.\n", __FUNCTION__);
                return;
            }
            int unpack_result = nfapi_nr_p7_message_unpack(msg_buf, msg_len, msg, sizeof(nfapi_nr_ul_tti_request_t), nullptr);
            if (unpack_result != 0) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Failed to unpack ul_tti_req");
                return;
            }
            break;
        }
        case NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST:
        {
            msg = malloc(sizeof(nfapi_nr_ul_dci_request_t));
            // msg = get_queue(&recv_buf_pool.free_pool);
            if (!msg) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s Could not allocate NFAPI message.\n", __FUNCTION__);
                return;
            }
            int unpack_result = nfapi_nr_p7_message_unpack(msg_buf, msg_len, msg, sizeof(nfapi_nr_ul_dci_request_t), nullptr);
            if (unpack_result != 0) {
                NFAPI_TRACE(NFAPI_TRACE_ERROR, "Failed to unpack ul_dci_req");
                return;
            }
            break;
        }
        default:
            NFAPI_TRACE(NFAPI_TRACE_INFO , "Unhandled message at Proxy message_id: %u", msg_id);
            return;
    }
    

    downlink_nfapi_msg_wrapper_t *msg_wrapper = (downlink_nfapi_msg_wrapper_t*)malloc(sizeof(downlink_nfapi_msg_wrapper_t));
    msg_wrapper->msg = msg;
    msg_wrapper->msg_id = msg_id;
    msg_wrapper->msg_len = msg_len;
    msg_wrapper->msg_buf = msg_buf;
    msg_wrapper->ref_count = num_ues;
    
    for (int i = 0; i < num_ues+1; ++i) {
        dl_msg_queue[i].push(msg_wrapper);
    }
}

void Multi_UE_NR_Proxy::oai_gnb_downlink_nfapi_task_par(int ue_idx)
{
    char buffer[NFAPI_MAX_PACKED_MESSAGE_SIZE];

    while (!stop_threads) 
    {
        downlink_nfapi_msg_wrapper_t *msg_wrapper = dl_msg_queue[ue_idx].pop();
        if (!msg_wrapper) {
            NFAPI_TRACE(NFAPI_TRACE_ERROR, "DL msg queue returned null msg_wrapper.");
            continue;
        }

        // NFAPI_TRACE(NFAPI_TRACE_DEBUG, "Downlink NFAPI task thread %d processing message", thread_idx);

        void *msg = msg_wrapper->msg;
        void *msg_buf = msg_wrapper->msg_buf;
        uint16_t msg_id = msg_wrapper->msg_id;
        int msg_len = msg_wrapper->msg_len;

        if (ue_idx == num_ues) {  // Thread dedicated to deallocating msg buffer
            while (msg_wrapper->ref_count > 0) {
                usleep(200);
            }
            put_queue(&recv_buf_pool.free_pool, msg_wrapper->msg_buf);
            // put_queue(&recv_buf_pool.free_pool, msg_wrapper->msg);
            free(msg_wrapper->msg);
            free(msg_wrapper);
            continue;
        }

        bool send_msg = true;

        ue_slot_info_t *ue_info = &ue_slot_info[ue_idx];

        switch (msg_id)
        {
            case NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST:
            {
                nfapi_nr_dl_tti_request_t *dl_tti_request = (nfapi_nr_dl_tti_request_t*)(msg);

                int dl_sfn = dl_tti_request->SFN;
                int dl_slot = dl_tti_request->Slot;
                uint16_t dl_numPDU = dl_tti_request->dl_tti_request_body.nPDUs;

                // if (ue_info->rnti != 0xFFFF) {
                    // UE has been assigned C-RNTI
                    int *valid_dci_ptr = ue_info->valid_dci_inds[dl_slot];
                    nfapi_nr_dl_tti_request_t dl_tti_request_ue_spec;
                    copy_dl_tti_request_for_rnti(dl_tti_request, &dl_tti_request_ue_spec, ue_info->rnti, valid_dci_ptr);
                    if (dl_tti_request_ue_spec.dl_tti_request_body.nPDUs != 0) { // PDUs for this RNTI
                        msg_len = nfapi_nr_p7_message_pack(&dl_tti_request_ue_spec, buffer, msg_len, nullptr);
                        msg_buf = buffer;
                        if (msg_len <= 0)
                        {
                            NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                            send_msg = false;
                        }
                    } else {
                        send_msg = false;
                    }
                // }
                        
                if (send_msg) {
                    NFAPI_TRACE(NFAPI_TRACE_INFO , "(Proxy UE) Prior to sending dl_tti_req to OAI UE. Frame: %d,"
                            " Slot: %d, Number of PDUs: %u",
                            dl_sfn, dl_slot, dl_numPDU);
                    assert(ue_tx_socket[ue_idx] > 2);
                    if (sendto(ue_tx_socket[ue_idx], msg_buf, msg_len, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST to OAI UE failed");
                    }
                    else
                    {
                        NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST forwarded from Proxy to UE");
                    }
                }
                break;
            }

            case NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST:
            {
                nfapi_nr_tx_data_request_t *tx_data_req = (nfapi_nr_tx_data_request_t*)(msg);

                // if (ue_info->rnti != 0xFFFF) {
                    // const nfapi_nr_tx_data_request_t *tx_data_req = (nfapi_nr_tx_data_request_t*)(msg);
                    int *valid_dci_ptr = ue_info->valid_dci_inds[tx_data_req->Slot];
                    nfapi_nr_tx_data_request_t tx_data_req_ue_spec;
                    copy_tx_data_request_for_rnti(tx_data_req, &tx_data_req_ue_spec, valid_dci_ptr);
                    if (tx_data_req_ue_spec.Number_of_PDUs != 0) { // PDUs for this RNTI
                        msg_len = nfapi_nr_p7_message_pack(&tx_data_req_ue_spec, buffer, msg_len, nullptr);
                        msg_buf = buffer;
                        if (msg_len <= 0)
                        {
                            NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                            send_msg = false;
                        }
                    } else {
                        send_msg = false;
                    }                 
                // }
        
                if (send_msg) {
                    if (sendto(ue_tx_socket[ue_idx], msg_buf, msg_len, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST to OAI UE failed");
                    }
                    else
                    {
                        NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST forwarded from Proxy to UE");
                    }
                }
                break;
            }
                
            case NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST:
            {
                nfapi_nr_ul_tti_request_t *ul_tti_req = (nfapi_nr_ul_tti_request_t*)(msg);
                nfapi_nr_ul_tti_request_t ul_tti_req_ue_spec;
                copy_ul_tti_request_for_rnti(ul_tti_req, &ul_tti_req_ue_spec, ue_info->rnti);
                if (ul_tti_req_ue_spec.n_pdus != 0) { // PDUs for this RNTI
                    msg_len = nfapi_nr_p7_message_pack(&ul_tti_req_ue_spec, buffer, msg_len, nullptr);
                    msg_buf = buffer;
                    if (msg_len <= 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                        send_msg = false;
                    }
                } else {
                    send_msg = false;
                }     

                if (send_msg) {
                    // assert(ue_tx_socket[ue_idx] > 2);
                    if (sendto(ue_tx_socket[ue_idx], msg_buf, msg_len, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST to OAI UE failed");
                    }
                    else
                    {
                        NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST forwarded from Proxy to UE");
                    }
                }
                break;
            }

            case NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST:
            {
                nfapi_nr_ul_dci_request_t *ul_dci_req = (nfapi_nr_ul_dci_request_t*)(msg);
                nfapi_nr_ul_dci_request_t ul_dci_req_ue_spec;
                copy_ul_dci_request_for_rnti(ul_dci_req, &ul_dci_req_ue_spec, ue_info->rnti);
                if (ul_dci_req_ue_spec.numPdus != 0) { // PDUs for this RNTI
                    msg_len = nfapi_nr_p7_message_pack(&ul_dci_req_ue_spec, buffer, msg_len, nullptr);
                    msg_buf = buffer;
                    if (msg_len <= 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "nfapi_nr_p7_message_pack failed");
                        send_msg = false;
                    }
                } else {
                    send_msg = false;
                }     

                if (send_msg) {
                    // assert(ue_tx_socket[ue_idx] > 2);
                    if (sendto(ue_tx_socket[ue_idx], msg_buf, msg_len, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
                    {
                        NFAPI_TRACE(NFAPI_TRACE_ERROR, "Send NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST to OAI UE failed");
                    }
                    else
                    {
                        NFAPI_TRACE(NFAPI_TRACE_INFO , "NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST forwarded from Proxy to UE");
                    }
                }
                break;
            }

            default:
                NFAPI_TRACE(NFAPI_TRACE_INFO , "Unhandled message at Proxy message_id: %u", msg_id);
                break;
        }

        msg_wrapper->ref_count--;
    }

}

void Multi_UE_NR_Proxy::pack_and_send_downlink_sfn_slot_msg(uint16_t sfn_slot)
{
    // lock_guard_t lock(mutex);

    for(int ue_idx = 0; ue_idx < num_ues; ue_idx++)
    {
        // inet_aton(oai_ue_ipaddr[ue_idx].c_str(), &address_tx_[ue_idx].sin_addr);
        // address_tx_[ue_idx].sin_port = htons(3612 + ue_idx * port_delta);
        assert(ue_tx_socket[ue_idx] > 2);
        if (sendto(ue_tx_socket[ue_idx], &sfn_slot, sizeof(sfn_slot), 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
        {
            int sfn = NFAPI_SFNSLOTDEC2SFN(MU, sfn_slot);
            int slot = NFAPI_SFNSLOTDEC2SLOT(MU, sfn_slot);
            NFAPI_TRACE(NFAPI_TRACE_DEBUG, "Send sfn_slot to OAI UE FAIL Frame: %d,Slot: %d", sfn, slot);
        }
    }
}

void Multi_UE_NR_Proxy::pack_and_send_downlink_sfn_slot_ch_info_msg(nr_phy_channel_params_t *ch_info, int ue_idx)
{
    // lock_guard_t lock(mutex);

    int encoded_size = sizeof(nr_phy_channel_params_t);
    uint8_t buffer[1024];
    memcpy(buffer, ch_info, encoded_size);

    // for(int ue_idx = 0; ue_idx < num_ues_; ue_idx++)
    // {
    // inet_aton(oai_ue_ipaddr[ue_idx].c_str(), &address_tx_[ue_idx].sin_addr);
    // address_tx_[ue_idx].sin_port = htons(3612 + ue_idx * port_delta);
    assert(ue_tx_socket[ue_idx] > 2);
    // if (sendto(ue_tx_socket[ue_idx], &sfn_slot, sizeof(sfn_slot), 0, (const struct sockaddr *) &address_tx_, sizeof(address_tx_)) < 0)
    if (sendto(ue_tx_socket[ue_idx], buffer, encoded_size, 0, (const struct sockaddr *) &address_tx_[ue_idx], sizeof(address_tx_[ue_idx])) < 0)
    {
        int sfn = NFAPI_SFNSLOTDEC2SFN(MU, ch_info->sfn_slot);
        int slot = NFAPI_SFNSLOTDEC2SLOT(MU, ch_info->sfn_slot);
        NFAPI_TRACE(NFAPI_TRACE_DEBUG, "Send sfn_slot to OAI UE FAIL Frame: %d,Slot: %d", sfn, slot);
    }
    // }
}

void enqueue_downlink_nfapi_msg(void *msg_buf, int msg_len, uint16_t msg_id) {
    instance->enqueue_downlink_nfapi_msg(msg_buf, msg_len, msg_id);
}


void transfer_downstream_nfapi_msg_to_nr_proxy(void *msg, uint16_t msg_id)
{
    instance->oai_gnb_downlink_nfapi_task(msg, msg_id);
}

void send_sfn_slot_to_ues(uint16_t sfn_slot)
{
    instance->pack_and_send_downlink_sfn_slot_msg(sfn_slot);
}

void send_sfn_slot_ch_info_to_ues(nr_phy_channel_params_t *ch_info, int ue_idx)
{
    instance->pack_and_send_downlink_sfn_slot_ch_info_msg(ch_info, ue_idx);
}

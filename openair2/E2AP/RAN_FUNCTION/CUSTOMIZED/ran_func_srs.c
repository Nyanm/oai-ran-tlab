/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "ran_func_srs.h"
#include "ran_func_srs_extern.h"
#include "ran_func_srs_subs.h"
#include <stdio.h>
#include <assert.h>
#include "openair2/E2AP/flexric/src/util/time_now_us.h"

#include "../../flexric/src/agent/e2_agent_api.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// for the packing functions
#include "nfapi/open-nFAPI/common/public_inc/nfapi.h"
#include "nfapi/open-nFAPI/fapi/inc/nr_fapi_p7.h"
#include "nfapi/open-nFAPI/fapi/inc/nr_fapi.h"
#include "nfapi/open-nFAPI/fapi/inc/nr_fapi_p7_utils.h"

#include "openair2/F1AP/f1ap_ids.h"
static seq_arr_t srs_subs_data = {0};

bool read_srs_sm(void* data)
{
  assert(data != NULL);
  assert(0!=0 && "Not implemented");
  return true;
}


void read_srs_setup_sm(void* data)
{
  assert(data != NULL);
  assert(0!=0 && "Not implemented");
}

sm_ag_if_ans_t write_ctrl_srs_sm(void const* src)
{
  assert(src != NULL);
  assert(0 !=0 && "Not supported");
}


static srs_ind_hdr_t fill_srs_ind_hdr(void)
{
  srs_ind_hdr_t hdr = {0};
  hdr.ev_trigger_cond_id = 2;
  return hdr;
}

static srs_ind_msg_t fill_srs_ind_msg(nfapi_nr_srs_indication_t *nfapi_srs_ind)
{
  srs_ind_msg_t msg = {0};
  msg.len = 1; // only for now
  msg.tstamp = time_now_us();
  
  if(msg.len > 0 ){  
    msg.indication_stats = calloc(msg.len, sizeof(srs_indication_stats_impl_t));
    assert(msg.indication_stats != NULL && "Memory exhausted");
  }

  for(uint32_t i = 0; i < msg.len; ++i){
    srs_indication_stats_impl_t* indication_stats = &msg.indication_stats[i];
  
    f1_ue_data_t ue_data = du_get_f1_ue_data(nfapi_srs_ind->pdu_list[0].rnti); // Unique CU-UE ID
    indication_stats->ue_id= ue_data.secondary_ue;
    printf("CU-UE ID: %u\n", indication_stats->ue_id);
    size_t ba_len = get_srs_indication_size(nfapi_srs_ind);
    byte_array_t ba = {.len = ba_len};
    ba.buf = malloc(ba.len);
    uint8_t *pPackedBuf = ba.buf;
    uint8_t *pWritePackedMessage    = pPackedBuf;
    uint8_t *pPackMessageEnd =  pPackedBuf + ba.len;

    printf("[RIC DEBUG INFO] pointer = %p\n",(void*)nfapi_srs_ind);
    printf("[RIC DEBUG INFO] ba initialized len: %zu bytes\n", ba.len);
    printf("[RIC DEBUG INFO] Sending SFN: %u\n", nfapi_srs_ind->sfn);
    printf("[RIC DEBUG INFO] Sending Slot: %u\n", nfapi_srs_ind->slot);
    printf("[RIC DEBUG INFO] Sending num srs: %d\n", nfapi_srs_ind->number_of_pdus);

    const uint8_t result = pack_nr_srs_indication(nfapi_srs_ind, &pWritePackedMessage, pPackMessageEnd, 0);
    assert(result != 0 && "Error in packing SRS Indication message"); 
    size_t packedBufLen = pWritePackedMessage - pPackedBuf;// this should be eq to the ba.len

    ba.len = packedBufLen;
    printf("[RIC DEBUG INFO] ba updated len: %zu bytes\n", ba.len);
    indication_stats->srs_indication_ba = copy_byte_array(ba);

    // Clean up
    free_byte_array(ba);
  }

  return msg;
}

static void send_ric_indication(const uint32_t ric_req_id, srs_ind_data_t* srs_ind_data)
{
  async_event_agent_api(ric_req_id, srs_ind_data);
  printf("Event for RIC Req ID %u generated\n", ric_req_id);
}


static void free_aperiodic_subscription(uint32_t ric_req_id)
{
  remove_srs_subs_data(&srs_subs_data, ric_req_id);
}


static srs_ind_data_t* fill_fapi_srs_indication(nfapi_nr_srs_indication_t *nfapi_srs_ind)
{
  srs_ind_data_t* srs_ind = calloc(1,sizeof(srs_ind_data_t));
  assert(srs_ind != NULL && "Memory exhausted");

  srs_ind->hdr = fill_srs_ind_hdr();
  srs_ind->msg = fill_srs_ind_msg(nfapi_srs_ind);

  return srs_ind;
}

void signal_nfapi_srs_indication(nfapi_nr_srs_indication_t *nfapi_srs_ind)
{
  // Check number of subscriptions:
  const size_t num_subs = seq_arr_size(&srs_subs_data);
  for (size_t sub_idx = 0; sub_idx < num_subs; sub_idx++) {
    const ran_param_data_t data = *(const ran_param_data_t *)seq_arr_at(&srs_subs_data, sub_idx);
    srs_ind_data_t* srs_ind_data = fill_fapi_srs_indication(nfapi_srs_ind);
    //Send RIC indication
    send_ric_indication(data.ric_req_id, srs_ind_data);
  }
}





sm_ag_if_ans_t write_subs_srs_sm(void const* src)
{
  assert(src != NULL);
  wr_srs_sub_data_t* wr_srs = (wr_srs_sub_data_t*)src;
  assert(wr_srs->srs.ad != NULL && "Cannot be NULL");

  sm_ag_if_ans_t ans = {0};

  const uint32_t ric_req_id = wr_srs->ric_req_id;

  struct ran_param_data data = { .ric_req_id = ric_req_id};
  init_srs_subs_data(&srs_subs_data);
  insert_srs_subs_data(&srs_subs_data,&data);

  ans.type = SUBS_OUTCOME_SM_AG_IF_ANS_V0;
  ans.subs_out.type = APERIODIC_SUBSCRIPTION_FLRC;
  ans.subs_out.aper.free_aper_subs = free_aperiodic_subscription;


  return ans;
}

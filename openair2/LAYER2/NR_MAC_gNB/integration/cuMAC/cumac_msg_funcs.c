#include "cumac_msg_funcs.h"

#include <nv_ipc.h>
#include <stdalign.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <asm-generic/errno-base.h>
static size_t get_padding(size_t offset, size_t alignment)
{
  return (alignment - (offset % alignment)) % alignment;
}
int l2_build_start_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args) {
  if (type != CUMAC_START_REQUEST)
    return -EINVAL;
  cumac_start_req_t* req = nvipc_buf->msg_buf;
  nvipc_buf->msg_id = CUMAC_START_REQUEST;
  nvipc_buf->cell_id = 0;
  nvipc_buf->msg_len = sizeof(cumac_start_req_t);
  nvipc_buf->data_len = 0;
  nvipc_buf->data_buf = NULL;
  nvipc_buf->data_pool = NV_IPC_MEMPOOL_CPU_MSG;
  req->header.message_count = 1;
  req->header.handle_id = 0;
  req->header.type_id = CUMAC_START_REQUEST;
  req->header.body_len =  sizeof(req->start_param);
  req->start_param = 0;
  return 0;
}


int l2_build_config_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args) {
  if (type != CUMAC_CONFIG_REQUEST)
    return -EINVAL;
  cumac_config_req_payload_t *payload = (cumac_config_req_payload_t*)args;
  cumac_config_req_t* req = (cumac_config_req_t*)nvipc_buf->msg_buf;
  nvipc_buf->msg_id = CUMAC_CONFIG_REQUEST;
  nvipc_buf->cell_id = 0;
  nvipc_buf->msg_len = sizeof(cumac_config_req_t);
  nvipc_buf->data_len = 0;
  nvipc_buf->data_buf = NULL;
  nvipc_buf->data_pool = NV_IPC_MEMPOOL_CPU_MSG;
  req->header.message_count = 1;
  req->header.handle_id = 0;
  req->header.type_id = CUMAC_CONFIG_REQUEST;
  req->header.body_len =  sizeof(cumac_config_req_payload_t);
  memcpy(req->body, payload, sizeof(cumac_config_req_payload_t));
  return 0;
}

int l2_build_sch_tti_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args) {
  if (type != CUMAC_SCH_TTI_REQUEST)
    return -EINVAL;
  cumac_sch_tti_req_args_t *config_args =  args;
  uint16_t frame     = config_args->frame;
  uint16_t slot      = config_args->slot;
  uint16_t nActiveUE = config_args->payload.nActiveUe;
  uint16_t nPRBGRP   = config_args->payload.nPrbGrp;
  uint8_t  nUEAnt    = config_args->payload.nUeAnt;
  cumac_tti_req_bufs_t *req_buf = config_args->buffers;

  cumac_sch_tti_req_t* req = (cumac_sch_tti_req_t*)nvipc_buf->msg_buf;
  uint8_t* data_buf = (uint8_t*)nvipc_buf->data_buf;
  nvipc_buf->msg_id = CUMAC_SCH_TTI_REQUEST;
  nvipc_buf->cell_id = config_args->payload.cellID;
  nvipc_buf->msg_len = sizeof(cumac_sch_tti_req_t);
  memcpy(&req->payload, &config_args->payload, sizeof(cumac_tti_req_payload_t));



  req->sfn =frame;
  req->slot=slot;

  req->header.message_count = 1;
  req->header.handle_id = 0;
  req->header.type_id = CUMAC_SCH_TTI_REQUEST;
  req->header.body_len =  sizeof(cumac_tti_req_payload_t);

  // Set all bytes to 0xFF since the unused offsets are to be set to 0xFFFFFFFF
  memset(&req->payload.offsets, 0xFF, sizeof req->payload.offsets);

  // Copy data which need to be sent to GPU to the nvipc_buf->data_buf
  uint32_t offset = 0;
  uint32_t data_size = 0;

  // Set the offset and copy PFM_UE_INFO data to the data buffer
  offset += get_padding(offset, alignof(uint16_t));
  req->payload.offsets.CRNTI = offset;
  data_size = sizeof(*req_buf->CRNTI) * nActiveUE;
  printf("CRNTI : ");
  for (int i = 0; i < nActiveUE; ++i)
    printf("%u ", req_buf->CRNTI[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->CRNTI, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(uint8_t));
  req->payload.offsets.prgMsk = offset;
  data_size = sizeof(*req_buf->prgMsk) * nPRBGRP;
  printf("PRG MASK : ");
  for (int i = 0; i < nPRBGRP; ++i)
    printf("%u ", req_buf->prgMsk[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->prgMsk, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.wbSinr = offset;
  data_size = sizeof(*req_buf->wbSinr) * nActiveUE * nUEAnt;
  printf("wbSinr : ");
  for (int i = 0; i < nActiveUE * nUEAnt; ++i)
    printf("%f ", req_buf->wbSinr[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->wbSinr, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.avgRatesActUe = offset;
  data_size = sizeof(*req_buf->avgRatesActUe) * nActiveUE;
  printf("avgRatesActUe : ");
  for (int i = 0; i < nActiveUE; ++i)
    printf("%f ", req_buf->avgRatesActUe[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->avgRatesActUe, data_size);
  offset += data_size;


  nvipc_buf->data_len = offset;
  return 0;
}

int l2_build_tti_end(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args) {
  if (type != CUMAC_TTI_END)
    return -EINVAL;
  cumac_sch_tti_end_args_t *tti_end_args = args;
  cumac_tti_end_t* req = nvipc_buf->msg_buf;
  nvipc_buf->msg_id = CUMAC_TTI_END;
  nvipc_buf->cell_id = 0;
  nvipc_buf->msg_len = sizeof(cumac_tti_end_t);
  nvipc_buf->data_len = 0;
  nvipc_buf->data_buf = NULL;
  nvipc_buf->data_pool = NV_IPC_MEMPOOL_CPU_MSG;
  req->header.message_count = 1;
  req->header.handle_id = 0;
  req->header.type_id = CUMAC_TTI_END;
  req->header.body_len =  sizeof(req->end_param);
  req->end_param = 0;
  req->sfn = tti_end_args->frame;
  req->slot = tti_end_args->slot;
  return 0;
}

void cumac_handle_sch_tti_response(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf) {
  if (type != CUMAC_SCH_TTI_RESPONSE)
    return;



}

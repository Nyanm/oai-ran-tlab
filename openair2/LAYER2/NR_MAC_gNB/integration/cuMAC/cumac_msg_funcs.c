#include "cumac_msg_funcs.h"

#include "cumac_nvipc.h"

#include <nv_ipc.h>
#include <nvlog.h>
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
  uint8_t  nBSAnt    = config_args->payload.nBsAnt;
  cumac_tti_req_bufs_t *req_buf = config_args->buffers;

  slot_data_entry_t *slot_entry = &slot_data[frame][slot];

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
  uint32_t num_data = 0;

// Fill Common buffers
  offset += get_padding(offset, alignof(uint16_t));
  req->payload.offsets.CRNTI = offset;
  num_data = nActiveUE;
  data_size = sizeof(*req_buf->CRNTI) * num_data;
  printf("CRNTI : ");
  for (int i = 0; i < num_data; ++i)
    printf("%u ", req_buf->CRNTI[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->CRNTI, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(uint8_t));
  req->payload.offsets.prgMsk = offset;
  num_data = nPRBGRP;
  data_size = sizeof(*req_buf->prgMsk) * num_data;
  printf("PRG MASK : ");
  for (int i = 0; i < num_data; ++i)
    printf("%u ", req_buf->prgMsk[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->prgMsk, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.wbSinr = offset;
  num_data =  nActiveUE * nUEAnt;
  data_size = sizeof(*req_buf->wbSinr) * num_data;
  printf("wbSinr : ");
  for (int i = 0; i < num_data; ++i)
    printf("%f ", req_buf->wbSinr[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->wbSinr, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.avgRatesActUe = offset;
  num_data =  nActiveUE;
  data_size = sizeof(*req_buf->avgRatesActUe) * num_data;
  printf("avgRatesActUe : ");
  for (int i = 0; i < num_data; ++i)
    printf("%f ", req_buf->avgRatesActUe[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->avgRatesActUe, data_size);
  offset += data_size;

// PRB Allocation specific buffers
  if (req->payload.taskBitMask & TASK_BIT(CUMAC_TASK_PRB_ALLOCATION)) {
    offset += get_padding(offset, alignof(float));
    req->payload.offsets.postEqSinr = offset;
    num_data =  nActiveUE * nPRBGRP * nUEAnt;
    data_size = sizeof(*req_buf->postEqSinr) * num_data;
    printf("postEqSinr : ");
    for (int i = 0; i < num_data; ++i)
      printf("%f ", req_buf->postEqSinr[i]);
    printf("\n");
    memcpy(data_buf + offset, req_buf->postEqSinr, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(float));
    req->payload.offsets.sinVal = offset;
    num_data =  slot_entry->nMaxSchUePerCell * nPRBGRP * nUEAnt;
    data_size = sizeof(*req_buf->sinVal) * num_data;
    printf("sinVal : ");
    for (int i = 0; i < num_data; ++i)
      printf("%f ", req_buf->sinVal[i]);
    printf("\n");
    memcpy(data_buf + offset, req_buf->sinVal, data_size);
    offset += data_size;

    uint32_t prdLen, detLen, hLen;
    if(req->payload.ULDLSch == 1)
    { // DL
      prdLen = slot_entry->nMaxSchUePerCell * nPRBGRP * nBSAnt*nBSAnt;
      detLen = slot_entry->nMaxSchUePerCell * nPRBGRP * nUEAnt * nUEAnt;
    }
    else
    { // UL
      prdLen = slot_entry->nMaxSchUePerCell * nPRBGRP *  nUEAnt * nUEAnt;
      detLen = slot_entry->nMaxSchUePerCell * nPRBGRP * nBSAnt*nBSAnt;
    }
    hLen = nPRBGRP * slot_entry->nMaxSchUePerCell * /*nMaxCell*/ 1 * nBSAnt * nUEAnt;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.detMat = offset;
    data_size = sizeof(*req_buf->detMat) * detLen;
    printf("detMat : ");
    for (int i = 0; i < detLen; ++i)
      printf("x = %f , y= %f ", req_buf->detMat[i].x, req_buf->detMat[i].y);
    printf("\n");
    memcpy(data_buf + offset, req_buf->detMat, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.estH_fr = offset;
    data_size = sizeof(*req_buf->estH_fr) * hLen;
    printf("estH_fr : ");
    for (int i = 0; i < hLen; ++i)
      printf("x = %f , y= %f ", req_buf->estH_fr[i].x, req_buf->estH_fr[i].y);
    printf("\n");
    memcpy(data_buf + offset, req_buf->estH_fr, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.prdMat = offset;
    data_size = sizeof(*req_buf->prdMat) * prdLen;
    printf("prdMat : ");
    for (int i = 0; i < prdLen; ++i)
      printf("x = %f , y= %f ", req_buf->prdMat[i].x, req_buf->prdMat[i].y);
    printf("\n");
    memcpy(data_buf + offset, req_buf->prdMat, data_size);
    offset += data_size;
  }

  if (req->payload.taskBitMask & TASK_BIT(CUMAC_TASK_MCS_SELECTION)) {
    offset += get_padding(offset, alignof(int8_t));
    req->payload.offsets.tbErrLastActUe = offset;
    data_size = sizeof(*req_buf->tbErrLastActUe) * nActiveUE;
    printf("tbErrLastActUe : ");
    for (int i = 0; i < nActiveUE; ++i)
      printf("%d ", req_buf->tbErrLastActUe[i]);
    printf("\n");
    memcpy(data_buf + offset, req_buf->tbErrLastActUe, data_size);
    offset += data_size;
  }

  offset += get_padding(offset, alignof(int8_t));
  req->payload.offsets.newDataActUe = offset;
  num_data =  nActiveUE;
  data_size = sizeof(*req_buf->newDataActUe) * num_data;
  printf("newDataActUe : ");
  for (int i = 0; i < num_data; ++i)
    printf("%d ", req_buf->newDataActUe[i]);
  printf("\n");
  memcpy(data_buf + offset, req_buf->newDataActUe, data_size);
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

void cumac_handle_sch_tti_response(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, slot_data_entry_t *slot_data_entry)
{
  if (type != CUMAC_SCH_TTI_RESPONSE)
    return;
  cumac_sch_tti_resp_t *resp = nvipc_buf->msg_buf;
  uint8_t *buf_home = (nvipc_buf->data_buf);
  uint16_t *setSchdUePerCellTTI =
      calloc(slot_data_entry->nMaxSchUePerCell, sizeof(uint16_t)); //!< Set of IDs of the selected UEs for the cell
  int16_t *allocSol = calloc(slot_data_entry->allocSolSize, sizeof(int16_t));
  ; //!< PRB group allocation solution for all active UEs in the cell
  int16_t *mcsSelSol = calloc(slot_data_entry->nActiveUe, sizeof(int16_t));
  ; //!< MCS selection solution for all active UEs in the cell
  uint8_t *layerSelSol = calloc(slot_data_entry->nActiveUe, sizeof(uint8_t));
  ; //!< Layer selection solution for all active UEs in the cell

  printf("Received SCH_TTI.response with data\n");
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  printf("message 0x%02x\n", type);
  printf("From TTI_REQ/TTI_END to RESPONSE %ld , %ld\n",
         nvlog_timespec_interval(&slot_data_entry->tti_req_timestamp, &now),
         nvlog_timespec_interval(&slot_data_entry->tti_end_timestamp, &now));
  printf("Used values in originating SCH_TTI.request:\n");
  printf("\tnMaxSchUePerCell = %d\n", slot_data_entry->nMaxSchUePerCell);
  printf("\tallocSolSize = %d\n", slot_data_entry->allocSolSize);
  printf("\tnActiveUe = %d\n", slot_data_entry->nActiveUe);

  printf("Offsets : \n\tsetSchdUePerCellTTI 0x%04x\n", resp->offsets.setSchdUePerCellTTI);
  printf("\tallocSol 0x%04x\n", resp->offsets.allocSol);
  printf("\tmcsSelSol 0x%04x\n", resp->offsets.mcsSelSol);
  printf("\tlayerSelSol 0x%04x\n", resp->offsets.layerSelSol);

  if (resp->offsets.setSchdUePerCellTTI != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.setSchdUePerCellTTI);
    memcpy(setSchdUePerCellTTI, src, slot_data_entry->nMaxSchUePerCell * sizeof(uint16_t));
    printf("setSchdUePerCellTTI :\n");
    for (int i = 0; i < slot_data_entry->nMaxSchUePerCell; i++) {
      printf("\tIDX %d = %d \n", i, setSchdUePerCellTTI[i]);
    }
  }
  if (resp->offsets.allocSol != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.allocSol);
    memcpy(allocSol, src, slot_data_entry->allocSolSize * sizeof(int16_t));
    printf("allocSol :\n");
    for (int i = 0; i < slot_data_entry->allocSolSize; i++) {
      printf("\tIDX %d = 0x%02x \n", i, allocSol[i]);
    }
  }
  if (resp->offsets.mcsSelSol != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.mcsSelSol);
    memcpy(mcsSelSol, src, slot_data_entry->nActiveUe * sizeof(int16_t));
    printf("mcsSelSol :\n");
    for (int i = 0; i < slot_data_entry->nActiveUe; i++) {
      printf("\tIDX %d = 0x%02x \n", i, mcsSelSol[i]);
    }
  }
  if (resp->offsets.layerSelSol != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.layerSelSol);
    memcpy(layerSelSol, src, slot_data_entry->nActiveUe);
    printf("layerSelSol :\n");
    for (int i = 0; i < slot_data_entry->nActiveUe; i++) {
      printf("\tIDX %d = 0x%02x \n", i, layerSelSol[i]);
    }
  }

  free(setSchdUePerCellTTI);
  free(allocSol);
  free(mcsSelSol);
  free(layerSelSol);
}

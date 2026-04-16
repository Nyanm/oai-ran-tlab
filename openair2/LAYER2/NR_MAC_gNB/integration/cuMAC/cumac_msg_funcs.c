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
  // ensure buffers are free
  if (slot_entry->setSchdUePerCellTTI != NULL) {
    free(slot_entry->setSchdUePerCellTTI);
  }
  if (slot_entry->allocSol != NULL) {
    free(slot_entry->allocSol);
  }
  if (slot_entry->mcsSelSol != NULL) {
    free(slot_entry->mcsSelSol);
  }
  if (slot_entry->layerSelSol != NULL) {
    free(slot_entry->layerSelSol);
  }
  if (slot_entry->c_rnti != NULL) {
    free(slot_entry->c_rnti);
  }
  slot_entry->c_rnti = calloc(nActiveUE, sizeof(*req_buf->CRNTI));
  memcpy(slot_entry->c_rnti, req_buf->CRNTI, sizeof(*req_buf->CRNTI) * nActiveUE);
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
  LOG_D(NR_MAC,"CRNTI : ");
  for (int i = 0; i < num_data; ++i)
    LOG_D(NR_MAC,"%u ", req_buf->CRNTI[i]);
  LOG_D(NR_MAC,"\n");
  memcpy(data_buf + offset, req_buf->CRNTI, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(uint8_t));
  req->payload.offsets.prgMsk = offset;
  num_data = nPRBGRP;
  data_size = sizeof(*req_buf->prgMsk) * num_data;
  LOG_D(NR_MAC,"PRG MASK : ");
  for (int i = 0; i < num_data; ++i)
    LOG_D(NR_MAC,"%u ", req_buf->prgMsk[i]);
  LOG_D(NR_MAC,"\n");
  memcpy(data_buf + offset, req_buf->prgMsk, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.wbSinr = offset;
  num_data =  nActiveUE * nUEAnt;
  data_size = sizeof(*req_buf->wbSinr) * num_data;
  LOG_D(NR_MAC,"wbSinr : ");
  for (int i = 0; i < num_data; ++i)
    LOG_D(NR_MAC,"%f ", req_buf->wbSinr[i]);
  LOG_D(NR_MAC,"\n");
  memcpy(data_buf + offset, req_buf->wbSinr, data_size);
  offset += data_size;

  offset += get_padding(offset, alignof(float));
  req->payload.offsets.avgRatesActUe = offset;
  num_data =  nActiveUE;
  data_size = sizeof(*req_buf->avgRatesActUe) * num_data;
  LOG_D(NR_MAC,"avgRatesActUe : ");
  for (int i = 0; i < num_data; ++i)
    LOG_D(NR_MAC,"%f ", req_buf->avgRatesActUe[i]);
  LOG_D(NR_MAC,"\n");
  memcpy(data_buf + offset, req_buf->avgRatesActUe, data_size);
  offset += data_size;

// PRB Allocation specific buffers
  if (req->payload.taskBitMask & TASK_BIT(CUMAC_TASK_PRB_ALLOCATION)) {
    offset += get_padding(offset, alignof(float));
    req->payload.offsets.postEqSinr = offset;
    num_data =  nActiveUE * nPRBGRP * nUEAnt;
    data_size = sizeof(*req_buf->postEqSinr) * num_data;
    LOG_D(NR_MAC,"postEqSinr : ");
    for (int i = 0; i < num_data; ++i)
      LOG_D(NR_MAC,"%f ", req_buf->postEqSinr[i]);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->postEqSinr, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(float));
    req->payload.offsets.sinVal = offset;
    num_data =  slot_entry->nMaxSchUePerCell * nPRBGRP * cumac_nPrbPerPrg() * nUEAnt;
    data_size = sizeof(*req_buf->sinVal) * num_data;
    LOG_D(NR_MAC,"sinVal : ");
    for (int i = 0; i < num_data; ++i)
      LOG_D(NR_MAC,"%f ", req_buf->sinVal[i]);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->sinVal, data_size);
    offset += data_size;

    uint32_t prdLen, detLen;
    if(req->payload.ULDLSch == 1)
    { // DL
      prdLen = slot_entry->nMaxSchUePerCell * nPRBGRP * cumac_nPrbPerPrg() * nBSAnt*nBSAnt;
      detLen = slot_entry->nMaxSchUePerCell * nPRBGRP * cumac_nPrbPerPrg() * nUEAnt * nUEAnt;
    }
    else
    { // UL
      prdLen = slot_entry->nMaxSchUePerCell * nPRBGRP * cumac_nPrbPerPrg() * nUEAnt * nUEAnt;
      detLen = slot_entry->nMaxSchUePerCell * nPRBGRP * cumac_nPrbPerPrg() * nBSAnt*nBSAnt;
    }
    const uint32_t hLen = nPRBGRP * cumac_nPrbPerPrg() * slot_entry->nMaxSchUePerCell * /*nMaxCell*/ 1 * nBSAnt * nUEAnt;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.detMat = offset;
    data_size = sizeof(*req_buf->detMat) * detLen;
    LOG_D(NR_MAC,"detMat : ");
    for (int i = 0; i < detLen; ++i)
      LOG_D(NR_MAC,"x = %f , y= %f ", req_buf->detMat[i].x, req_buf->detMat[i].y);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->detMat, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.estH_fr = offset;
    data_size = sizeof(*req_buf->estH_fr) * hLen;
    LOG_D(NR_MAC,"estH_fr : ");
    for (int i = 0; i < hLen; ++i)
      LOG_D(NR_MAC,"x = %f , y= %f ", req_buf->estH_fr[i].x, req_buf->estH_fr[i].y);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->estH_fr, data_size);
    offset += data_size;

    offset += get_padding(offset, alignof(cuComplex));
    req->payload.offsets.prdMat = offset;
    data_size = sizeof(*req_buf->prdMat) * prdLen;
    LOG_D(NR_MAC,"prdMat : ");
    for (int i = 0; i < prdLen; ++i)
      LOG_D(NR_MAC,"x = %f , y= %f ", req_buf->prdMat[i].x, req_buf->prdMat[i].y);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->prdMat, data_size);
    offset += data_size;

    if (req_buf->allocSolLastTxActUe != NULL) {
      offset += get_padding(offset, alignof(int16_t));
      req->payload.offsets.allocSolLastTxActUe = offset;
      data_size = sizeof(*req_buf->allocSolLastTxActUe) * nActiveUE * 2;
      memcpy(data_buf + offset, req_buf->allocSolLastTxActUe, data_size);
      offset += data_size;
    }
  }

  if (req->payload.taskBitMask & TASK_BIT(CUMAC_TASK_MCS_SELECTION)) {
    offset += get_padding(offset, alignof(int8_t));
    req->payload.offsets.tbErrLastActUe = offset;
    data_size = sizeof(*req_buf->tbErrLastActUe) * nActiveUE;
    LOG_D(NR_MAC,"tbErrLastActUe : ");
    for (int i = 0; i < nActiveUE; ++i)
      LOG_D(NR_MAC,"%d ", req_buf->tbErrLastActUe[i]);
    LOG_D(NR_MAC,"\n");
    memcpy(data_buf + offset, req_buf->tbErrLastActUe, data_size);
    offset += data_size;

    if (req_buf->mcsSelSolLastTxActUe != NULL) {
      offset += get_padding(offset, alignof(int16_t));
      req->payload.offsets.mcsSelSolLastTxActUe = offset;
      data_size = sizeof(*req_buf->mcsSelSolLastTxActUe) * nActiveUE;
      memcpy(data_buf + offset, req_buf->mcsSelSolLastTxActUe, data_size);
      offset += data_size;
    }
  }

  if (req->payload.taskBitMask & TASK_BIT(CUMAC_TASK_LAYER_SELECTION)) {
    if (req_buf->layerSelSolLastTxActUe != NULL) {
      offset += get_padding(offset, alignof(int8_t));
      req->payload.offsets.layerSelSolLastTxActUe = offset;
      data_size = sizeof(*req_buf->layerSelSolLastTxActUe) * nActiveUE;
      memcpy(data_buf + offset, req_buf->layerSelSolLastTxActUe, data_size);
      offset += data_size;
    }
  }

  offset += get_padding(offset, alignof(int8_t));
  req->payload.offsets.newDataActUe = offset;
  num_data =  nActiveUE;
  data_size = sizeof(*req_buf->newDataActUe) * num_data;
  LOG_D(NR_MAC,"newDataActUe : ");
  for (int i = 0; i < num_data; ++i)
    LOG_D(NR_MAC,"%d ", req_buf->newDataActUe[i]);
  LOG_D(NR_MAC,"\n");
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
  //!< Set of IDs of the selected UEs for the cell
  slot_data_entry->setSchdUePerCellTTI =
      calloc(slot_data_entry->nMaxSchUePerCell, sizeof(uint16_t));
  //!< PRB group allocation solution for all active UEs in the cell
  slot_data_entry->allocSol = calloc(slot_data_entry->allocSolSize, sizeof(int16_t));
  //!< MCS selection solution for all active UEs in the cell
  slot_data_entry->mcsSelSol = calloc(slot_data_entry->nActiveUe, sizeof(int16_t));
  //!< Layer selection solution for all active UEs in the cell
  slot_data_entry->layerSelSol = calloc(slot_data_entry->nActiveUe, sizeof(uint8_t));

  LOG_D(NR_MAC,"Received SCH_TTI.response with data\n");
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);
  LOG_D(NR_MAC,"message 0x%02x\n", type);
  LOG_D(NR_MAC,"From TTI_REQ/TTI_END to RESPONSE %ld , %ld\n",
         nvlog_timespec_interval(&slot_data_entry->tti_req_timestamp, &now),
         nvlog_timespec_interval(&slot_data_entry->tti_end_timestamp, &now));
  LOG_D(NR_MAC,"Used values in originating SCH_TTI.request:\n");
  LOG_D(NR_MAC,"\tnMaxSchUePerCell = %d\n", slot_data_entry->nMaxSchUePerCell);
  LOG_D(NR_MAC,"\tallocSolSize = %d\n", slot_data_entry->allocSolSize);
  LOG_D(NR_MAC,"\tnActiveUe = %d\n", slot_data_entry->nActiveUe);

  LOG_D(NR_MAC,"Offsets : \n\tsetSchdUePerCellTTI 0x%04x\n", resp->offsets.setSchdUePerCellTTI);
  LOG_D(NR_MAC,"\tallocSol 0x%04x\n", resp->offsets.allocSol);
  LOG_D(NR_MAC,"\tmcsSelSol 0x%04x\n", resp->offsets.mcsSelSol);
  LOG_D(NR_MAC,"\tlayerSelSol 0x%04x\n", resp->offsets.layerSelSol);

  if (resp->offsets.setSchdUePerCellTTI != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.setSchdUePerCellTTI);
    memcpy(slot_data_entry->setSchdUePerCellTTI, src, slot_data_entry->nMaxSchUePerCell * sizeof(uint16_t));
    LOG_D(NR_MAC,"setSchdUePerCellTTI :\n");
    for (int i = 0; i < slot_data_entry->nMaxSchUePerCell; i++) {
      LOG_D(NR_MAC,"\tIDX %d = %d \n", i, slot_data_entry->setSchdUePerCellTTI[i]);
    }
  }
  if (resp->offsets.allocSol != 0xFFFFFFFF) {
    const uint16_t *src = (uint16_t *)(buf_home + resp->offsets.allocSol);
    memcpy(slot_data_entry->allocSol, src, slot_data_entry->allocSolSize * sizeof(*slot_data_entry->allocSol));
    LOG_D(NR_MAC,"allocSol:\n");
    for (int i = 0; i < slot_data_entry->allocSolSize; i++) {
      LOG_D(NR_MAC,"\tIDX %d = 0x%02x \n", i, src[i]);

    }
  }
  if (resp->offsets.mcsSelSol != 0xFFFFFFFF) {
    const int16_t *src = (int16_t *)(buf_home + resp->offsets.mcsSelSol);
    memcpy(slot_data_entry->mcsSelSol, src, slot_data_entry->nActiveUe * sizeof(*slot_data_entry->mcsSelSol));
    LOG_D(NR_MAC,"mcsSelSol :\n");
    for (int i = 0; i < slot_data_entry->nActiveUe; i++) {
      LOG_D(NR_MAC,"\tIDX %d = 0x%02x \n", i, slot_data_entry->mcsSelSol[i]);
    }
  }
  if (resp->offsets.layerSelSol != 0xFFFFFFFF) {
    const uint8_t *src = (uint8_t *)(buf_home + resp->offsets.layerSelSol);
    memcpy(slot_data_entry->layerSelSol, src, slot_data_entry->nActiveUe);
    LOG_D(NR_MAC,"layerSelSol :\n");
    for (int i = 0; i < slot_data_entry->nActiveUe; i++) {
      LOG_D(NR_MAC,"\tIDX %d = 0x%02x \n", i, slot_data_entry->layerSelSol[i]);
    }
  }
}

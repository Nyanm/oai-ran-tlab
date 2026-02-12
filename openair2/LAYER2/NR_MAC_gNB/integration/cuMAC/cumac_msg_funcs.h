//
// Created by user on 10 Oct 2025.
//

#ifndef OPENAIRINTERFACE_CUMAC_MSG_FUNCS_H
#define OPENAIRINTERFACE_CUMAC_MSG_FUNCS_H
#include "cumac_msg.h"
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <nv_ipc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#define SCH_TTI_UL 0
#define SCH_TTI_DL 1

// Buffer offsets for SCH_TTI.req
typedef struct {
  uint16_t *CRNTI; // C-RNTIs of all active UEs in the cell
  uint16_t *srsCRNTI; // C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
  uint8_t *prgMsk; // Bit map for the availability of each PRG for allocation
  float *postEqSinr; // Array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
  float *wbSinr; // Array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
  cuComplex
      *estH_fr; // For FP32. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
   __nv_bfloat162_raw
      *estH_fr_half; // For FP16. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
  cuComplex *prdMat; // Array of the precoder/beamforming weights for all active UEs in the cell
  cuComplex *detMat; // Array of the detector/beamforming weights for all active UEs in the cell
  float *sinVal; // Array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
  float *avgRatesActUe; // Array of the long-term average data rates of all active UEs in the cell
  uint16_t *prioWeightActUe; // For priority-based UE selection. Priority weights of all active UEs in the cell
  int8_t *tbErrLastActUe; // TB decoding error indicators of all active UEs in the cell
  int8_t *newDataActUe; // Indicators of initial transmission/retransmission for all active UEs in the cell
  int16_t *allocSolLastTxActUe; // The PRG allocation solution for the last transmissions of all active UEs in the cell
  int16_t *mcsSelSolLastTxActUe; // MCS selection solution for the last transmissions of all active UEs in the cell
  int8_t *layerSelSolLastTxActUe; // Layer selection solution for the last transmissions of all active UEs in the cell
} cumac_tti_req_bufs_t;

typedef struct {
  int32_t cellId;
  cumac_config_req_payload_t payload;
} cumac_config_req_args_t;

typedef struct {
  uint16_t frame    ;
  uint16_t slot     ;
  cumac_tti_req_payload_t payload;
  cumac_tti_req_bufs_t *buffers;
} cumac_sch_tti_req_args_t;

typedef struct {
  uint16_t frame    ;
  uint16_t slot     ;
} cumac_sch_tti_end_args_t;

#define TASK_BIT(t) (1u << (t))

typedef int (*build_cumac_msg_fn_v_t)(cumac_msg_t type,nv_ipc_msg_t *nvipc_buf,  void* args);
int l2_build_start_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args);
int l2_build_config_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args);
int l2_build_sch_tti_request(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args);
int l2_build_tti_end(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf, void* args);
void cumac_handle_sch_tti_response(cumac_msg_t type, nv_ipc_msg_t *nvipc_buf) ;
#endif // OPENAIRINTERFACE_CUMAC_MSG_FUNCS_H

/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * phy_messages_types.h
 */

#ifndef PHY_MESSAGES_TYPES_H_
#define PHY_MESSAGES_TYPES_H_

#include "PHY/defs_common.h"

//-------------------------------------------------------------------------------------------//
// Defines to access message fields.
#define PHY_CONFIGURATION_REQ(mSGpTR)       (mSGpTR)->ittiMsg.phy_configuration_req

#define PHY_DEACTIVATE_REQ(mSGpTR)          (mSGpTR)->ittiMsg.phy_deactivate_req

#define PHY_FIND_CELL_REQ(mSGpTR)           (mSGpTR)->ittiMsg.phy_find_cell_req
#define PHY_FIND_NEXT_CELL_REQ(mSGpTR)      (mSGpTR)->ittiMsg.phy_find_next_cell_req

#define PHY_FIND_CELL_IND(mSGpTR)           (mSGpTR)->ittiMsg.phy_find_cell_ind

#define PHY_MEAS_THRESHOLD_REQ(mSGpTR)      (mSGpTR)->ittiMsg.phy_meas_threshold_req
#define PHY_MEAS_THRESHOLD_CONF(mSGpTR)     (mSGpTR)->ittiMsg.phy_meas_threshold_conf
#define PHY_MEAS_REPORT_IND(mSGpTR)         (mSGpTR)->ittiMsg.phy_meas_report_ind
//-------------------------------------------------------------------------------------------//
#define MAX_REPORTED_CELL   10

/* Enhance absolute radio frequency channel number */
typedef uint16_t    Earfcn;

/* Physical cell identity, valid value are from 0 to 503 */
typedef int16_t     PhyCellId;

/* Reference signal received power, valid value are from 0 (rsrp < -140 dBm) to 97 (rsrp <= -44 dBm) */
typedef int8_t      Rsrp;

/* Reference signal received quality, valid value are from 0 (rsrq < -19.50 dB) to 34 (rsrq <= -3 dB) */
typedef int8_t      Rsrq;

typedef struct CellInfo_s {
  Earfcn      earfcn;
  PhyCellId   cell_id;
  Rsrp        rsrp;
  Rsrq        rsrq;
} CellInfo;

//-------------------------------------------------------------------------------------------//
// eNB: ENB_APP -> PHY messages
typedef struct PhyConfigurationReq_s {
  frame_type_t            frame_type[MAX_NUM_CCs];
  lte_prefix_type_t       prefix_type[MAX_NUM_CCs];
  uint32_t                downlink_frequency[MAX_NUM_CCs];
  int32_t                 uplink_frequency_offset[MAX_NUM_CCs];
  int32_t                 nb_antennas_tx[MAX_NUM_CCs];
  int32_t                 nb_antennas_rx[MAX_NUM_CCs];
  int32_t                 tx_gain[MAX_NUM_CCs];
  int32_t                 rx_gain[MAX_NUM_CCs];
} PhyConfigurationReq;

// UE: RRC -> PHY messages
typedef struct PhyDeactivateReq_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} PhyDeactivateReq;

typedef struct PhyFindCellReq_s {
  Earfcn                  earfcn_start;
  Earfcn                  earfcn_end;
} PhyFindCellReq;

typedef struct PhyFindNextCellReq_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} PhyFindNextCellReq;

typedef struct PhyMeasThresholdReq_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} PhyMeasThresholdReq;

typedef struct PhyMeasReportInd_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} PhyMeasReportInd;

// UE: PHY -> RRC messages
typedef struct PhyFindCellInd_s {
  uint8_t                  cell_nb;
  CellInfo                 cells[MAX_REPORTED_CELL];
} PhyFindCellInd;

typedef struct PhyMeasThresholdConf_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} PhyMeasThresholdConf;
#endif /* PHY_MESSAGES_TYPES_H_ */

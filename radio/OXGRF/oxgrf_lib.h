/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

#include "oxgrf_api.h"
#include "common_lib.h"

/** @addtogroup _OXGRF_PHY_RF_INTERFACE_
 * @{
 */

typedef enum {
    Y230,
    Y240,
    Y380,
    Y390,
    Y550,
    Y590,
    Y750,
    Y790,
    IQX6000,
    IQX7000,
    IQX7100,
    IQX7200,
    IQX7400,
    IQX7402,  //split
    IQX7600,
    IQX8400,
    IQX8800,
    UNKNOWN = -1,
} OXGRFBoardType;

#if 0
typedef struct {
    OXGRFBoardType Model;
    struct {
        uint64_t sample_rate;
        uint64_t bandwidth;
        int8_t  tx_adv;
    } tables[16];
} oxgrf_cfg_table;

oxgrf_cfg_table TABLE_IQX8x00 = { // for IQX8200 & IQX8400 & IQX8800
    .Model = IQX8800,
    .tables = {
        {491520000, 400000000, 151},
        {245760000, 200000000, 126},
        {122880000, 100000000, 111},
        {61440000,  40000000,  172},
        {30720000,  20000000,  169},
        {15360000,  10000000,  110},
        {7680000,   5000000,   110},
        {3840000,   3000000,   110},
        {1920000,   1400000,   110},
    },
};

oxgrf_cfg_table TABLE_IQX7x00 = {  // for IQX7100 & IQX7400
    .Model = IQX7400,
    .tables = {
        {122880000, 100000000, 573},
        {61440000,  40000000,  286},
        {30720000,  20000000,  143},
        {15360000,  10000000,   73},
        {7680000,   5000000,    36},
        {3840000,   3000000,    17},
        {1920000,   1400000,    9},
    },
};

oxgrf_cfg_table TABLE_IQX7200 = {  // for IQX7200
    .Model = IQX7200,
    .tables = {
        {245760000, 200000000, 612},
        {122880000, 100000000, 306},
        {61440000,  40000000,  152},
        {30720000,  20000000,   76},
        {15360000,  10000000,   38},
        {7680000,   5000000,    19},
        {3840000,   3000000,    10},
        {1920000,   1400000,     5},
    },
};

oxgrf_cfg_table TABLE_Y590 = {  // for Y590neo
    .Model = Y590,
    .tables = {
        {245760000, 200000000, 502},
        {122880000, 100000000, 251},
        {61440000,  40000000,  125},
        {30720000,  20000000,   63},
        {15360000,  10000000,   32},
        {7680000,   5000000,    16},
        {3840000,   3000000,     8},
        {1920000,   1400000,     4},
    },
};

oxgrf_cfg_table TABLE_Y790 = {  // for Y790s New/Old Version & Y750s
    .Model = Y790,
    .tables = {
        {491520000, 400000000, 149},
        {245760000, 200000000, 125},
        {122880000, 100000000, 110},
        {61440000,  40000000,  172},
        {30720000,  20000000,  168},
    },
};

oxgrf_cfg_table TABLE_Y780 = {  // for Y780s New Version
    .Model = Y780,
    .tables = {
        {245760000, 200000000, 359},
        {122880000, 100000000, 179},
        {61440000,  40000000,   88},
        {30720000,  20000000,   43},
        {15360000,  10000000,   22},
        {7680000,   5000000,    11},
        {3840000,   3000000,     6},
        {1920000,   1400000,     3},
    },
};

#endif

/*! \brief OXGRF specific data structure */
typedef struct {

  //! opaque OXGRF device struct. An empty ("") or NULL device identifier will result in the first encountered device being opened (using the first discovered backend)
  OXGRF_DESCRIPTOR *dev;
  int16_t *rx_buffer;
  int16_t *tx_buffer;
  //! Sample rate
  unsigned int sample_rate;

  int rx_num_channels;
  int tx_num_channels;
  uint64_t tx_lo_freq;
  uint64_t rx_lo_freq;

  // --------------------------------
  // Debug and output control
  // --------------------------------
  //! Number of underflows
  int num_underflows;
  //! Number of overflows
  int num_overflows;
  //! number of RX errors
  int num_rx_errors;
  //! Number of TX errors
  int num_tx_errors;

  //! timestamp of current TX
  uint64_t tx_current_ts;
  //! timestamp of current RX
  uint64_t rx_current_ts;
  //! number of TX samples
  uint64_t tx_nsamps;
  //! number of RX samples
  uint64_t rx_nsamps;
  //! number of TX count
  uint64_t tx_count;
  //! number of RX count
  uint64_t rx_count;
  //! timestamp of RX packet
  openair0_timestamp rx_timestamp;
  OXGRFBoardType BoardType;
} oxgrf_state_t;

/*! \brief get current timestamp
 *\param device the hardware to use
 */
openair0_timestamp trx_get_timestamp(openair0_device *device);

/*@}*/

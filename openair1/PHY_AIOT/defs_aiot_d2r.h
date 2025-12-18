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

/*! \file PHY/defs_aiot_d2r.h
 \brief AIOT PHY layer D2R definitions
 \author Vojtech Masny
 \date 2025
 \version 0.1
 \company Eurecom
 \email: 
 \note
 \warning
*/

#ifndef __PHY_DEFS_AIOT_D2R__H__
#define __PHY_DEFS_AIOT_D2R__H__

// D-TAS preambles in length of 7 and 31 bits
#define D_TAS_7_BITS     0x4E
#define D_TAS_31_BITS    0x242BB1F3

#define POSTAMBLE_N 4
#define POSTAMBLE   0xF

const double T_BIT_TABLE[8] = { 2.0, 1.0, 1/2.0, 1/4.0, 1/8.0, 1/16.0, 1/32.0, 1/96.0 };
const int I_BIT_TABLE[4] = { 48, 96, 168, 240 };

#define MAX_AIOT_D2R_PAYLOAD_SIZE  125 // bytes
#define MAX_AIOT_D2R_PACKET_SIZE   (MAX_AIOT_D2R_PAYLOAD_SIZE + 2)

#endif
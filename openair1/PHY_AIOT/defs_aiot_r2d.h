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

/*! \file PHY/defs_aiot_R2D.h
 \brief AIOT PHY layer R2D definitions
 \author Vojtech Masny
 \date 2025
 \version 0.1
 \company Eurecom
 \email: 
 \note
 \warning
*/

#ifndef __PHY_DEFS_AIOT_R2D__H__
#define __PHY_DEFS_AIOT_R2D__H__

#define R_TAS_SIP    0xC8
#define N_R_TAS_SIP  8

#define R_TAS_CAP    0xA
#define N_R_TAS_CAP  4

#define R2D_POSTAMBLE    0xF
#define N_R2D_POSTAMBLE  4

#include "precomputed_SIP.h"
#include "precomputed_CAS.h"
#include "precomputed_payload.h"
#include "precomputed_Postamble.h"

#endif
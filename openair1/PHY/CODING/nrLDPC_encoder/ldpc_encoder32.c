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

/*!\file ldpc_encoder.c
 * \brief Defines the LDPC encoder
 * \author Florian Kaltenberger, Raymond Knopp, Kien le Trung (Eurecom)
 * \email openair_tech@eurecom.fr
 * \date 27-03-2018
 * \version 1.0
 * \note
 * \warning
 */



#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "defs.h"
#include "assertions.h"
#include "openair1/PHY/CODING/nrLDPC_defs.h"
#include "openair1/PHY/CODING/nrLDPC_extern.h"
#include "ldpc_generate_coefficient.c"


void cuda_support_init() {
   return;
}
int LDPCencoder32(uint8_t **input, uint32_t output[4][68*384], encoder_implemparams_t *impp)
{
	AssertFatal(1==0,"Should not be getting here\n");
}


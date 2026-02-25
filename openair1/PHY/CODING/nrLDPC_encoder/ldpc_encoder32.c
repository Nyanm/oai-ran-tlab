/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
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
uint32_t **LDPCencoder32(uint8_t **input,encoder_implemparams_t *impp)
{
	AssertFatal(1==0,"Should not be getting here\n");
}


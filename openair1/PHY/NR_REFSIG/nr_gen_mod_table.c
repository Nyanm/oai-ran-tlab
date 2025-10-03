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
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>
#include "nr_refsig.h"
#include "nr_mod_table.h"
c16_t nr_qpsk_mod_table[4];

int32_t nr_16qam_mod_table[16];

simde__m128i nr_qpsk_byte_mod_table[2048];

int64_t nr_16qam_byte_mod_table[1024];

int64_t nr_64qam_mod_table[4096];

int32_t nr_256qam_mod_table[512];

#ifdef FLT16_MAX
cf16_t nr_qpsk_mod_table_fp16[4];
int32_t nr_16qam_mod_table_fp16[16];
simde__m128i nr_qpsk_byte_mod_table_fp16[2048];
int64_t nr_16qam_byte_mod_table_fp16[1024];
int64_t nr_64qam_mod_table_fp16[4096];
int32_t nr_256qam_mod_table_fp16[512];
#endif

void nr_generate_modulation_table() {
  float sqrt2 = 0.70711;
  float sqrt10 = 0.31623;
  float sqrt42 = 0.15430;
  float sqrt170 = 0.076696;
  float val = 32768.0;
  int i,j;
  int16_t* table;
#ifdef FLT16_MAX
  _Float16* table2;
#endif
  // QPSK
  for (i=0; i<4; i++) {
    nr_qpsk_mod_table[i].r = (short)(1 - 2 * (i & 1)) * val * sqrt2 * sqrt2;
    nr_qpsk_mod_table[i].i = (short)(1 - 2 * ((i >> 1) & 1)) * val * sqrt2 * sqrt2;
#ifdef FLT16_MAX
    nr_qpsk_mod_table_fp16[i].r = (_Float16)((1 - 2 * (i & 1)) * val * sqrt2 * sqrt2);
    nr_qpsk_mod_table_fp16[i].i = (_Float16)((1 - 2 * ((i >> 1) & 1)) * val * sqrt2 * sqrt2);
#endif
    //printf("%d j%d\n",nr_qpsk_mod_table[i*2],nr_qpsk_mod_table[i*2+1]);
  }

  //QPSK m128
  table = (int16_t*) nr_qpsk_byte_mod_table;
#ifdef FLT16_MAX
  table2 = (_Float16*) nr_qpsk_byte_mod_table_fp16;
#endif
  for (i=0; i<256; i++) {
    for (j=0; j<4; j++) {
      *table++ = (int16_t)(1 - 2 * ((i >> (j * 2)) & 1)) * val * sqrt2 * sqrt2;
      *table++ = (int16_t)(1 - 2 * ((i >> (j * 2 + 1)) & 1)) * val * sqrt2 * sqrt2;
#ifdef FLT16_MAX
      *table2++ = (_Float16)((1 - 2 * ((i >> (j * 2)) & 1)) * sqrt2 * sqrt2);
      *table2++ = (_Float16)((1 - 2 * ((i >> (j * 2 + 1)) & 1)) * sqrt2 * sqrt2);
#endif
      //printf("%d j%d\n",nr_qpsk_byte_mod_table[i*8+(j*2)],nr_qpsk_byte_mod_table[i*8+(j*2)+1]);
    }
  }

  //16QAM
  table = (int16_t*) nr_16qam_byte_mod_table;
#ifdef FLT16_MAX
  table2 = (_Float16*) nr_16qam_byte_mod_table_fp16;
#endif
  for (i=0; i<256; i++) {
    for (j=0; j<2; j++) {
      *table++ = (int16_t)((1 - 2 * ((i >> (j * 4)) & 1)) * (2 - (1 - 2 * ((i >> (j * 4 + 2)) & 1)))) * val * sqrt10 * sqrt2;
      *table++ = (int16_t)((1 - 2 * ((i >> (j * 4 + 1)) & 1)) * (2 - (1 - 2 * ((i >> (j * 4 + 3)) & 1)))) * val * sqrt10 * sqrt2;
#ifdef FLT16_MAX
      *table2++ = (_Float16)(((1 - 2 * ((i >> (j * 4)) & 1)) * (2 - (1 - 2 * ((i >> (j * 4 + 2)) & 1)))) * sqrt10 * sqrt2);
      *table2++ = (_Float16)(((1 - 2 * ((i >> (j * 4 + 1)) & 1)) * (2 - (1 - 2 * ((i >> (j * 4 + 3)) & 1)))) * sqrt10 * sqrt2);
#endif
      // printf("%d j%d\n",nr_16qam_byte_mod_table[i*4+(j*2)],nr_16qam_byte_mod_table[i*4+(j*2)+1]);
    }
  }

  table = (int16_t*) nr_16qam_mod_table;
#ifdef FLT16_MAX
  table2 = (_Float16*) nr_16qam_mod_table_fp16;
#endif
  for (i=0; i<16; i++) {
    *table++ = (int16_t)((1 - 2 * (i & 1)) * (2 - (1 - 2 * ((i >> 2) & 1)))) * val * sqrt10 * sqrt2;
    *table++ = (int16_t)((1 - 2 * ((i >> 1) & 1)) * (2 - (1 - 2 * ((i >> 3) & 1)))) * val * sqrt10 * sqrt2;
#ifdef FLT16_MAX
    *table2++ = (_Float16)(((1 - 2 * (i & 1)) * (2 - (1 - 2 * ((i >> 2) & 1)))) * sqrt10 * sqrt2);
    *table2++ = (_Float16)(((1 - 2 * ((i >> 1) & 1)) * (2 - (1 - 2 * ((i >> 3) & 1)))) * sqrt10 * sqrt2);
#endif
      //printf("%d j%d\n",table[i*2],table[i*2+1]);
  }

  //64QAM
  table = (short*) nr_64qam_mod_table;
#ifdef FLT16_MAX
  table2 = (_Float16*) nr_64qam_mod_table_fp16;
#endif
  for (i=0; i<4096; i++) {
    for (j=0; j<2; j++) {
      *table++ = (short)((1 - 2 * ((i >> (j * 6)) & 1))
                         * (4 - (1 - 2 * ((i >> (j * 6 + 2)) & 1)) * (2 - (1 - 2 * ((i >> (j * 6 + 4)) & 1)))))
                 * val * sqrt42 * sqrt2;
      *table++ = (short)((1 - 2 * ((i >> (j * 6 + 1)) & 1))
                         * (4 - (1 - 2 * ((i >> (j * 6 + 3)) & 1)) * (2 - (1 - 2 * ((i >> (j * 6 + 5)) & 1)))))
                 * val * sqrt42 * sqrt2;
#ifdef FLT16_MAX
      *table2++ = (_Float16)((1 - 2 * ((i >> (j * 6)) & 1))
                         * (4 - (1 - 2 * ((i >> (j * 6 + 2)) & 1)) * (2 - (1 - 2 * ((i >> (j * 6 + 4)) & 1)))))
                 * sqrt42 * sqrt2;
      *table2++ = (_Float16)((1 - 2 * ((i >> (j * 6 + 1)) & 1))
                         * (4 - (1 - 2 * ((i >> (j * 6 + 3)) & 1)) * (2 - (1 - 2 * ((i >> (j * 6 + 5)) & 1)))))
                 * sqrt42 * sqrt2;
#endif
      //printf("%d j%d\n",table[i*4+(j*2)],table[i*4+(j*2)+1]);
    }
  }

  //256QAM
  table = (short*) nr_256qam_mod_table;
#ifdef FLT16_MAX
  table2 = (_Float16*) nr_256qam_mod_table_fp16;
#endif
  for (i=0; i<256; i++) {
    table[2*i] = (short)((1 - 2 * (i & 1))
                       * (8 - (1 - 2 * ((i >> 2) & 1)) * (4 - (1 - 2 * ((i >> 4) & 1)) * (2 - (1 - 2 * ((i >> 6) & 1))))))
               * val * sqrt170 * sqrt2;
    table[1+2*i] = (short)((1 - 2 * ((i >> 1) & 1))
                       * (8 - (1 - 2 * ((i >> 3) & 1)) * (4 - (1 - 2 * ((i >> 5) & 1)) * (2 - (1 - 2 * ((i >> 7) & 1))))))
               * val * sqrt170 * sqrt2;
#ifdef FLT16_MAX
    table2[2*i] = (_Float16)(((1 - 2 * (i & 1))
                       * (8 - (1 - 2 * ((i >> 2) & 1)) * (4 - (1 - 2 * ((i >> 4) & 1)) * (2 - (1 - 2 * ((i >> 6) & 1))))))
               * sqrt170 * sqrt2);
    table2[1+2*i] = (_Float16)(((1 - 2 * ((i >> 1) & 1))
                       * (8 - (1 - 2 * ((i >> 3) & 1)) * (4 - (1 - 2 * ((i >> 5) & 1)) * (2 - (1 - 2 * ((i >> 7) & 1))))))
               * sqrt170 * sqrt2);
   printf("256QAM %d : %f,%f, (%f,%f) (%f,%f)\n",i,(float)table2[2*i],(float)table2[1+2*i],
		  (float)((1 - 2 * (i & 1))
		                         * (8 - (1 - 2 * ((i >> 2) & 1)) * (4 - (1 - 2 * ((i >> 4) & 1)) * (2 - (1 - 2 * ((i >> 6) & 1))))))
		  * sqrt170 * sqrt2,
		  (float)((1 - 2 * ((i >> 1) & 1))
		                        * (8 - (1 - 2 * ((i >> 3) & 1)) * (4 - (1 - 2 * ((i >> 5) & 1)) * (2 - (1 - 2 * ((i >> 7) & 1))))))
		  * sqrt170 * sqrt2, ((double)table[2*i])/32768.0,((double)table[1+2*i])/32768  
		   
		   );
#endif
  }
}


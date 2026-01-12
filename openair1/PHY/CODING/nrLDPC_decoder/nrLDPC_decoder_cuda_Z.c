

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

/*!\file nrLDPC_decoder.c
 * \brief Defines thenrLDPC decoder
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
#include "openair1/PHY/CODING/coding_defs.h"
#include "openair1/PHY/CODING/nrLDPC_extern.h"
#include "nrLDPC_CUDA_shared_Z.h"

bool encoder_streamsCreated = false;
cudaStream_t encoderStreams[4];
extern int cuda_support_set;
int32_t LDPCinit_cuda()
{
    printf("Calling encoder initializations\n");
    if (cuda_support_set == 0) {
        cuda_support_init();
        printf("CUDA LDPC decoder initiating\n");
    }
    if (!encoder_streamsCreated) {
    for (int s = 0; s < 4; ++s) {
      cudaStreamCreateWithFlags(&encoderStreams[s], cudaStreamNonBlocking);
    }
    encoder_streamsCreated = true;
    // Initialize 
    ldpc_decoder_cuda_init();
    return 0;
    
    }
}


uint8_t reverse_bits_test(uint8_t b) {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}


static inline void nrLDPC_decoder_core( int8_t* p_llr,
                                        uint8_t* p_out,
                                        int num_cws
                                    )
{
    int num_batches = (num_cws + CWS_PER_BATCH - 1) / CWS_PER_BATCH;
    int batch = 0;

    while (batch < num_batches) {
        printf("==================== Batch %d/%d ====================\n", batch + 1, num_batches);

        int cw_start = batch * CWS_PER_BATCH;
        int cw_in_this_batch = (batch == num_batches - 1) 
                            ? (num_cws - cw_start) 
                            : CWS_PER_BATCH;

        // ===== Input: Host -> Pinned  =======
        size_t llr_bytes_to_copy = cw_in_this_batch * BG1_MAX_CW_LEN * sizeof(int8_t);
        memcpy(host_mem->h_big_pinned_llr,
            p_llr + cw_start * BG1_MAX_CW_LEN,
            llr_bytes_to_copy);

        // Padding only needed for last batch
        if (batch == num_batches - 1 && cw_in_this_batch < CWS_PER_BATCH) {
            size_t padding_bytes = (CWS_PER_BATCH - cw_in_this_batch) * BG1_MAX_CW_LEN * sizeof(int8_t);
            memset((uint8_t*)host_mem->h_big_pinned_llr + llr_bytes_to_copy,
                0,
                padding_bytes);
        }

        // ===== GPU Execution =====
        for (int s = 0; s < MAX_STREAMS; s++) {
            CUDA_CHECK(cudaGraphLaunch(cudaGraphExecs[s], cudaStreams[s]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== Output: Pinned -> Host =====
        // size_t hard_bytes_to_copy = cw_in_this_batch * (BG1_MAX_INFO_LEN / 8) * sizeof(uint8_t);
        // memcpy(p_out + cw_start * (BG1_MAX_INFO_LEN / 8),
        //     host_mem->h_big_hard_bits,
        //     hard_bytes_to_copy);

        for (int i = 0; i < cw_in_this_batch; ++i) {
            memcpy(p_out + i * BG1_MAX_INFO_LEN,
                host_mem->h_big_hard_bits + i * BG1_MAX_INFO_LEN / 8,
                BG1_MAX_INFO_LEN / 8);
        }


        for (int i = 0; i < cw_in_this_batch*BG1_MAX_INFO_LEN; ++i) {
            p_out[i] = reverse_bits_test(p_out[i]);
        }

        // FILE*f_in;
        // f_in = fopen("ldpc_input.bin","wb");
        // fwrite(p_llr + cw_start * BG1_MAX_CW_LEN,sizeof(int8_t),cw_in_this_batch * BG1_MAX_CW_LEN,f_in);
        // fclose(f_in);

        // FILE*f_out;
        // f_out = fopen("ldpc_output.bin","wb");
        // fwrite(p_out + cw_start * (BG1_MAX_INFO_LEN / 8),sizeof(uint8_t),cw_in_this_batch * (BG1_MAX_INFO_LEN / 8),f_out);
        // fclose(f_out);

        batch++;
    }
}

int32_t LDPCdecoder_cuda(t_nrLDPC_dec_params* p_decParams,
                         int8_t* p_llr,
                         uint8_t* p_out,
                         t_nrLDPC_time_stats* p_profiler,
                         decode_abort_t* ab)
{
    e_nrLDPC_outMode outMode = p_decParams->outMode;
    if ((p_decParams->BG != 1) || (p_decParams->Z != 384)) { // format check
        AssertFatal(false, "Format cuda not support, only support BG = 1, Zc = 384 right now\n");
        return 0;
    }
    if(p_decParams->R != 13 && p_decParams->R != 23) {
        AssertFatal(false, "Only support rate 1/3 and 2/3 in cuda decoder\n");
        return 0;
    }
    // nrLDPC_outMode_BIT is default 
    if(outMode != nrLDPC_outMode_BIT && outMode != nrLDPC_outMode_BITINT8) {
        AssertFatal(false, "Only support output mode BIT and BITINT8 in cuda decoder\n");
        return 0;
    }
    nrLDPC_decoder_core(
        p_llr,
        p_out,
        p_decParams->n_segments
    );

    set_abort(ab, true);

    return params->n_iterations;
}

// shutdown LDPC decoder and free resources
int32_t LDPCshutdown_cuda(){

    for (int s=0; s< 4; s++) {
    if (encoder_streamsCreated) {
      cudaStreamDestroy(encoderStreams[s]);
    }
  }

    ldpc_decoder_cuda_free();

    return 0;

}


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

extern int cuda_support_set;

int32_t LDPCinit_cuda()
{
    printf("Calling encoder initializations\n");
    if (cuda_support_set == 0) {
        cuda_support_init();
        printf("CUDA LDPC decoder initiating\n");
    }

    params = (struct ldpc_params*)malloc(sizeof(struct ldpc_params));

    // Decoder parameter configuration
    params->rate = 13;
    params->bg_index = 1;
    params->Z = 384;
    params->N = 26112;
    params->K = 8448;
    params->Kb = 22;
    params->mb = BG1_ROW;
    params->nb = BG1_COL;
    params->H = h_base_1_i1;   // Parity-check matrix H
    params->compH_cn = NULL;
    params->compH_vn = NULL;  // Compressed representation of H for CN and VN processing
    params->n_iterations = 5;
    params->num_streams = 6;
    params->num_group_per_stream = 10; 
    params->num_cw_per_stream = params->num_group_per_stream * SIMD_WIDTH;  // Each group contains 4 codewords (SIMD4)
    params->num_cw_per_batch = params->num_streams * params->num_cw_per_stream;

    // Initialize CUDA kernel grid and block dimensions
    setup_cuda_grid(params->Z, params->mb, params->nb, params->Kb, params->num_group_per_stream, params->num_cw_per_stream);

    // Allocate host and device memory
    setup_host_memory(params->num_streams, params->num_cw_per_stream);
    setup_device_memory(params->num_streams, params->num_group_per_stream, params->num_cw_per_stream);

    // Initialize GPU constant memory
    init_decoder_constant(params->H, params->mb, params->nb);

    // Create CUDA streams
    cudaStreams = (cudaStream_t*)malloc(params->num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < params->num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    /*
        cudaGraph_t represents an editable graph (a "capture" graph) that can be modified
        (e.g., by adding nodes or edges) but cannot be executed directly.
        It must be instantiated via cudaGraphInstantiate() to produce a cudaGraphExec_t,
        which is the executable form of the graph.
    */
    cudaGraphs = (cudaGraph_t*)malloc(params->num_streams * sizeof(cudaGraph_t));
    cudaGraphExecs = (cudaGraphExec_t*)malloc(params->num_streams * sizeof(cudaGraphExec_t));

    if(params->bg_index == 1 && params->Z == 384 && params->rate == 13){
        init_graph_BG1_R13_Z384(
            params->n_iterations,
            params->num_streams,
            params->num_group_per_stream,
            params->num_cw_per_stream
        );

    }else{
        printf("LDPC decoder for the specified BG and Z is not implemented yet.\n");
        exit(-1);
    }
}

int32_t LDPCdecoder_cuda(t_nrLDPC_dec_params* p_decParams,
                         int8_t* p_llr,
                         uint8_t* p_out,
                         t_nrLDPC_time_stats* p_profiler,
                         decode_abort_t* ab)
{
    e_nrLDPC_outMode outMode = p_decParams->outMode;
    if ((p_decParams->R != 13) || (p_decParams->BG != 1) || (p_decParams->Z != 384)) { // format check
        AssertFatal(false, "Format cuda not support, only support BG = 1, Zc = 384 and R = 13 right now\n");
        return 0;
    }
    nrLDPC_decoder_core(
        p_llr,
        p_out,
        p_decParams->n_segments,
        params->num_streams,
        params->num_cw_per_stream
    );

    set_abort(ab, true);

    return params->n_iterations;
}

static inline void nrLDPC_decoder_core( int8_t* p_llr,
                                        uint8_t* p_out,
                                        int num_cws,
                                        int num_streams,
                                        int num_cw_per_stream
                                    )
{
    const int num_cw_per_batch = num_streams * num_cw_per_stream; // 显式定义
    const int num_batches = (num_cws + num_cw_per_batch - 1) / num_cw_per_batch;
    int batch = 0;

    while (batch < num_batches) {
        printf("==================== Batch %d/%d =====================\n", batch + 1, num_batches);

        int cw_start = batch * num_cw_per_batch;
        int cw_in_this_batch = (batch == num_batches - 1) 
                            ? (num_cws - cw_start) 
                            : num_cw_per_batch;

        // ===== Input: Host -> Pinned (整块拷贝 + padding if last batch) =====
        size_t llr_bytes_to_copy = cw_in_this_batch * BG1_MAX_CW_LEN * sizeof(int8_t);
        memcpy(host_mem->h_big_pinned_llr,
            p_llr + cw_start * BG1_MAX_CW_LEN,
            llr_bytes_to_copy);

        // Padding only needed for last batch
        if (batch == num_batches - 1 && cw_in_this_batch < num_cw_per_batch) {
            size_t padding_bytes = (num_cw_per_batch - cw_in_this_batch) * BG1_MAX_CW_LEN * sizeof(int8_t);
            memset((uint8_t*)host_mem->h_big_pinned_llr + llr_bytes_to_copy,
                0,
                padding_bytes);
        }

        // ===== GPU Execution =====
        for (int s = 0; s < num_streams; s++) {
            CUDA_CHECK(cudaGraphLaunch(cudaGraphExecs[s], cudaStreams[s]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== Output: Pinned -> Host =====
        size_t hard_bytes_to_copy = cw_in_this_batch * (BG1_R13_Z384_K / 8) * sizeof(uint8_t);
        memcpy(p_out + cw_start * (BG1_R13_Z384_K / 8),
            host_mem->h_big_hard_bits,
            hard_bytes_to_copy);

        batch++;
    }
}


template<typename T>
void free_global_memory(T** d_ptr, T* big_block) {
    if(d_ptr){
        free(d_ptr);
    }
    if(big_block){
        CUDA_CHECK(cudaFree(big_block));
    }
}

// Free pinned host memory for multi-stream usage
template<typename T>
void free_pinned_memory(T** h_ptr, T* big_block) {
    if (h_ptr) {
        free(h_ptr);
    }
    if (big_block) {
        CUDA_CHECK(cudaFreeHost(big_block));
    }
}

// Free all global device memory
void free_device_mem_all(){
    if (dev_mem == NULL) return;

    free_global_memory<int8_t>(dev_mem->d_init_llr, dev_mem->d_big_init_llr);
    free_global_memory<uint32_t>(dev_mem->d_app, dev_mem->d_big_app);  
    free_global_memory<uint32_t>(dev_mem->d_c2v, dev_mem->d_big_c2v);
    free_global_memory<uint32_t>(dev_mem->d_delta_c2v, dev_mem->d_big_delta_c2v);
    free_global_memory<int8_t>(dev_mem->d_app_reordered, dev_mem->d_big_app_reordered);
    free_global_memory<uint8_t>(dev_mem->d_hard_bits, dev_mem->d_big_hard_bits);

    // Free the structure itself
    free(dev_mem);
}

// Free all host pinned memory
void free_host_mem_all() {
    if (host_mem == NULL) return;

    free_pinned_memory<int8_t>(host_mem->h_pinned_llr, host_mem->h_big_pinned_llr);
    free_pinned_memory<uint8_t>(host_mem->h_pinned_hard, host_mem->h_big_hard_bits);

    // Free the structure itself
    free(host_mem);
}

// shutdown LDPC decoder and free resources
int32_t LDPCshutdown_cuda(){

    int num_streams = params->num_streams;

    // destroy graphs and streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(cudaStreams[i]));
        CUDA_CHECK(cudaGraphExecDestroy(cudaGraphExecs[i]));
        CUDA_CHECK(cudaGraphDestroy(cudaGraphs[i]));
    }
    free(cudaStreams);
    free(cudaGraphs);
    free(cudaGraphExecs);

    // free memory
    free(g);
    free_device_mem_all();
    free_host_mem_all();
    free(params);
    return 0;
}
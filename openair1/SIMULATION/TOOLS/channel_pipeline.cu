#include <stdio.h>
#include <cuda_runtime.h>
#include "oai_cuda.h"

extern "C" {

void run_channel_pipeline_cuda(
    float **tx_sig_re, float **tx_sig_im,
    c16_t **output_signal,
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples,
    float path_loss, float *h_channel_coeffs,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map,
    int slot_offset, int delay,
    void *d_tx_sig_void, void *d_intermediate_sig_void, void* d_final_output_void,
    void *d_curand_states_void, void* h_tx_sig_pinned_void, void* h_final_output_pinned_void
)
{
    float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float2 *d_intermediate_sig = (float2*)d_intermediate_sig_void;
    short2 *d_final_output = (short2*)d_final_output_void;
    curandState_t *d_curand_states = (curandState_t*)d_curand_states_void;
    float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
    short2* h_final_output_pinned = (short2*)h_final_output_pinned_void;

    // --- STAGE 1: Copy Transmit Signal (Host to Device) ---
    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }
    cudaMemcpy(d_tx_sig, h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice);

    // --- STAGE 2: Run Multipath Channel Kernel ---
    update_channel_coeffs_symbol(h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2));
 
    
    dim3 threads_multipath(512, 1);
    dim3 blocks_multipath((num_samples + threads_multipath.x - 1) / threads_multipath.x, nb_rx);
    size_t sharedMemSize = (threads_multipath.x + channel_length - 1) * sizeof(float2);

    multipath_channel_kernel<<<blocks_multipath, threads_multipath, sharedMemSize>>>(
        d_tx_sig, d_intermediate_sig, num_samples, channel_length, nb_tx, nb_rx, path_loss);

    // --- STAGE 3: Run Noise Kernel (reads directly from intermediate buffer) ---
    dim3 threads_noise(256, 1);
    dim3 blocks_noise((num_samples + threads_noise.x - 1) / threads_noise.x, nb_rx);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;
    
    add_noise_and_phase_noise_kernel<<<blocks_noise, threads_noise>>>(
        d_intermediate_sig, d_final_output, d_curand_states, num_samples,
        sqrtf(sigma2 / 2.0f), sqrtf(pn_variance), 
        pdu_bit_map, ptrs_bit_map
    );

    // --- STAGE 4: Copy Final Result (Device to Host) ---
    cudaMemcpy(
        h_final_output_pinned,
        d_final_output,
        nb_rx * num_samples * sizeof(short2),
        cudaMemcpyDeviceToHost
    );
    
    // --- STAGE 5: Distribute to Final OAI Buffers with Offset ---
    for (int ii = 0; ii < nb_rx; ii++) {
        memcpy(
            // Apply the slot_offset and delay to place data correctly
            output_signal[ii] + slot_offset + delay, 
            h_final_output_pinned + ii * num_samples,
            num_samples * sizeof(short2)
        );
    }
}
} // extern "C"
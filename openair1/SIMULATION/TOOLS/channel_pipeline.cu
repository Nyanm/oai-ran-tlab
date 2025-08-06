#include <stdio.h>
#include <cuda_runtime.h>
#include "oai_cuda.h"


#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
static void checkCuda(cudaError_t result, const char* const func, const char *const file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

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
    void *d_curand_states_void, void* h_tx_sig_pinned_void, void* h_final_output_pinned_void,
    void *d_channel_coeffs_void
)
{
    // --- Cast void pointers ---
    float2 *d_intermediate_sig = (float2*)d_intermediate_sig_void;
    short2 *d_final_output = (short2*)d_final_output_void;
    curandState_t *d_curand_states = (curandState_t*)d_curand_states_void;
    float2 *d_channel_coeffs = (float2*)d_channel_coeffs_void;

    // --- Determine the correct input pointer ---
    float2* kernel_input_ptr;
#if defined(USE_UNIFIED_MEMORY) || defined(USE_ATS_MEMORY)
    // For UM and ATS, the CPU prepares the data and the kernel can access it directly.
    // The h_tx_sig_pinned pointer holds the host-accessible (malloc or managed) data.
    kernel_input_ptr = (float2*)h_tx_sig_pinned_void;
#else
    // For explicit copy, we must copy from host to a separate device buffer.
    float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }
    CHECK_CUDA( cudaMemcpy(d_tx_sig, h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice) );
    kernel_input_ptr = d_tx_sig;
#endif

    size_t channel_size_bytes = nb_tx * nb_rx * channel_length * sizeof(float2);
    CHECK_CUDA( cudaMemcpy(d_channel_coeffs, h_channel_coeffs, channel_size_bytes, cudaMemcpyHostToDevice) );


    // --- STAGE 2: Run Multipath Channel Kernel ---
    // update_channel_coeffs_symbol(h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2));
    dim3 threads_multipath(512, 1);
    dim3 blocks_multipath((num_samples + threads_multipath.x - 1) / threads_multipath.x, nb_rx);
    size_t sharedMemSize = (threads_multipath.x + channel_length - 1) * sizeof(float2);
    multipath_channel_kernel<<<blocks_multipath, threads_multipath, sharedMemSize>>>(
        d_channel_coeffs,
        kernel_input_ptr, d_intermediate_sig, num_samples, channel_length, nb_tx, nb_rx, path_loss);

    // --- STAGE 3: Run Noise Kernel ---
    dim3 threads_noise(256, 1);
    dim3 blocks_noise((num_samples + threads_noise.x - 1) / threads_noise.x, nb_rx);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;
    add_noise_and_phase_noise_kernel<<<blocks_noise, threads_noise>>>(
        d_intermediate_sig, d_final_output, d_curand_states, num_samples,
        sqrtf(sigma2 / 2.0f), sqrtf(pn_variance), 
        pdu_bit_map, ptrs_bit_map
    );

    // --- Synchronize and Finalize Output ---
    cudaDeviceSynchronize();


#if defined(USE_UNIFIED_MEMORY)
    // For UM, the CPU can read the result directly from the managed buffer.
    short2* h_final_output_pinned = (short2*)h_final_output_pinned_void;
    for (int ii = 0; ii < nb_rx; ii++) {
        memcpy(
            output_signal[ii] + slot_offset + delay, 
            h_final_output_pinned + ii * num_samples,
            num_samples * sizeof(short2)
        );
    }
#else
    // For Explicit Copy and ATS, we need an explicit DtoH copy.
    short2* h_final_output_pinned = (short2*)h_final_output_pinned_void;
    CHECK_CUDA( cudaMemcpy(
        h_final_output_pinned,
        d_final_output,
        nb_rx * num_samples * sizeof(short2),
        cudaMemcpyDeviceToHost
    ));
    for (int ii = 0; ii < nb_rx; ii++) {
        memcpy(
            output_signal[ii] + slot_offset + delay, 
            h_final_output_pinned + ii * num_samples,
            num_samples * sizeof(short2)
        );
    }
#endif
}
} // extern "C"
#include <stdio.h>
#include <cuda_runtime.h>
#include "oai_cuda.h"


__global__ void multipath_channel_kernel_batched(
    const float2* __restrict__ d_channel_coeffs, const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig, int num_samples, int channel_length,
    int nb_tx, int nb_rx, const float* __restrict__ path_loss_batch);

__global__ void add_noise_and_phase_noise_kernel_batched(
    const float2* __restrict__ r_sig, short2* __restrict__ output_sig,
    curandState_t* states, int num_samples, int nb_rx, float sigma,
    float pn_std_dev, uint16_t pdu_bit_map, uint16_t ptrs_bit_map);

#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
static void checkCuda(cudaError_t result, const char* const func, const char *const file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


__global__ void sum_outputs_kernel(
    const short2* __restrict__ * __restrict__ individual_outputs,
    short2* __restrict__ final_summed_output,
    int num_channels,
    int num_samples_per_antenna
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples_per_antenna) return;

    float2 sum = make_float2(0.0f, 0.0f);

    for (int c = 0; c < num_channels; c++) {
        sum.x += individual_outputs[c][i].x;
        sum.y += individual_outputs[c][i].y;
    }

    final_summed_output[i].x = (short)fmaxf(-32768.0f, fminf(32767.0f, sum.x));
    final_summed_output[i].y = (short)fmaxf(-32768.0f, fminf(32767.0f, sum.y));
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

    float2* kernel_input_ptr;
    #if defined(USE_UNIFIED_MEMORY) || defined(USE_ATS_MEMORY)
            kernel_input_ptr = (float2*)h_tx_sig_pinned_void;
    #else
            // Explicit copy model
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

    dim3 threads_multipath(512, 1);
    dim3 blocks_multipath((num_samples + threads_multipath.x - 1) / threads_multipath.x, nb_rx);
    size_t sharedMemSize = (threads_multipath.x + channel_length - 1) * sizeof(float2);
    multipath_channel_kernel<<<blocks_multipath, threads_multipath, sharedMemSize>>>(
        d_channel_coeffs,
        kernel_input_ptr, d_intermediate_sig, num_samples, channel_length, nb_tx, nb_rx, path_loss);

    dim3 threads_noise(256, 1);
    dim3 blocks_noise((num_samples + threads_noise.x - 1) / threads_noise.x, nb_rx);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;
    add_noise_and_phase_noise_kernel<<<blocks_noise, threads_noise>>>(
        d_intermediate_sig, d_final_output, d_curand_states, num_samples,
        sqrtf(sigma2 / 2.0f), sqrtf(pn_variance), 
        pdu_bit_map, ptrs_bit_map
    );

    cudaDeviceSynchronize();

    // If output_signal is NULL, the caller intends to keep the data on the GPU
    // for further processing (e.g., summing outputs). Otherwise, copy back to host.
    if (output_signal != NULL) {
        #if defined(USE_UNIFIED_MEMORY)
            short2* h_final_output_pinned = (short2*)h_final_output_pinned_void;
            for (int ii = 0; ii < nb_rx; ii++) {
                memcpy(
                    output_signal[ii] + slot_offset + delay,
                    h_final_output_pinned + ii * num_samples,
                    num_samples * sizeof(short2)
                );
            }
        #else
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
}


void sum_channel_outputs_cuda(
    void **d_individual_outputs,
    void *d_final_output,
    int num_channels,
    int nb_rx,
    int num_samples
)
{
    void **d_ptr_array;
    size_t ptr_array_size = num_channels * sizeof(void*);
    CHECK_CUDA( cudaMalloc(&d_ptr_array, ptr_array_size) );
    
    // Copy the array of device pointers from host to device
    CHECK_CUDA( cudaMemcpy(d_ptr_array, d_individual_outputs, ptr_array_size, cudaMemcpyHostToDevice) );

    int num_total_samples = nb_rx * num_samples;
    dim3 threads(256, 1);
    dim3 blocks((num_total_samples + threads.x - 1) / threads.x, 1);

    sum_outputs_kernel<<<blocks, threads>>>(
        (const short2**)d_ptr_array,
        (short2*)d_final_output,
        num_channels,
        num_total_samples
    );

    CHECK_CUDA( cudaFree(d_ptr_array) );
}


void run_channel_pipeline_cuda_streamed(
    float **tx_sig_re, float **tx_sig_im,
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples,
    float path_loss, float *h_channel_coeffs,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map,
    void *d_tx_sig_void, void *d_intermediate_sig_void, void* d_final_output_void,
    void *d_curand_states_void, void* h_tx_sig_pinned_void,
    void *d_channel_coeffs_void,
    void* stream_void)
{
    cudaStream_t stream = (cudaStream_t)stream_void;

    float2 *d_intermediate_sig = (float2*)d_intermediate_sig_void;
    short2 *d_final_output = (short2*)d_final_output_void;
    curandState_t *d_curand_states = (curandState_t*)d_curand_states_void;
    float2 *d_channel_coeffs = (float2*)d_channel_coeffs_void;
    float2* kernel_input_ptr;

    #if defined(USE_UNIFIED_MEMORY) || defined(USE_ATS_MEMORY)
            // In these modes, the GPU can access host memory directly.
            float2* h_tx_sig_interleaved = (float2*)h_tx_sig_pinned_void;
            for (int j = 0; j < nb_tx; j++) {
                for (int i = 0; i < num_samples; i++) {
                    h_tx_sig_interleaved[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                }
            }
            kernel_input_ptr = h_tx_sig_interleaved;
    #else
            // Explicit Copy model
            float2 *d_tx_sig = (float2*)d_tx_sig_void;
            float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
            for (int j = 0; j < nb_tx; j++) {
                for (int i = 0; i < num_samples; i++) {
                    h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                }
            }
            CHECK_CUDA( cudaMemcpyAsync(d_tx_sig, h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice, stream) );
            kernel_input_ptr = d_tx_sig;
    #endif

    size_t channel_size_bytes = nb_tx * nb_rx * channel_length * sizeof(float2);
    CHECK_CUDA( cudaMemcpyAsync(d_channel_coeffs, h_channel_coeffs, channel_size_bytes, cudaMemcpyHostToDevice, stream) );

    // --- Launch Kernels Asynchronously in the Stream ---
    dim3 threads_multipath(512, 1);
    dim3 blocks_multipath((num_samples + threads_multipath.x - 1) / threads_multipath.x, nb_rx);
    size_t sharedMemSize = (threads_multipath.x + channel_length - 1) * sizeof(float2);
    multipath_channel_kernel<<<blocks_multipath, threads_multipath, sharedMemSize, stream>>>(
        d_channel_coeffs, kernel_input_ptr, d_intermediate_sig, num_samples, channel_length, nb_tx, nb_rx, path_loss);

    dim3 threads_noise(256, 1);
    dim3 blocks_noise((num_samples + threads_noise.x - 1) / threads_noise.x, nb_rx);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;

    add_noise_and_phase_noise_kernel<<<blocks_noise, threads_noise, 0, stream>>>(
        d_intermediate_sig, d_final_output, d_curand_states, num_samples,
        sqrtf(sigma2 / 2.0f), sqrtf(pn_variance), 
        pdu_bit_map, ptrs_bit_map
    );
}


void run_channel_pipeline_cuda_batched(
    int num_channels,
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples,
    void *d_path_loss_batch, void *d_channel_coeffs_batch,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map,
    void *d_tx_sig_batch, void *d_intermediate_sig_batch, void *d_final_output_batch,
    void *d_curand_states)
{

    float2 *d_tx = (float2*)d_tx_sig_batch;
    float2 *d_intermediate = (float2*)d_intermediate_sig_batch;
    short2 *d_final = (short2*)d_final_output_batch;
    float2 *d_coeffs = (float2*)d_channel_coeffs_batch;
    float *d_pl = (float*)d_path_loss_batch;
    curandState_t *d_states = (curandState_t*)d_curand_states;

    dim3 threads_multipath(512, 1, 1);
    dim3 blocks_multipath((num_samples + threads_multipath.x - 1) / threads_multipath.x, nb_rx, num_channels);
    size_t sharedMemSize = (threads_multipath.x + channel_length - 1) * sizeof(float2);

    multipath_channel_kernel_batched<<<blocks_multipath, threads_multipath, sharedMemSize>>>(
        d_coeffs, d_tx, d_intermediate, num_samples, channel_length, nb_tx, nb_rx, d_pl);

    dim3 threads_noise(256, 1, 1);
    dim3 blocks_noise((num_samples + threads_noise.x - 1) / threads_noise.x, nb_rx, num_channels);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;

    add_noise_and_phase_noise_kernel_batched<<<blocks_noise, threads_noise>>>(
        d_intermediate, d_final, d_states, num_samples, nb_rx,
        sqrtf(sigma2 / 2.0f), sqrtf(pn_variance), 
        pdu_bit_map, ptrs_bit_map
    );

    // Note that synchronization happens in the benchmark (test_channel_scalability) after this call returns.
}

} // extern "C"
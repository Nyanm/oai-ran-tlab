#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "oai_cuda.h"

// --- CUDA Helper Functions ---

#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)
static void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
        exit(1);
    }
}

__device__ __forceinline__ float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// --- Kernel to initialize cuRAND states ---

__global__ void init_curand_states_kernel(curandState_t *states, unsigned long long seed, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// --- Main Noise Kernel ---

__global__ void add_noise_and_phase_noise_kernel(
    const float2* __restrict__ r_sig,
    short2* __restrict__ output_sig,
    curandState_t* states,
    int num_samples,
    float sigma,
    float pn_std_dev,
    uint16_t pdu_bit_map,
    uint16_t ptrs_bit_map
)
{
    __shared__ curandState_t shared_states[512]; // Shared memory for cuRAND states

    const int tid = threadIdx.x;
    const int i = (blockIdx.x * blockDim.x + tid) * 2; // Process two samples per thread
    const int ii = blockIdx.y;

    // Load cuRAND state into shared memory
    if (tid < blockDim.x && i < num_samples) {
        shared_states[tid] = states[ii * num_samples + i];
    }
    __syncthreads();

    if (i >= num_samples) return;

    curandState_t local_state = shared_states[tid];

    // Process first sample
    float2 noisy_signal = r_sig[ii * num_samples + i];
    float2 awgn = curand_normal2(&local_state);
    noisy_signal.x += awgn.x * sigma;
    noisy_signal.y += awgn.y * sigma;
    float2 final_signal = noisy_signal;

    if (pdu_bit_map & ptrs_bit_map) {
        float phase_error = curand_normal(&local_state) * pn_std_dev;
        float cos_phi, sin_phi;
        __sincosf(phase_error, &sin_phi, &cos_phi);
        float2 phase_rot = make_float2(cos_phi, sin_phi);
        final_signal = complex_mul(noisy_signal, phase_rot);
    }

    output_sig[ii * num_samples + i] = make_short2(
        (short)fmaxf(-32768.0f, fminf(32767.0f, final_signal.x)),
        (short)fmaxf(-32768.0f, fminf(32767.0f, final_signal.y))
    );

    // Process second sample if within bounds
    if (i + 1 < num_samples) {
        noisy_signal = r_sig[ii * num_samples + i + 1];
        awgn = curand_normal2(&local_state);
        noisy_signal.x += awgn.x * sigma;
        noisy_signal.y += awgn.y * sigma;
        final_signal = noisy_signal;

        if (pdu_bit_map & ptrs_bit_map) {
            float phase_error = curand_normal(&local_state) * pn_std_dev;
            float cos_phi, sin_phi;
            __sincosf(phase_error, &sin_phi, &cos_phi);
            float2 phase_rot = make_float2(cos_phi, sin_phi);
            final_signal = complex_mul(noisy_signal, phase_rot);
        }

        output_sig[ii * num_samples + i + 1] = make_short2(
            (short)fmaxf(-32768.0f, fminf(32767.0f, final_signal.x)),
            (short)fmaxf(-32768.0f, fminf(32767.0f, final_signal.y))
        );
    }

    // Update shared state
    shared_states[tid] = local_state;
    __syncthreads();

    // Write back to global memory
    if (tid < blockDim.x && i < num_samples) {
        states[ii * num_samples + i] = shared_states[tid];
    }
}

// --- Main Noise Function ---

extern "C" {
void add_noise_cuda_fast(
    const float **r_re,
    const float **r_im,
    c16_t **output_signal,
    int num_samples,
    int nb_rx,
    float sigma2,
    double ts,
    int slot_offset,
    int delay,
    uint16_t pdu_bit_map,
    uint16_t ptrs_bit_map,
    void *d_r_sig_void,
    void *d_output_sig_void,
    void *d_curand_states_void,
    void *h_r_sig_void,
    void *h_output_temp_void
)
{
    float2 *d_r_sig = (float2*)d_r_sig_void;
    short2 *d_output_sig = (short2*)d_output_sig_void;
    curandState_t *d_curand_states = (curandState_t*)d_curand_states_void;
    float2* h_r_sig = (float2*)h_r_sig_void;
    short2* h_output_temp = (short2*)h_output_temp_void;

    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Copy input signal to pinned host memory
    for (int ii = 0; ii < nb_rx; ii++) {
        for (int i = 0; i < num_samples; i++) {
            h_r_sig[ii * num_samples + i] = make_float2(r_re[ii][i], r_im[ii][i]);
        }
    }

    // Asynchronous transfer to device
    CHECK_CUDA(cudaMemcpyAsync(d_r_sig, h_r_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyHostToDevice, stream));

    // Configure kernel launch with dynamic grid sizing
    const int SM_COUNT = 132; // NVIDIA H100
    const int threads_per_block = 512;
    const int samples_per_thread = 2;
    int effective_threads = (num_samples + samples_per_thread - 1) / samples_per_thread;
    int blocks_x = (effective_threads + threads_per_block - 1) / threads_per_block;
    int full_waves = (blocks_x + SM_COUNT - 1) / SM_COUNT;
    if (full_waves * SM_COUNT * threads_per_block * samples_per_thread > num_samples * 1.1) {
        blocks_x = (effective_threads + threads_per_block - 1) / threads_per_block;
    } else {
        blocks_x = full_waves * SM_COUNT;
    }
    dim3 numBlocks(blocks_x, nb_rx);
    float pn_variance = 1e-5f * 2.0f * 3.1415926535f * 300.0f * (float)ts;
    float sigma = sqrtf(sigma2 / 2.0f);
    float pn_std_dev = sqrtf(pn_variance);

    // Launch kernel in stream
    add_noise_and_phase_noise_kernel<<<numBlocks, threads_per_block, 0, stream>>>(
        d_r_sig, d_output_sig, d_curand_states, num_samples,
        sigma, pn_std_dev, pdu_bit_map, ptrs_bit_map
    );

    // Asynchronous copy results back to host
    CHECK_CUDA(cudaMemcpyAsync(
        h_output_temp,
        d_output_sig,
        nb_rx * num_samples * sizeof(short2),
        cudaMemcpyDeviceToHost,
        stream
    ));

    // Synchronize stream
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Destroy stream
    CHECK_CUDA(cudaStreamDestroy(stream));

    // Distribute to output_signal
    for (int ii = 0; ii < nb_rx; ii++) {
        memcpy(
            output_signal[ii] + slot_offset + delay,
            h_output_temp + ii * num_samples,
            num_samples * sizeof(short2)
        );
    }
}

// --- Helper Functions for cuRAND State Management ---

void* create_and_init_curand_states_cuda(int num_elements, unsigned long long seed) {
    void* d_states_void;
    CHECK_CUDA(cudaMalloc(&d_states_void, num_elements * sizeof(curandState_t)));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    const int SM_COUNT = 132; // NVIDIA H100
    const int threads = (num_elements > 100000) ? 1024 : 512; // Dynamic block size
    int blocks = (num_elements + threads - 1) / threads;
    int full_waves = (blocks + SM_COUNT - 1) / SM_COUNT;
    blocks = full_waves * SM_COUNT;
    init_curand_states_kernel<<<blocks, threads, 0, stream>>>( (curandState_t*)d_states_void, seed, num_elements);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return d_states_void;
}

void destroy_curand_states_cuda(void* d_curand_states) {
    if (d_curand_states) {
        CHECK_CUDA(cudaFree(d_curand_states));
    }
}

} // extern "C"
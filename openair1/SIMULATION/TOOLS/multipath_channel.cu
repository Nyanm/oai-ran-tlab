#include <stdio.h>
#include <cuda_runtime.h>
#include "oai_cuda.h"

// --- CUDA Helper Functions & Kernel (Unchanged) ---
// ... (checkCuda, complex_mul, complex_add) ...
// ... (multipath_channel_kernel_optimized) ...

__device__ __forceinline__ float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

#define MAX_CHANNEL_ELEMENTS (4096) // Increased for safety
__constant__ float2 d_channel_const[MAX_CHANNEL_ELEMENTS];


__global__ void multipath_channel_kernel_optimized(
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx)
{
    extern __shared__ float2 tx_shared[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int ii = blockIdx.y;

    if (i >= num_samples) return;

    float2 rx_tmp = make_float2(0.0f, 0.0f);

    for (int j = 0; j < nb_tx; j++) {
        const int tid = threadIdx.x;
        const int block_start_idx = blockIdx.x * blockDim.x;
        const int shared_mem_size = blockDim.x + channel_length - 1;

        for (int k = tid; k < shared_mem_size; k += blockDim.x) {
            int load_idx = block_start_idx + k - (channel_length - 1);
            if (load_idx >= 0 && load_idx < num_samples) {
                tx_shared[k] = tx_sig[j * num_samples + load_idx];
            } else {
                tx_shared[k] = make_float2(0.0f, 0.0f);
            }
        }
        __syncthreads();

        for (int l = 0; l < channel_length; l++) {
            float2 tx_sample = tx_shared[tid + (channel_length - 1) - l];
            int chan_link_idx = ii + (j * nb_rx);
            float2 chan_weight = d_channel_const[chan_link_idx * channel_length + l];
            rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
        }
        __syncthreads();
    }
    
    rx_sig[ii * num_samples + i] = rx_tmp;
}


// ====================================================================================
// Host Wrapper - Updated to use the new decoupled signature
// ====================================================================================
extern "C" {
void multipath_channel_cuda_fast(
    float **tx_sig_re, float **tx_sig_im,
    float **rx_sig_re, float **rx_sig_im,
    int nb_tx, int nb_rx, int channel_length,
    uint32_t length, uint64_t channel_offset,
    float path_loss,
    float *h_channel_coeffs,
    void *d_tx_sig_void, void *d_rx_sig_void
)
{
    float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float2 *d_rx_sig = (float2*)d_rx_sig_void;

    int num_samples = length - (int)channel_offset;

    float2* h_tx_sig_interleaved = (float2*)malloc(nb_tx * num_samples * sizeof(float2));
    if (!h_tx_sig_interleaved) return;

    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig_interleaved[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }

    cudaMemcpy(d_tx_sig, h_tx_sig_interleaved, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_channel_const, h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2));

    // Use the dynamic block size for the launch configuration
    dim3 threadsPerBlock(512, 1);
    dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);
    size_t sharedMemSize = (threadsPerBlock.x + channel_length - 1) * sizeof(float2);

    multipath_channel_kernel_optimized<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
    
    float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));
    if (!h_rx_sig) { free(h_tx_sig_interleaved); return; }

    cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost);
    
    for (int ii = 0; ii < nb_rx; ii++) {
        for (int i = 0; i < num_samples; i++) {
            float2 result = h_rx_sig[ii * num_samples + i];
            rx_sig_re[ii][i + channel_offset] = result.x * path_loss;
            rx_sig_im[ii][i + channel_offset] = result.y * path_loss;
        }
    }

    free(h_tx_sig_interleaved);
    free(h_rx_sig);
}

} // extern "C"
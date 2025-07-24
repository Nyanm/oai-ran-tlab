/**
 * @file multipath_channel.cu
 * @brief CUDA implementation of the multipath_channel function.
 * Final version with a fixed block size of 512 and a grid-stride loop
 * to increase arithmetic intensity per thread.
 * @author Nika Ghaderi & Gemini
 * @date July 24, 2025
 */

#include <stdio.h>
#include <cuda_runtime.h>
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

__device__ __forceinline__ float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// --- Constant Memory ---
#define MAX_CHANNEL_ELEMENTS (4096)
__constant__ float2 d_channel_const[MAX_CHANNEL_ELEMENTS];


// ====================================================================================
// KERNEL with Grid-Stride Loop
// ====================================================================================
__global__ void multipath_channel_kernel_gridstride(
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx)
{
    extern __shared__ float2 tx_shared[];
    
    // Each thread will process multiple output samples using a grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_samples;
         i += gridDim.x * blockDim.x)
    {
        const int ii = blockIdx.y; // Receiver antenna index
        float2 rx_tmp = make_float2(0.0f, 0.0f);

        for (int j = 0; j < nb_tx; j++) {
            // --- Cooperative Loading into Shared Memory ---
            // This part is more complex with a grid-stride loop, as we can't
            // assume a simple block-per-tile mapping. We load data relative
            // to the current sample 'i'.
            const int tid = threadIdx.x;
            
            // Each thread loads one element needed for its convolution window
            int load_offset = i - 1;
            if (tid < channel_length) {
                 int load_idx = i + tid - (channel_length - 1);
                 if (load_idx >= 0 && load_idx < num_samples) {
                    tx_shared[tid] = tx_sig[j * num_samples + load_idx];
                 } else {
                    tx_shared[tid] = make_float2(0.0f, 0.0f);
                 }
            }
            __syncthreads();


            // --- Convolution using Shared Memory ---
            // This loop is now much simpler as shared memory contains the exact window
            if (tid < channel_length) {
                float2 tx_sample = tx_shared[channel_length - 1 - tid];
                int chan_link_idx = ii + (j * nb_rx);
                float2 chan_weight = d_channel_const[chan_link_idx * channel_length + tid];
                rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
            }
        } // End loop over TX antennas
        
        // Atomically add partial results if multiple threads contributed to rx_tmp
        // For this simplified version, we assume one thread computes the full result for sample 'i'.
        // A more complex kernel might sum partial results.
        
        rx_sig[ii * num_samples + i] = rx_tmp;
    } // End grid-stride loop
}


// ====================================================================================
// Host Wrapper - Reverted to the simpler, faster version
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

    // The host no longer needs to interleave the data.
    // It creates a simple float2 array for each antenna.
    float2* h_tx_sig = (float2*)malloc(nb_tx * num_samples * sizeof(float2));
    if (!h_tx_sig) return;

    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }

    CHECK_CUDA(cudaMemcpy(d_tx_sig, h_tx_sig, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_channel_const, h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2)));

    // Use the fixed, optimal block size of 512
    dim3 threadsPerBlock(512, 1);
    
    // We can launch a smaller grid now, letting the grid-stride loop handle all samples.
    // Let's aim for a grid that's large enough to saturate the SMs.
    // A good starting point is 2x the number of SMs.
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    dim3 numBlocks(num_sms * 2, nb_rx);

    size_t sharedMemSize = channel_length * sizeof(float2); // Simplified shared mem calculation

    // Call the new grid-stride kernel
    multipath_channel_kernel_gridstride<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
    
    float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));
    if (!h_rx_sig) { free(h_tx_sig); return; }

    CHECK_CUDA(cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost));
    
    for (int ii = 0; ii < nb_rx; ii++) {
        for (int i = 0; i < num_samples; i++) {
            float2 result = h_rx_sig[ii * num_samples + i];
            rx_sig_re[ii][i + channel_offset] = result.x * path_loss;
            rx_sig_im[ii][i + channel_offset] = result.y * path_loss;
        }
    }

    free(h_tx_sig);
    free(h_rx_sig);
}

} // extern "C"

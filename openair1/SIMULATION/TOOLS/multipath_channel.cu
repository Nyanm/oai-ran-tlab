/**
 * @file multipath_channel.cu
 * @brief CUDA implementation of the multipath_channel function.
 * This version correctly implements a grid-stride loop to increase
 * arithmetic intensity and leverage the H100's compute power.
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
// NEW KERNEL with a Correctly Implemented Grid-Stride Loop
// This version forgoes shared memory to achieve higher arithmetic intensity,
// relying on the H100's L1/L2 cache.
// ====================================================================================
__global__ void multipath_channel_kernel_gridstride(
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx)
{
    // Calculate the total number of threads in the grid
    const int grid_stride = gridDim.x * blockDim.x;

    // Grid-stride loop: each thread processes multiple output samples
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_samples;
         i += grid_stride)
    {
        const int ii = blockIdx.y; // Current receiver antenna
        float2 rx_tmp = make_float2(0.0f, 0.0f);

        // Loop over each transmit antenna
        for (int j = 0; j < nb_tx; j++) {
            
            // This thread is responsible for its own convolution.
            // It reads directly from global memory, relying on the L1/L2 cache.
            for (int l = 0; l < channel_length; l++) {
                int load_idx = i - l;

                if (load_idx >= 0) {
                    // Direct global memory access for the TX signal
                    float2 tx_sample = tx_sig[j * num_samples + load_idx];

                    // Constant memory access for the channel coefficients
                    int chan_link_idx = ii + (j * nb_rx);
                    float2 chan_weight = d_channel_const[chan_link_idx * channel_length + l];
                    
                    // Accumulate the result
                    rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
                }
            }
        }
        // Write the final result for this sample to global memory
        rx_sig[ii * num_samples + i] = rx_tmp;
    }
}


// ====================================================================================
// Host Wrapper - Launches the new grid-stride kernel
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

    // Prepare host data in a simple [antenna][sample] layout
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
    
    // For a grid-stride loop, we don't need a block for every tile.
    // We launch a smaller grid, ensuring enough blocks to keep the GPU busy.
    // A good heuristic is to launch 2-4x the number of SMs.
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    dim3 numBlocks(num_sms * 4, nb_rx);

    // Call the new grid-stride kernel. Note that it does not use shared memory.
    multipath_channel_kernel_gridstride<<<numBlocks, threadsPerBlock, 0>>>(
        d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
    
    float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));
    if (!h_rx_sig) { free(h_tx_sig); return; }

    CHECK_CUDA(cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost));
    
    // De-interleave results and apply path loss
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

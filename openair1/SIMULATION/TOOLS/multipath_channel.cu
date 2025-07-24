/**
 * @file multipath_channel.cu
 * @brief CUDA implementation of the multipath_channel function.
 * @author Nika Ghaderi & Gemini
 * @date July 24, 2025
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "oai_cuda.h" 

// ====================================================================================
// Constant Memory for Channel Coefficients
// ====================================================================================
#define MAX_CHANNEL_ELEMENTS (1024)
__constant__ float2 d_channel_const[MAX_CHANNEL_ELEMENTS];


// ====================================================================================
// CUDA Helper Functions
// ====================================================================================

static void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
        exit(1);
    }
}
#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

__device__ __forceinline__ float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// ====================================================================================
// FINAL OPTIMIZED KERNEL with Correct Shared Memory Halo Handling
// ====================================================================================
__global__ void multipath_channel_kernel_optimized(
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx)
{
    // Dynamically allocated shared memory. The size is passed during kernel launch.
    extern __shared__ float2 tx_shared[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Global time sample index
    const int ii = blockIdx.y;                          // Receiver antenna index

    if (i >= num_samples) return;

    float2 rx_tmp = make_float2(0.0f, 0.0f);

    for (int j = 0; j < nb_tx; j++) {
        // --- Step 1: Cooperative Loading into Shared Memory with Halo ---
        const int tid = threadIdx.x;
        const int block_start_idx = blockIdx.x * blockDim.x;
        const int shared_mem_size = blockDim.x + channel_length - 1;

        for (int k = tid; k < shared_mem_size; k += blockDim.x) {
            int load_idx = block_start_idx + k - (channel_length - 1);
            
            if (load_idx >= 0 && load_idx < num_samples) {
                tx_shared[k] = tx_sig[j * num_samples + load_idx];
            } else {
                tx_shared[k] = make_float2(0.0f, 0.0f); // Zero-pad for samples before t=0
            }
        }
        __syncthreads(); // Wait for all threads to finish loading

        // --- Step 2: Convolution using Shared and Constant Memory ---
        for (int l = 0; l < channel_length; l++) {
            float2 tx_sample = tx_shared[tid + (channel_length - 1) - l];
            int chan_link_idx = ii + (j * nb_rx);
            float2 chan_weight = d_channel_const[chan_link_idx * channel_length + l];
            rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
        }
        __syncthreads(); // Wait before the next 'j' loop iteration
    }
    
    // Write the final result to global memory
    rx_sig[ii * num_samples + i] = rx_tmp;
}


// ====================================================================================
// Host Wrapper - Updated to the new decoupled signature
// ====================================================================================
extern "C" {
void multipath_channel_cuda_fast(
    // Input Signal Buffers
    float **tx_sig_re,
    float **tx_sig_im,
    // Output Signal Buffers
    float **rx_sig_re,
    float **rx_sig_im,
    // Simulation & Channel Parameters
    int nb_tx,
    int nb_rx,
    int channel_length,
    uint32_t length,
    uint64_t channel_offset,
    float path_loss,
    // Host pointer to the flattened channel coefficients
    float *h_channel_coeffs,
    // Pre-allocated GPU memory
    void *d_tx_sig_void,
    void *d_rx_sig_void)
{
    // Cast void pointers to their actual types
    float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float2 *d_rx_sig = (float2*)d_rx_sig_void;

    int num_samples = length - (int)channel_offset;

    // Allocate temporary host memory for interleaving the signal
    float2* h_tx_sig_interleaved = (float2*)malloc(nb_tx * num_samples * sizeof(float2));
    if (!h_tx_sig_interleaved) { /* handle error */ return; }

    // Interleave the real and imaginary parts of the TX signal for the GPU
    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig_interleaved[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }

    // --- GPU Operations ---
    CHECK_CUDA(cudaMemcpy(d_tx_sig, h_tx_sig_interleaved, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_channel_const, h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2)));

    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);
    size_t sharedMemSize = (threadsPerBlock.x + channel_length - 1) * sizeof(float2);

    // Launch Kernel
    multipath_channel_kernel_optimized<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
    
    // Allocate host memory for the result and copy it back
    float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));
    if (!h_rx_sig) { /* handle error */ free(h_tx_sig_interleaved); return; }

    CHECK_CUDA(cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost));
    
    // De-interleave the result and apply path loss
    for (int ii = 0; ii < nb_rx; ii++) {
        for (int i = 0; i < num_samples; i++) {
            float2 result = h_rx_sig[ii * num_samples + i];
            rx_sig_re[ii][i + channel_offset] = result.x * path_loss;
            rx_sig_im[ii][i + channel_offset] = result.y * path_loss;
        }
    }

    // Free temporary host memory
    free(h_tx_sig_interleaved);
    free(h_rx_sig);
}

} // extern "C"
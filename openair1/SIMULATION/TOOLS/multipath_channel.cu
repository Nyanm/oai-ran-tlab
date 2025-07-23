#include <stdio.h>
#include <cuda_runtime.h>

#include "PHY/TOOLS/tools_defs.h"

extern "C" {
  #include "SIMULATION/TOOLS/sim.h"
}

#include "SIMULATION/TOOLS/oai_cuda.h"

// ====================================================================================
// Constant Memory for Channel Coefficients
// ====================================================================================
// Using constant memory for the channel taps is ideal because it's cached
// and optimized for when all threads in a warp read the same address.
#define MAX_CHANNEL_ELEMENTS (1024) // Safe upper bound for 4x4 MIMO with 64 taps
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
// OPTIMIZED KERNEL with Shared and Constant Memory
// ====================================================================================
__global__ void multipath_channel_kernel_optimized(
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx)
{
    // Statically allocated shared memory. Size must be known at compile time.
    // We size it for a block of 256 threads and a max channel length of 64.
    // This is much faster than dynamic shared memory.
    __shared__ float2 tx_shared[256 + 64 - 1];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int ii = blockIdx.y;

    if (i >= num_samples) return;

    float2 rx_tmp = make_float2(0.0f, 0.0f);

    for (int j = 0; j < nb_tx; j++) {
        // --- Cooperative loading into Shared Memory ---
        const int tid = threadIdx.x;
        const int block_start_idx = blockIdx.x * blockDim.x;
        
        // Each thread loads a piece of the input signal into our shared tile
        int load_idx = block_start_idx + tid;
        if (load_idx < num_samples) {
            tx_shared[tid] = tx_sig[j * num_samples + load_idx];
        }

        // The last (channel_length - 1) threads of the block load the "halo"
        // for the *next* block to use. This is a common optimization pattern.
        if (tid < channel_length - 1) {
            int halo_load_idx = block_start_idx + blockDim.x + tid;
            if (halo_load_idx < num_samples) {
                 tx_shared[blockDim.x + tid] = tx_sig[j * num_samples + halo_load_idx];
            }
        }
        __syncthreads();

        // --- Convolution using fast Shared and Constant memory ---
        for (int l = 0; l < channel_length; l++) {
            int tx_index = i - l;
            if (tx_index >= 0) {
                float2 tx_sample = tx_shared[tid - l];
                int chan_link_idx = ii + (j * nb_rx);
                float2 chan_weight = d_channel_const[chan_link_idx * channel_length + l];
                rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
            }
        }
        __syncthreads();
    }
    
    rx_sig[ii * num_samples + i] = rx_tmp;
}


// ====================================================================================
// Host Wrapper
// ====================================================================================
extern "C" {
    void multipath_channel_cuda_fast(channel_desc_t *desc,
                                     float **tx_sig_re,
                                     float **tx_sig_im,
                                     float **rx_sig_re,
                                     float **rx_sig_im,
                                     uint32_t length,
                                     void *d_tx_sig_void,
                                     void *d_channel_void, // Unused
                                     void *d_rx_sig_void)
    {
        random_channel(desc, 0);

        int nb_tx = desc->nb_tx;
        int nb_rx = desc->nb_rx;
        int channel_length = desc->channel_length;
        uint64_t dd = desc->channel_offset;
        int num_samples = length - (int)dd;
        float path_loss = (float)pow(10, desc->path_loss_dB / 20.0);

        float2 *d_tx_sig = (float2*)d_tx_sig_void;
        float2 *d_rx_sig = (float2*)d_rx_sig_void;

        float2* h_tx_sig = (float2*)malloc(nb_tx * num_samples * sizeof(float2));
        float2* h_channel = (float2*)malloc(nb_tx * nb_rx * channel_length * sizeof(float2));
        float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));

        for (int j = 0; j < nb_tx; j++) {
            for (int i = 0; i < num_samples; i++) {
                h_tx_sig[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
            }
        }

        for (int link = 0; link < nb_tx * nb_rx; link++) {
            for (int l = 0; l < channel_length; l++) {
                h_channel[link * channel_length + l] = make_float2((float)desc->ch[link][l].r, (float)desc->ch[link][l].i);
            }
        }

        CHECK_CUDA(cudaMemcpy(d_tx_sig, h_tx_sig, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyToSymbol(d_channel_const, h_channel, nb_tx * nb_rx * channel_length * sizeof(float2)));

        dim3 threadsPerBlock(256, 1);
        dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);

        // Call the optimized kernel. No dynamic shared memory needed.
        multipath_channel_kernel_optimized<<<numBlocks, threadsPerBlock>>>(
            d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
        
        CHECK_CUDA(cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost));
        
        for (int ii = 0; ii < nb_rx; ii++) {
            for (int i = 0; i < num_samples; i++) {
                float2 result = h_rx_sig[ii * num_samples + i];
                rx_sig_re[ii][i + dd] = result.x * path_loss;
                rx_sig_im[ii][i + dd] = result.y * path_loss;
            }
        }

        free(h_tx_sig);
        free(h_channel);
        free(h_rx_sig);
    }
} // extern "C"

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

__device__ __forceinline__ float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}



__global__ void multipath_channel_kernel(
    const float2* __restrict__ d_channel_coeffs,
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx,
    float path_loss)
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
            float2 chan_weight = d_channel_coeffs[chan_link_idx * channel_length + l];
            rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
        }
        __syncthreads();
    }
    
    rx_sig[ii * num_samples + i].x = rx_tmp.x * path_loss;
    rx_sig[ii * num_samples + i].y = rx_tmp.y * path_loss;
}


__global__ void multipath_channel_kernel_batched(
    const float2* __restrict__ d_channel_coeffs,
    const float2* __restrict__ tx_sig,
    float2* __restrict__ rx_sig,
    int num_samples,
    int channel_length,
    int nb_tx,
    int nb_rx,
    const float* __restrict__ path_loss_batch)
{
    extern __shared__ float2 tx_shared[];
    

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int ii = blockIdx.y;                         
    const int c = blockIdx.z;                         

    if (i >= num_samples) return;

    float2 rx_tmp = make_float2(0.0f, 0.0f);
    const float path_loss = path_loss_batch[c]; 

    const int channel_tx_offset = c * nb_tx * num_samples;
    const int channel_rx_offset = c * nb_rx * num_samples;

    for (int j = 0; j < nb_tx; j++) {
        const int tid = threadIdx.x;
        const int block_start_idx = blockIdx.x * blockDim.x;
        const int shared_mem_size = blockDim.x + channel_length - 1;

        for (int k = tid; k < shared_mem_size; k += blockDim.x) {
            int load_idx = block_start_idx + k - (channel_length - 1);
            if (load_idx >= 0 && load_idx < num_samples) {
                tx_shared[k] = tx_sig[channel_tx_offset + j * num_samples + load_idx];
            } else {
                tx_shared[k] = make_float2(0.0f, 0.0f);
            }
        }
        __syncthreads();

        for (int l = 0; l < channel_length; l++) {
            float2 tx_sample = tx_shared[tid + (channel_length - 1) - l];
            int chan_link_idx = (c * nb_tx * nb_rx) + (ii + j * nb_rx);
            float2 chan_weight = d_channel_coeffs[chan_link_idx * channel_length + l];
            rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
        }
        __syncthreads();
    }
    
    rx_sig[channel_rx_offset + ii * num_samples + i].x = rx_tmp.x * path_loss;
    rx_sig[channel_rx_offset + ii * num_samples + i].y = rx_tmp.y * path_loss;
}



extern "C" {

void multipath_channel_cuda(
    float **tx_sig_re, float **tx_sig_im,
    float **rx_sig_re, float **rx_sig_im,
    int nb_tx, int nb_rx, int channel_length,
    uint32_t length, uint64_t channel_offset,
    float path_loss,
    float *h_channel_coeffs,
    void *d_tx_sig_void, void *d_rx_sig_void,
    void *d_channel_coeffs_void,
    void *h_tx_sig_pinned_void 
)
{
    float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float2 *d_rx_sig = (float2*)d_rx_sig_void;
    float2 *d_channel_coeffs = (float2*)d_channel_coeffs_void;
    int num_samples = length - (int)channel_offset;
    float2* kernel_input_ptr;

    #if defined(USE_UNIFIED_MEMORY)
            for (int j = 0; j < nb_tx; j++) {
                for (int i = 0; i < num_samples; i++) {
                    d_tx_sig[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                }
            }
            kernel_input_ptr = d_tx_sig;
    #elif defined(USE_ATS_MEMORY)
            float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
            for (int j = 0; j < nb_tx; j++) {
                for (int i = 0; i < num_samples; i++) {
                    h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                }
            }
            kernel_input_ptr = h_tx_sig_pinned; 
    #else // EXPLICIT COPY
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

    dim3 threadsPerBlock(512, 1);
    dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);
    size_t sharedMemSize = (threadsPerBlock.x + channel_length - 1) * sizeof(float2);
    multipath_channel_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_channel_coeffs, kernel_input_ptr, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx, path_loss);
         
    #if defined(USE_UNIFIED_MEMORY)
            CHECK_CUDA( cudaDeviceSynchronize() );
            for (int ii = 0; ii < nb_rx; ii++) {
                for (int i = 0; i < num_samples; i++) {
                    float2 result = d_rx_sig[ii * num_samples + i];
                    rx_sig_re[ii][i + channel_offset] = result.x;
                    rx_sig_im[ii][i + channel_offset] = result.y;
                }
            }
    #else
            CHECK_CUDA( cudaDeviceSynchronize() ); 
            float2* h_rx_sig = (float2*)malloc(nb_rx * num_samples * sizeof(float2));
            CHECK_CUDA( cudaMemcpy(h_rx_sig, d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemcpyDeviceToHost) );
            for (int ii = 0; ii < nb_rx; ii++) {
                for (int i = 0; i < num_samples; i++) {
                    float2 result = h_rx_sig[ii * num_samples + i];
                    rx_sig_re[ii][i + channel_offset] = result.x;
                    rx_sig_im[ii][i + channel_offset] = result.y;
                }
            }
            free(h_rx_sig);
    #endif
}

} // extern "C"
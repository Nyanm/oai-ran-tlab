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

__global__ void interleave_output_kernel(const float* __restrict__ rx_re,
                                         const float* __restrict__ rx_im,
                                         float2* __restrict__ output_interleaved,
                                         int num_total_samples)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_total_samples) {
        output_interleaved[i].x = rx_re[i];
        output_interleaved[i].y = rx_im[i];
    }
}

__global__ void multipath_channel_kernel(
    const float2* __restrict__ d_channel_coeffs,
    // const float2* __restrict__ tx_sig,
    const float* __restrict__ tx_sig,
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
                // tx_shared[k] = tx_sig[j * num_samples + load_idx];
                // --- CHANGED: Read two floats and construct a float2 ---
                int interleaved_idx = 2 * (j * num_samples + load_idx);
                tx_shared[k] = make_float2(tx_sig[interleaved_idx], tx_sig[interleaved_idx + 1]);
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
    
    rx_sig[ii * num_samples + i].x = rx_tmp.x;
    rx_sig[ii * num_samples + i].y = rx_tmp.y;
}


__global__ void multipath_channel_kernel_batched(
    const float2* __restrict__ d_channel_coeffs,
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
    const int c = blockIdx.z;                         

    if (i >= num_samples) return;

    float2 rx_tmp = make_float2(0.0f, 0.0f);

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
    
    rx_sig[channel_rx_offset + ii * num_samples + i].x = rx_tmp.x;
    rx_sig[channel_rx_offset + ii * num_samples + i].y = rx_tmp.y;
}



extern "C" {

void multipath_channel_cuda(
    // float **tx_sig_re, float **tx_sig_im,
    float **tx_sig_interleaved,
    float **rx_sig_re, float **rx_sig_im,
    int nb_tx, int nb_rx, int channel_length,
    uint32_t length, uint64_t channel_offset,
    float *h_channel_coeffs,
    void *d_tx_sig_void, void *d_rx_sig_void,
    void *d_channel_coeffs_void,
    void *h_tx_sig_pinned_void 
)
{
    // float2 *d_tx_sig = (float2*)d_tx_sig_void;
    float *d_tx_sig = (float*)d_tx_sig_void;
    float2 *d_rx_sig = (float2*)d_rx_sig_void;
    float2 *d_channel_coeffs = (float2*)d_channel_coeffs_void;
    int num_samples = length - (int)channel_offset;
    // float2* kernel_input_ptr;
    float* kernel_input_ptr;

    #if defined(USE_UNIFIED_MEMORY)
            // for (int j = 0; j < nb_tx; j++) {
            //     for (int i = 0; i < num_samples; i++) {
            //         d_tx_sig[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
            //     }
            // }
            // For UM, we can just copy the host data into the managed buffer
            for (int j = 0; j < nb_tx; j++) {
                memcpy(d_tx_sig + j * num_samples * 2, tx_sig_interleaved[j], num_samples * 2 * sizeof(float));
            }
            kernel_input_ptr = d_tx_sig;
    #elif defined(USE_ATS_MEMORY)
            // float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
            float* h_tx_sig_pinned = (float*)h_tx_sig_pinned_void;
            for (int j = 0; j < nb_tx; j++) {
                // for (int i = 0; i < num_samples; i++) {
                //     h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                // }
                memcpy(h_tx_sig_pinned + j * num_samples * 2, tx_sig_interleaved[j], num_samples * 2 * sizeof(float));
            }
            kernel_input_ptr = h_tx_sig_pinned; 
    #else // EXPLICIT COPY
            // float2* h_tx_sig_pinned = (float2*)h_tx_sig_pinned_void;
            float* h_tx_sig_pinned = (float*)h_tx_sig_pinned_void;
            for (int j = 0; j < nb_tx; j++) {
                // for (int i = 0; i < num_samples; i++) {
                //     h_tx_sig_pinned[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
                // }
            // }
            // CHECK_CUDA( cudaMemcpy(d_tx_sig, h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice) );
            memcpy(h_tx_sig_pinned + j * num_samples * 2, tx_sig_interleaved[j], num_samples * 2 * sizeof(float));
            }
            CHECK_CUDA( cudaMemcpy(d_tx_sig, h_tx_sig_pinned, nb_tx * num_samples * 2 * sizeof(float), cudaMemcpyHostToDevice) );
            
            kernel_input_ptr = d_tx_sig;
    #endif

    size_t channel_size_bytes = nb_tx * nb_rx * channel_length * sizeof(float2);
    CHECK_CUDA( cudaMemcpy(d_channel_coeffs, h_channel_coeffs, channel_size_bytes, cudaMemcpyHostToDevice) );

    dim3 threadsPerBlock(512, 1);
    dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);
    size_t sharedMemSize = (threadsPerBlock.x + channel_length - 1) * sizeof(float2);
    multipath_channel_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_channel_coeffs, kernel_input_ptr, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
         
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


void interleave_channel_output_cuda(float **rx_sig_re,
                                    float **rx_sig_im,
                                    void **output_interleaved_void,
                                    int nb_rx,
                                    int num_samples)
{
    // Cast back to the proper type inside the implementation
    float2 **output_interleaved = (float2**)output_interleaved_void;
    
    int num_total_samples = nb_rx * num_samples;
    size_t total_bytes_per_plane = num_total_samples * sizeof(float);
    size_t total_bytes_interleaved = num_total_samples * sizeof(float2);

    float *h_re_flat = (float*)malloc(total_bytes_per_plane);
    float *h_im_flat = (float*)malloc(total_bytes_per_plane);
    float2 *h_out_flat = (float2*)malloc(total_bytes_interleaved);

    for (int i = 0; i < nb_rx; i++) {
        memcpy(h_re_flat + i * num_samples, rx_sig_re[i], num_samples * sizeof(float));
        memcpy(h_im_flat + i * num_samples, rx_sig_im[i], num_samples * sizeof(float));
    }
    float *d_re, *d_im;
    float2 *d_out;
    CHECK_CUDA( cudaMalloc((void**)&d_re, total_bytes_per_plane) );
    CHECK_CUDA( cudaMalloc((void**)&d_im, total_bytes_per_plane) );
    CHECK_CUDA( cudaMalloc((void**)&d_out, total_bytes_interleaved) );
    CHECK_CUDA( cudaMemcpy(d_re, h_re_flat, total_bytes_per_plane, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_im, h_im_flat, total_bytes_per_plane, cudaMemcpyHostToDevice) );

    int threadsPerBlock = 512;
    int blocksPerGrid = (num_total_samples + threadsPerBlock - 1) / threadsPerBlock;
    interleave_output_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_re, d_im, d_out, num_total_samples);

    CHECK_CUDA( cudaMemcpy(h_out_flat, d_out, total_bytes_interleaved, cudaMemcpyDeviceToHost) );
    
    for (int i = 0; i < nb_rx; i++) {
        memcpy(output_interleaved[i], h_out_flat + i * num_samples, num_samples * sizeof(float2));
    }
    free(h_re_flat);
    free(h_im_flat);
    free(h_out_flat);
    CHECK_CUDA( cudaFree(d_re) );
    CHECK_CUDA( cudaFree(d_im) );
    CHECK_CUDA( cudaFree(d_out) );
}

} // extern "C"
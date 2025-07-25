#include <stdio.h>
#include <stdlib.h> // For atexit()
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


// --- Kernel (The proven, faster version) ---
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
// State Management for CUDA Graph
// ====================================================================================
static bool is_graph_initialized = false;
static cudaGraph_t graph;
static cudaGraphExec_t graph_exec;

static int graph_nb_tx = 0;
static int graph_nb_rx = 0;
static int graph_channel_length = 0;
static int graph_num_samples = 0;

void cleanup_cuda_graph() {
    if (is_graph_initialized) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        is_graph_initialized = false;
        printf("\n[CUDA] Graph resources cleaned up.\n");
    }
}

// ====================================================================================
// Host Wrapper with Corrected CUDA Graph Implementation
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

    float2* h_tx_sig = (float2*)malloc(nb_tx * num_samples * sizeof(float2));
    if (!h_tx_sig) return;

    for (int j = 0; j < nb_tx; j++) {
        for (int i = 0; i < num_samples; i++) {
            h_tx_sig[j * num_samples + i] = make_float2(tx_sig_re[j][i], tx_sig_im[j][i]);
        }
    }

    bool need_recapture = !is_graph_initialized ||
                          nb_tx != graph_nb_tx ||
                          nb_rx != graph_nb_rx ||
                          channel_length != graph_channel_length ||
                          num_samples != graph_num_samples;

    if (need_recapture) {
        if (is_graph_initialized) {
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
        }

        printf("[CUDA] Capturing CUDA graph for config (MIMO: %dx%d, Samples: %d)...\n", nb_tx, nb_rx, num_samples);

        if (!is_graph_initialized) {
            atexit(cleanup_cuda_graph);
        }

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // --- CORRECTED: Only record the GPU-side kernel launch in the graph ---
        dim3 threadsPerBlock(512, 1);
        dim3 numBlocks((num_samples + threadsPerBlock.x - 1) / threadsPerBlock.x, nb_rx);
        size_t sharedMemSize = (threadsPerBlock.x + channel_length - 1) * sizeof(float2);

        multipath_channel_kernel_optimized<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
            d_tx_sig, d_rx_sig, num_samples, channel_length, nb_tx, nb_rx);
        
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
        CHECK_CUDA(cudaStreamDestroy(stream));

        graph_nb_tx = nb_tx;
        graph_nb_rx = nb_rx;
        graph_channel_length = channel_length;
        graph_num_samples = num_samples;
        is_graph_initialized = true;
    }

    // --- Data Transfer (happens on every call, outside the graph) ---
    CHECK_CUDA(cudaMemcpy(d_tx_sig, h_tx_sig, nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_channel_const, h_channel_coeffs, nb_tx * nb_rx * channel_length * sizeof(float2)));
    
    // --- Graph Replay ---
    CHECK_CUDA(cudaGraphLaunch(graph_exec, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

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

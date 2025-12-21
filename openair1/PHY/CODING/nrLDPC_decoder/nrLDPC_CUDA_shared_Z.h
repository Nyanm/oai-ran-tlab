#ifndef __NR_LDPC_CUDA_SHARED_Z__H__
#define __NR_LDPC_CUDA_SHARED_Z__H__
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(call) ErrorCheck((call), __FILE__, __LINE__)
/**
 * @brief Checks CUDA error status and prints detailed diagnostic info if an error occurred.
 *
 * @param error_code The CUDA error code returned from a CUDA runtime API call.
 * @param filename   The name of the source file where the error occurred.
 * @param lineNumber The line number in the source file where the error occurred.
 * @return cudaError_t Returns the same error code passed in, for optional further handling.
 */
inline cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber)
{
  if (error_code != cudaSuccess) {
    printf("[CUDA ERROR] %s (%d): %s\nOccurred in file: %s at line %d\n",
           cudaGetErrorName(error_code),
           error_code,
           cudaGetErrorString(error_code),
           filename,
           lineNumber);
  }
  return error_code;
}

#define BG1_ROW 46      // number of rows in base graph (BG1)
#define BG1_COL 68      // number of cols in base graph (BG1)

#define BG2_ROW 42      // number of rows in base graph (BG2)
#define BG2_COL 52      // number of cols in base graph (BG2)

#define BG1_MAX_NNZ 316  // Maximum number of non-zero elements in BG1
#define BG1_MAX_ROW_DEGREE 19  // Maximum row degree in BG1

#define MAX_Z 384     // Maximum lift size in 5G NR

#define ALPHA 0.75f    // Scaling factor for Min-Sum algorithm

// Constants for quantization

#define SIMD_MAX_LLR 0x7F7F7F7FU
#define SIMD_WIDTH 4


#define BG1_MAX_CW_LEN 26112    // Maximum codeword length for BG1
#define BG1_MAX_INFO_LEN 8448
#define BG1_R13_Z384_Kb 22


#define MAX_STREAMS 6 // Maximum number of CUDA streams
#define GROUPS_PER_STREAM 4 // Number of groups per stream
#define CWS_PER_STREAM (GROUPS_PER_STREAM * SIMD_WIDTH) // Number of codewords per stream
#define CWS_PER_BATCH (MAX_STREAMS * CWS_PER_STREAM) // Number of codewords per batch


struct CompressedH_cn {
    int* row_ptr;     
    int* col_idx;     
    int* shift_cn;    
    int nnz;          
};

struct CompressedH_vn {
    int* col_ptr;    
    int* row_idx;     
    int* shift_vn;    
    int* csc2csr;     
    int nnz; 
};

struct ldpc_params {
    int rate;                     // Code rate
    int bg_index;                   // Index of the 5G NR base graph (1 or 2)
    int Z;                          // Lifting factor
    int N;                          // Codeword length (N = nb * Z)
    int K;                          // Number of information bits (K = (nb - mb) * Z)
    int Kb;                         // Number of information bits in the base graph (Kb = nb - mb)
    int mb;                         // Number of rows in the base graph (i.e., number of parity check equations)
    int nb;                         // Number of columns in the base graph (i.e., total variable nodes in base graph)
    int* H;                         // Pointer to the parity-check matrix H 
    struct CompressedH_cn* compH_cn; // Compressed representation of H for check-node (CN) processing (CSR)
    struct CompressedH_vn* compH_vn; // Compressed representation of H for variable-node (VN) processing (CSC)
    int n_iterations;               // Maximum number of decoding iterations
};

struct kernel_compH_cn {
    int row_ptr[BG1_ROW + 1];       // Row pointer array (CSR format)

    int col_idx[BG1_MAX_NNZ];       // Column indices
    int shift_cn[BG1_MAX_NNZ];      // Shift values for check nodes
    int max_row_degree;             // Maximum row degree (max number of variable nodes connected to any check node)
    int nnz;                        // Number of non-zero elements
};

struct kernel_compH_vn {
    int col_ptr[BG1_COL + 1];       // Column pointer array (CSC format)

    int row_idx[BG1_MAX_NNZ];       // Row indices
    int shift_vn[BG1_MAX_NNZ];      // Shift values for variable nodes
    int csc2csr[BG1_MAX_NNZ];       // Mapping from CSC to CSR indexing
    int max_col_degree;             // Maximum column degree (max number of check nodes connected to any variable node)
    int nnz;                        // Number of non-zero elements
};


struct host_memory {
    // Large pinned (page-locked) host memory blocks
    int8_t* h_big_pinned_llr;        // Pinned buffer for LLRs
    uint8_t* h_big_hard_bits;        // Pinned buffer for hard bits

    // Arrays of pointers
    int8_t** h_pinned_llr;           // Pointers to initial LLR values in pinned memory
    uint8_t** h_pinned_hard;         // Pointers to hard decision results in pinned memory
};

struct device_memory {
    // Regular device memory
    int8_t* d_big_init_llr;          // Large buffer for initial LLRs on device
    uint32_t* d_big_app;             // Large buffer for APP values on device
    int8_t* d_big_app_reordered;     // Large buffer for reordered APP values on device
    uint32_t* d_big_c2v;             // Large buffer for check-to-variable messages on device
    uint32_t* d_big_delta_c2v;       // Large buffer for delta (min-sum correction) in C2V messages on device
    uint8_t* d_big_hard_bits;        // Large buffer for hard decision bits on device

    // Arrays of device pointers
    int8_t** d_init_llr;             // Pointers to initial LLR values on device
    uint32_t** d_app;                // Pointers to APP values on device
    int8_t** d_app_reordered;        // Pointers to reordered APP values on device
    uint32_t** d_c2v;                // Pointers to check-to-variable (C2V) messages on device
    uint32_t** d_delta_c2v;          // Pointers to delta C2V messages on device
    uint8_t** d_hard_bits;           // Pointers to hard decision results on device
};


struct cuda_grid{
    dim3 order_grid;    // Order kernel grid
    dim3 order_block;   // Order kernel block
    dim3 reorder_grid; // Reorder kernel grid
    dim3 reorder_block; // Reorder kernel block
    dim3 cnp_grid;      // CNP kernel grid
    dim3 cnp_block;     // CNP kernel block
    dim3 vnp_grid;      // VNP kernel grid
    dim3 vnp_block;     // VNP kernel block
    dim3 hard_grid;    // Hard decision kernel grid
    dim3 hard_block;   // Hard decision kernel block
    size_t cnp_shared_size; // Shared memory size for CNP kernel
};

void ldpc_decoder_cuda_init();
void ldpc_decoder_cuda_free();


// Global variables
extern struct ldpc_params *params;

// Create Streams and Graphs, and execute graphs
extern cudaStream_t* cudaStreams;
extern cudaGraph_t* cudaGraphs;
extern cudaGraphExec_t* cudaGraphExecs;

// Define host and device memory
extern struct host_memory *host_mem;
extern struct device_memory *dev_mem;

#ifdef __cplusplus
}
#endif

#endif

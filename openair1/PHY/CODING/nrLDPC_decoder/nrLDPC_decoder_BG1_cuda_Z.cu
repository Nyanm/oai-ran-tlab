#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_CUDA_shared_Z.h"

// Global variables
struct ldpc_params *params = NULL;

// Create Streams and Graphs, and execute graphs
cudaStream_t* cudaStreams = NULL;
cudaGraph_t* cudaGraphs_R13 = NULL;
cudaGraph_t* cudaGraphs_R23 = NULL;
cudaGraphExec_t* cudaGraphExecs_R13 = NULL;
cudaGraphExec_t* cudaGraphExecs_R23 = NULL;

// Define host and device memory
struct host_memory *host_mem = NULL;
struct device_memory *dev_mem = NULL;

__device__ __constant__ kernel_compH_cn d_compH_cn_r13;
__device__ __constant__ kernel_compH_vn d_compH_vn_r13;

__device__ __constant__ kernel_compH_cn d_compH_cn_r23;
__device__ __constant__ kernel_compH_vn d_compH_vn_r23;


#define GET_COMPH_CN(cn) ((cn) == 46 ? &d_compH_cn_r13 : &d_compH_cn_r23)
#define GET_COMPH_VN(vn) ((vn) == 68 ? &d_compH_vn_r13 : &d_compH_vn_r23)

// ===================================== Functions Declared ===================================

size_t copy_BG1_R13_Z384_constant();
size_t copy_BG1_R23_Z384_constant();

void init_decoder_constant();
template<typename T>
T** allocate_global_memory(int size_in_bytes, T** out_big_block);
template<typename T>
T** allocate_pinned_memory(int size_in_bytes, T** out_big_block);
void setup_host_memory();
void setup_device_memory();
struct cuda_grid* setup_cuda_grid(int Z, int mb, int nb, int cn, int vn, int Kb);
__device__ __forceinline__ uint32_t scale_int8x4(uint32_t packed, float scale);
__device__ __forceinline__ uint32_t __vapply_sign4(uint32_t val, uint32_t sign_mask);
__global__ void order_kernel(
    uint32_t * d_ordered_llr,   // Output: reordered LLR values packed as uint32_t
    int8_t * d_init_llr,        // Input: initial LLR values (4 codewords interleaved)
    int num
) ;
__global__ void reorder_kernel(
    uint32_t* d_app,        
    int8_t*  d_app_reordered,   
    int num
);
__global__ void cnp_kernel_simd4_1st_iter(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_c2v,         // Output: packed C2V messages
    uint32_t* d_delta_c2v,    // Output: packed delta C2V messages
    int Z,
    int N,
    int cn
);
__global__ void cnp_kernel_simd4(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_c2v,         // Output: packed C2V messages
    uint32_t* d_delta_c2v,    // Output: packed delta (incremental) C2V messages
    int Z,
    int N,
    int cn
);
__global__ void vnp_kernel_simd4(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_delta_c2v,    // Input: packed delta C2V messages
    int Z,
    int N,
    int vn
) ;
// Modified hard decision kernel with bit-reversed order
__global__ void hard_decision_bit_reverse_kernel(
    int8_t* d_app_reordered,   // Input: reordered APP values (information bits only)
    uint8_t* d_hard_bits,       // Output: packed hard-decision bits (1 bit per LLR sign)
    int Z,
    int N,
    int Kb
);
void build_ldpc_graph_per_stream(
    cudaGraph_t graph,
    cuda_grid* g,
    int n_iterations,
    int stream_id,
    int N,
    int Z,
    int cn,
    int vn,
    int Kb
);
void init_graph_BG1_R13_Z384(int n_iterations);
void init_graph_BG1_R23_Z384(int n_iterations);
template<typename T>
void free_global_memory(T** d_ptr, T* big_block);
template<typename T>
void free_pinned_memory(T** h_ptr, T* big_block);
void free_device_mem_all();
void free_host_mem_all();

// ===================================== Functions Implemented ===================================
size_t copy_BG1_R13_Z384_constant() {
    kernel_compH_cn h_compH_cn_r13;
    for (int i = 0; i < BG1_R13_Z384_CN + 1; i++) {
        h_compH_cn_r13.row_ptr[i] = bg1_r13_z384_row_ptr[i];
    }
    for (int i = 0; i < BG1_R13_Z384_NNZ; i++) {
        h_compH_cn_r13.col_idx[i]  = bg1_r13_z384_col_idx[i];
        h_compH_cn_r13.shift_cn[i] = bg1_r13_z384_shift_cn[i];
    }
    h_compH_cn_r13.nnz = BG1_R13_Z384_NNZ;
    

    kernel_compH_vn h_compH_vn_r13;
    for (int j = 0; j <= BG1_R13_Z384_VN; j++) {
        h_compH_vn_r13.col_ptr[j] = bg1_r13_z384_col_ptr[j];
    }
    for (int i = 0; i < BG1_R13_Z384_NNZ; i++) {
        h_compH_vn_r13.row_idx[i]  = bg1_r13_z384_row_idx[i];
        h_compH_vn_r13.shift_vn[i] = bg1_r13_z384_shift_vn[i];
        h_compH_vn_r13.csc2csr[i]  = bg1_r13_z384_csc2csr[i];
    }
    h_compH_vn_r13.nnz = BG1_R13_Z384_NNZ;

    CUDA_CHECK(cudaMemcpyToSymbol(d_compH_cn_r13, &h_compH_cn_r13, sizeof(kernel_compH_cn)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_compH_vn_r13, &h_compH_vn_r13, sizeof(kernel_compH_vn)));

    return sizeof(kernel_compH_cn) + sizeof(kernel_compH_vn);
}

size_t copy_BG1_R23_Z384_constant() {
    kernel_compH_cn h_compH_cn_r23;
    for (int i = 0; i < BG1_R23_Z384_CN + 1; i++) {
        h_compH_cn_r23.row_ptr[i] = bg1_r23_z384_row_ptr[i];
    }
    for (int i = 0; i < BG1_R23_Z384_NNZ; i++) {
        h_compH_cn_r23.col_idx[i]  = bg1_r23_z384_col_idx[i];
        h_compH_cn_r23.shift_cn[i] = bg1_r23_z384_shift_cn[i];
    }
    h_compH_cn_r23.nnz = BG1_R23_Z384_NNZ;
    
    kernel_compH_vn h_compH_vn_r23;
    for (int j = 0; j <= BG1_R23_Z384_VN; j++) {
        h_compH_vn_r23.col_ptr[j] = bg1_r23_z384_col_ptr[j];
    }
    for (int i = 0; i < BG1_R23_Z384_NNZ; i++) {
        h_compH_vn_r23.row_idx[i]  = bg1_r23_z384_row_idx[i];
        h_compH_vn_r23.shift_vn[i] = bg1_r23_z384_shift_vn[i];
        h_compH_vn_r23.csc2csr[i]  = bg1_r23_z384_csc2csr[i];
    }
    h_compH_vn_r23.nnz = BG1_R23_Z384_NNZ;

    CUDA_CHECK(cudaMemcpyToSymbol(d_compH_cn_r23, &h_compH_cn_r23, sizeof(kernel_compH_cn)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_compH_vn_r23, &h_compH_vn_r23, sizeof(kernel_compH_vn)));

    return sizeof(kernel_compH_cn) + sizeof(kernel_compH_vn);   
}

// init gpu decoder constant memory
void init_decoder_constant(){

    size_t BG1_R13_Z384_size = copy_BG1_R13_Z384_constant();
    size_t BG1_R23_Z384_size = copy_BG1_R23_Z384_constant();

    // size_t params_size =  copy_ldpc_params_constant(params);

    size_t total_constant_memory = BG1_R13_Z384_size + BG1_R23_Z384_size;

    printf("Total constant memory used: %zu bytes\n", total_constant_memory);
}

// allocate gpu global memory for multiple streams
template<typename T>
T** allocate_global_memory(int size_in_bytes, T** out_big_block) {
    size_t total_bytes = static_cast<size_t>(MAX_STREAMS) * size_in_bytes;
    T* big_block = NULL;
    CUDA_CHECK(cudaMalloc((void**)&big_block, total_bytes));

    T** d_ptr = (T**)malloc(MAX_STREAMS * sizeof(T*));
    size_t elems_per_stream = size_in_bytes / sizeof(T); 

    for (int i = 0; i < MAX_STREAMS; ++i) {
        d_ptr[i] = big_block + i * elems_per_stream;
    }

    if (out_big_block) {
        *out_big_block = big_block;
    }
    return d_ptr;
}


// allocate cpu pinned memory for multiple streams
template<typename T>
T** allocate_pinned_memory(int size_in_bytes, T** out_big_block) {
    // 1. Allocate a large contiguous pinned (page-locked) host memory block to hold data for all streams
    size_t total_bytes = static_cast<size_t>(MAX_STREAMS) * size_in_bytes;
    T* big_block = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&big_block, total_bytes));

    // 2. Allocate an array of pointers in regular host memory (not pinned)
    T** h_ptr = (T**)malloc(MAX_STREAMS * sizeof(T*));
    size_t elements_per_stream = size_in_bytes / sizeof(T);

    // 3. Point each h_ptr[i] to the corresponding segment within big_block
    for (int i = 0; i < MAX_STREAMS; ++i) {
        h_ptr[i] = big_block + i * elements_per_stream;
    }

    if (out_big_block) {
        *out_big_block = big_block;
    }
    return h_ptr;
}


void setup_host_memory() {

    host_mem = (struct host_memory*)malloc(sizeof(struct host_memory));

    // Allocate pinned host memory for storing initial LLRs
    int llr_size_per_stream = BG1_MAX_CW_LEN * CWS_PER_STREAM * sizeof(int8_t);
    host_mem->h_pinned_llr = allocate_pinned_memory<int8_t>(llr_size_per_stream, &host_mem->h_big_pinned_llr);

    // Allocate pinned host memory for storing hard decision results
    int hard_bits_size_per_stream = BG1_MAX_INFO_LEN / 8 * CWS_PER_STREAM * sizeof(uint8_t);
    host_mem->h_pinned_hard = allocate_pinned_memory<uint8_t>(hard_bits_size_per_stream, &host_mem->h_big_hard_bits);

    int total_pinned_memory = llr_size_per_stream + hard_bits_size_per_stream;
    printf("Total pinned memory used per stream: %d bytes -> llr: %d bytes, hard_bits: %d bytes", total_pinned_memory, llr_size_per_stream, hard_bits_size_per_stream);
}


void setup_device_memory() {

    dev_mem = (struct device_memory*)malloc(sizeof(struct device_memory));

    // Allocate GPU global memory for initial LLRs
    int llr_size_per_stream = BG1_MAX_CW_LEN * CWS_PER_STREAM * sizeof(int8_t);
    dev_mem->d_init_llr = allocate_global_memory<int8_t>(llr_size_per_stream, &dev_mem->d_big_init_llr);

    // Allocate GPU global memory for storing sorted APP values
    int app_size_per_stream = BG1_MAX_CW_LEN * GROUPS_PER_STREAM * sizeof(uint32_t);
    dev_mem->d_app = allocate_global_memory<uint32_t>(app_size_per_stream, &dev_mem->d_big_app);

    // Allocate GPU global memory for storing C2V messages
    int c2v_size_per_stream = GROUPS_PER_STREAM * MAX_Z * BG1_MAX_NNZ * sizeof(uint32_t);
    dev_mem->d_c2v = allocate_global_memory<uint32_t>(c2v_size_per_stream, &dev_mem->d_big_c2v);
    // Allocate GPU global memory for delta C2V; size is the same as C2V
    dev_mem->d_delta_c2v = allocate_global_memory<uint32_t>(c2v_size_per_stream, &dev_mem->d_big_delta_c2v);

    // Allocate GPU global memory for storing reordered APP values
    int app_reordered_size_per_stream = BG1_MAX_CW_LEN * CWS_PER_STREAM * sizeof(int8_t);
    dev_mem->d_app_reordered = allocate_global_memory<int8_t>(app_reordered_size_per_stream, &dev_mem->d_big_app_reordered);

    // Allocate GPU global memory for storing hard decision results
    int hard_bits_size_per_stream = BG1_MAX_INFO_LEN / 8 * CWS_PER_STREAM * sizeof(uint8_t);
    dev_mem->d_hard_bits = allocate_global_memory<uint8_t>(hard_bits_size_per_stream, &dev_mem->d_big_hard_bits);

    int total_device_memory = llr_size_per_stream + app_size_per_stream + c2v_size_per_stream * 2 + app_reordered_size_per_stream + hard_bits_size_per_stream;
    printf("Total global memory used per stream: %d bytes -> llr: %d bytes, app: %d bytes, c2v: %d bytes, "
           "app_reordered: %d bytes, hard_bits: %d bytes\n", total_device_memory, llr_size_per_stream, app_size_per_stream,
           c2v_size_per_stream, app_reordered_size_per_stream, hard_bits_size_per_stream);
}


//  grid block
struct cuda_grid* setup_cuda_grid(int Z, int mb, int nb, int cn, int vn, int Kb){

    struct cuda_grid *g = (struct cuda_grid*)malloc(sizeof(struct cuda_grid));
    g->order_grid = dim3(nb, GROUPS_PER_STREAM, 1);      
    g->order_block = dim3(Z, 1, 1);   

    g->cnp_grid = dim3(cn, GROUPS_PER_STREAM, 1);           
    g->cnp_block = dim3(Z, 1, 1);

    g->vnp_grid = dim3(vn, GROUPS_PER_STREAM, 1);
    g->vnp_block = dim3(Z, 1, 1);

    g->reorder_grid = dim3(nb, GROUPS_PER_STREAM, 1);
    g->reorder_block = dim3(Z, 1, 1);

    g->hard_grid = dim3(Kb, CWS_PER_STREAM, 1);
    g->hard_block = dim3(Z / 8, 1, 1);

    g->cnp_shared_size = Z * BG1_MAX_ROW_DEGREE * sizeof(uint32_t);
    printf("Shared memory used for cnp kernel per block: %zu bytes\n", g->cnp_shared_size);
    return g;
}


__device__ __forceinline__ uint32_t scale_int8x4(uint32_t packed, float scale) {
    // Unpack into char4 (4 int8 values)
    char4 v = *reinterpret_cast<char4*>(&packed);

    // Multiply each element by the float scale factor and convert back to int8
    v.x = (char)__float2int_rn((float)v.x * scale);
    v.y = (char)__float2int_rn((float)v.y * scale);
    v.z = (char)__float2int_rn((float)v.z * scale);
    v.w = (char)__float2int_rn((float)v.w * scale);

    // Pack back into a single uint32_t
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t __vapply_sign4(uint32_t val, uint32_t sign_mask)
{
    // sign_mask: each byte should be 0x00 (positive) or 0xFF (negative)
    uint32_t neg_val = __vnegss4(val); // Saturating negation (-128 → -127)
    return (neg_val & sign_mask) | (val & ~sign_mask);
}


__global__ void order_kernel(
    uint32_t * d_ordered_llr,   // Output: reordered LLR values packed as uint32_t
    int8_t * d_init_llr,        // Input: initial LLR values (4 codewords interleaved)
    int num
) 
{
    int group_id = blockIdx.y;          // Group index within batch [0, GROUPS_PER_STREAM - 1]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index within codeword

    if (tid < num) {
        // Base input address for current thread's bit position across 4 codewords
        int base_in = group_id * SIMD_WIDTH * num + tid;

        // Load LLR values from the same bit position across 4 codewords
        int8_t a = d_init_llr[base_in + 0 * num];  // Codeword 0
        int8_t b = d_init_llr[base_in + 1 * num];  // Codeword 1
        int8_t c = d_init_llr[base_in + 2 * num];  // Codeword 2
        int8_t d = d_init_llr[base_in + 3 * num];  // Codeword 3

        // Pack four 8-bit LLRs into one 32-bit word (little-endian byte order)
        uint32_t packed = (uint32_t)(uint8_t)a |
                          ((uint32_t)(uint8_t)b << 8) |
                          ((uint32_t)(uint8_t)c << 16) |
                          ((uint32_t)(uint8_t)d << 24);

        // Store packed result to output buffer
        d_ordered_llr[group_id * num + tid] = packed;
    }
}

__global__ void reorder_kernel(
    uint32_t* d_app,        
    int8_t*  d_app_reordered,   
    int num
)
{
    int group_id = blockIdx.y;  // Each block.y corresponds to one group
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num) {
        // Read packed value
        uint32_t packed = d_app[group_id * num + tid];

        // Unpack bytes
        int8_t a = (int8_t)(packed & 0xFF);
        int8_t b = (int8_t)((packed >> 8) & 0xFF);
        int8_t c = (int8_t)((packed >> 16) & 0xFF);
        int8_t d = (int8_t)((packed >> 24) & 0xFF);

        // Write back to original codeword layout
        int base_out = group_id * SIMD_WIDTH * num + tid;
        d_app_reordered[base_out + 0 * num] = a;
        d_app_reordered[base_out + 1 * num] = b;
        d_app_reordered[base_out + 2 * num] = c;
        d_app_reordered[base_out + 3 * num] = d;
    }
}

__global__ void cnp_kernel_simd4_1st_iter(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_c2v,         // Output: packed C2V messages
    uint32_t* d_delta_c2v,    // Output: packed delta C2V messages
    int Z,
    int N,
    int cn
) {
    // Declare shared memory
    extern __shared__ uint32_t shared_mem[];
    uint32_t* shared_v2c = shared_mem; 

    int group_idx = blockIdx.y;  // Group index within the batch [0, GROUPS_PER_STREAM - 1]
    int z_idx = threadIdx.x;     // Lane index within the circulant block [0, Z - 1]
    int row = blockIdx.x;        // Index of the current check node layer

    const kernel_compH_cn* d_compH_cn = GET_COMPH_CN(cn);
    const int nnz = d_compH_cn->nnz;          // Number of non-zero elements

    // Boundary checks
    if (z_idx >= Z) return;
    if (row >= cn) return;
    if (group_idx >= GROUPS_PER_STREAM) return;

    int start_idx = d_compH_cn->row_ptr[row];
    int end_idx = d_compH_cn->row_ptr[row + 1];
    int num_edges = end_idx - start_idx;   // Number of edges (variable nodes) connected to this check node

    // Initialize min, second-min, and sign accumulator
    uint32_t sign_temp = 0x00000000U;
    uint32_t min_val = SIMD_MAX_LLR;        // Max positive int8 repeated in all lanes
    uint32_t second_min = SIMD_MAX_LLR;
    uint32_t old_min_val = SIMD_MAX_LLR;
    uint32_t tmp_max = SIMD_MAX_LLR;

    // First pass: gather V2C messages and compute min/second-min per lane
    for (int edge = 0; edge < num_edges; edge++) {
        int edge_idx = start_idx + edge;
        int col = d_compH_cn->col_idx[edge_idx];          // Connected variable node column
        int shift = d_compH_cn->shift_cn[edge_idx];       // Cyclic shift offset
        int shift_val = (z_idx + shift) % Z;             // Compute shifted position within circulant

        int app_addr = group_idx * N + col * Z + shift_val; // Address in APP array
        uint32_t app_val = d_app[app_addr];              // Packed APP values for 4 codewords

        int v2c_addr = edge * Z + z_idx;                 // Shared memory address for V2C
        uint32_t v2c_val = app_val;                      // For first iteration: L_vj->ci = L_vj
        shared_v2c[v2c_addr] = v2c_val;

        // Accumulate sign (XOR across all messages)
        sign_temp ^= v2c_val;

        // Compute absolute values (with saturation for -128 → 127)
        uint32_t v2c_abs = __vabsss4(v2c_val);

        // Update min and second-min per lane
        old_min_val = min_val;
        min_val = __vminu4(min_val, v2c_abs);
        tmp_max = __vmaxu4(old_min_val, v2c_abs);
        second_min = __vminu4(second_min, tmp_max);
    }

    // Compute final sign mask: 0x00 for non-negative, 0xFF for negative
    uint32_t sign = __vcmplts4(sign_temp, 0x00000000U);

    // Second pass: compute C2V messages for each edge
    for (int edge = 0; edge < num_edges; ++edge) {
        int edge_idx = start_idx + edge;
        int v2c_addr = edge * Z + z_idx;
        uint32_t v2c_val = shared_v2c[v2c_addr];

        // Sign of current V2C message
        uint32_t v2c_sign = __vcmplts4(v2c_val, 0x00000000U);
        uint32_t sign_except_edge = sign ^ v2c_sign;  // Global sign excluding current edge

        // Determine if current edge holds the minimum value (per lane)
        uint32_t is_min = __vcmpeq4(__vabsss4(v2c_val), min_val);
        uint32_t min_except_edge = (second_min & is_min) | (min_val & ~is_min);

        // Apply normalization factor α
        uint32_t norm_min_except_edge = scale_int8x4(min_except_edge, ALPHA);

        // Apply correct sign to produce C2V message
        uint32_t new_c2v = __vapply_sign4(norm_min_except_edge, sign_except_edge);

        // Store results
        int c2v_addr = group_idx * Z * nnz + edge_idx * Z + z_idx;
        d_c2v[c2v_addr] = new_c2v;
        d_delta_c2v[c2v_addr] = new_c2v;  // Same as C2V in first iteration
    }
}

__global__ void cnp_kernel_simd4(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_c2v,         // Output: packed C2V messages
    uint32_t* d_delta_c2v,    // Output: packed delta (incremental) C2V messages
    int Z,
    int N,
    int cn
) {
    // Declare shared memory
    extern __shared__ uint32_t shared_mem[];
    uint32_t* shared_v2c = shared_mem;

    int group_idx = blockIdx.y;  // Group index within the batch [0, GROUPS_PER_STREAM - 1]
    int z_idx = threadIdx.x;     // Index within lifting size [0, Z - 1]
    int row = blockIdx.x;        // Current check node layer being processed

    const kernel_compH_cn* d_compH_cn = GET_COMPH_CN(cn);
    const int nnz = d_compH_cn->nnz;          // Number of non-zero elements

    // Boundary checks
    if (z_idx >= Z) return;
    if (row >= cn) return;
    if (group_idx >= GROUPS_PER_STREAM) return;

    int start_idx = d_compH_cn->row_ptr[row];
    int end_idx = d_compH_cn->row_ptr[row + 1];
    int num_edges = end_idx - start_idx;   // Number of edges connected to this check node

    // Initialize min, second-min, and sign accumulator
    uint32_t sign_temp = 0x00000000U;
    uint32_t min_val = SIMD_MAX_LLR;
    uint32_t second_min = SIMD_MAX_LLR;
    uint32_t old_min_val = SIMD_MAX_LLR;
    uint32_t tmp_max = SIMD_MAX_LLR;
    uint32_t v2c_val = 0;

    // First pass: compute V2C = APP - C2V, and gather min/second-min per lane
    for (int edge = 0; edge < num_edges; edge++) {
        int edge_idx = start_idx + edge;
        int col = d_compH_cn->col_idx[edge_idx];          // Connected variable node column
        int shift = d_compH_cn->shift_cn[edge_idx];       // Cyclic shift offset
        int shift_val = (z_idx + shift) % Z;             // Compute shifted position

        int app_addr = group_idx * N + col * Z + shift_val; // Address in APP array
        uint32_t app_val = d_app[app_addr];              // Packed APP for 4 codewords

        int c2v_addr = group_idx * Z * nnz + edge_idx * Z + z_idx; // C2V message address
        uint32_t c2v_val = d_c2v[c2v_addr];

        int v2c_addr = edge * Z + z_idx;                 // Shared memory address for V2C

        // Update V2C message: L_vj->ci = L_vj - L_ci->vj
        v2c_val = __vsubss4(app_val, c2v_val);
        shared_v2c[v2c_addr] = v2c_val;

        // Accumulate sign via XOR
        sign_temp ^= v2c_val;

        // Compute absolute value (with saturation: -128 → 127)
        uint32_t v2c_abs = __vabsss4(v2c_val);

        // Update min and second-min per lane
        old_min_val = min_val;
        min_val = __vminu4(min_val, v2c_abs);
        tmp_max = __vmaxu4(old_min_val, v2c_abs);
        second_min = __vminu4(second_min, tmp_max);
    }

    // Generate sign mask: 0x00 if ≥0, 0xFF if <0
    uint32_t sign = __vcmplts4(sign_temp, 0x00000000U);

    // Second pass: compute new C2V and delta C2V
    for (int edge = 0; edge < num_edges; ++edge) {
        int edge_idx = start_idx + edge;
        int v2c_addr = edge * Z + z_idx;
        uint32_t v2c_val = shared_v2c[v2c_addr];

        int c2v_addr = group_idx * Z * nnz + edge_idx * Z + z_idx;
        uint32_t c2v_val = d_c2v[c2v_addr];

        // Sign of current V2C message
        uint32_t v2c_sign = __vcmplts4(v2c_val, 0);
        uint32_t sign_except_edge = sign ^ v2c_sign;  // Global sign excluding current edge

        // Check if current edge holds the minimum value (per lane)
        uint32_t is_min = __vcmpeq4(__vabsss4(v2c_val), min_val);
        uint32_t min_except_edge = (second_min & is_min) | (min_val & ~is_min);

        // Apply normalization factor α
        uint32_t norm_min_except_edge = scale_int8x4(min_except_edge, ALPHA);

        // Apply sign to produce new C2V message
        uint32_t new_c2v = __vapply_sign4(norm_min_except_edge, sign_except_edge);

        // Store updated C2V
        d_c2v[c2v_addr] = new_c2v;

        // Compute and store delta C2V = new_C2V - old_C2V
        uint32_t delta_c2v = __vsubss4(new_c2v, c2v_val);
        int delta_c2v_addr = group_idx * Z * nnz + edge_idx * Z + z_idx;
        d_delta_c2v[delta_c2v_addr] = delta_c2v;
    }
}

__global__ void vnp_kernel_simd4(
    uint32_t* d_app,         // Input: APP values for 4 codewords in SoA format (packed)
    uint32_t* d_delta_c2v,    // Input: packed delta C2V messages
    int Z,
    int N,
    int vn
) {
    int z_idx = threadIdx.x;                  // Index within lifting dimension [0, Z - 1]
    int group_idx = blockIdx.y;               // Group index within batch [0, GROUPS_PER_STREAM - 1]
    int col = blockIdx.x;                     // Current variable node column being processed

    const kernel_compH_vn* d_compH_vn = GET_COMPH_VN(vn);
    const int nnz = d_compH_vn->nnz;           // Number of non-zero elements

    // Boundary checks
    if (z_idx >= Z) return;
    if (col >= vn) return;
    if (group_idx >= GROUPS_PER_STREAM) return;

    int start_idx = d_compH_vn->col_ptr[col];
    int end_idx = d_compH_vn->col_ptr[col + 1];
    int num_edges = end_idx - start_idx;      // Number of connected check nodes

    int app_addr = group_idx * N + col * Z + z_idx; // Address of current APP value
    uint32_t app_val = d_app[app_addr];

    // Accumulate all incoming delta C2V messages
    for (int edge = 0; edge < num_edges; edge++) {
        int edge_idx = start_idx + edge;
        int shift = d_compH_vn->shift_vn[edge_idx];       // Cyclic shift for this edge

        // Compute reverse-shifted position (for proper alignment)
        int shift_val = (z_idx - shift + Z) % Z;

        // Map CSC index to CSR index for delta C2V lookup
        int csc2csr_idx = d_compH_vn->csc2csr[edge_idx];

        // Address in delta C2V buffer
        int delta_c2v_addr = group_idx * Z * nnz + csc2csr_idx * Z + shift_val;
        uint32_t delta_c2v_val = d_delta_c2v[delta_c2v_addr];

        // Update APP: APP += delta_C2V
        app_val = __vaddss4(app_val, delta_c2v_val);
    }

    // Write back updated APP value
    d_app[app_addr] = app_val;
}

// Modified hard decision kernel with bit-reversed order
__global__ void hard_decision_bit_reverse_kernel(
    int8_t* d_app_reordered,   // Input: reordered APP values (information bits only)
    uint8_t* d_hard_bits,       // Output: packed hard-decision bits (1 bit per LLR sign)
    int Z,
    int N,
    int Kb
) {
    // --- Thread indexing ---
    int z_idx = threadIdx.x * 8;    // Start index in Z dimension, process 8 bits per thread
    int col = blockIdx.x;           // Column (variable node) index
    int cw_idx = blockIdx.y;        // Codeword index within stream [0, CWS_PER_STREAM - 1]

    // Boundary checks
    if (z_idx >= Z) return;
    if (col >= Kb) return;
    if (cw_idx >= CWS_PER_STREAM) return;

    uint8_t val = 0;

    int app_addr = cw_idx * N + col * Z + z_idx; // Base address for 8 consecutive APP values

    // Perform hard decision on 8 consecutive LLRs
    for (int i = 0; i < 8; i++) {
        int8_t app_val = d_app_reordered[app_addr + i];
        // Hard decision: APP ≥ 0 → bit = 0, APP < 0 → bit = 1
        int bit = (app_val >= 0) ? 0 : 1;
        // modify to bit-reverse order
        val |= (bit << (7 - i));
    }

    // Compute output byte address (Kb * Z total info bits, packed 8 per byte)
    int hard_bits_addr = cw_idx * (Kb * Z / 8) + col * (Z / 8) + (z_idx / 8);

    d_hard_bits[hard_bits_addr] = val;
}


// Build a complete CUDA Graph for the specified stream (stream_id), 
// describing the entire LDPC decoding pipeline:
/*
    H2D(LLR) -> order_kernel -> CNP₀ → VNP₀ -> CNP₁ → VNP₁ -> ... -> CNP_{n-1} → VNP_{n-1} 
        -> reorder_kernel -> hard_decision_kernel -> D2H(Hard Bits)
*/
void build_ldpc_graph_per_stream(
    cudaGraph_t graph,
    cuda_grid* g,
    int n_iterations,
    int stream_id,
    int N,
    int Z,
    int cn,
    int vn,
    int Kb
) {
    // cudaGraphNode_t is the handle type for each operation (node) in a CUDA Graph
    cudaGraphNode_t memcpyH2D, orderNode;
    cudaGraphNode_t* cnpNodes = (cudaGraphNode_t*)malloc(n_iterations * sizeof(cudaGraphNode_t));
    cudaGraphNode_t* vnpNodes = (cudaGraphNode_t*)malloc(n_iterations * sizeof(cudaGraphNode_t));
    cudaGraphNode_t reorderNode, hardNode, memcpyD2H;

    // ----------------------------------- H2D -----------------------------------

    int llr_size_per_stream = BG1_MAX_CW_LEN * CWS_PER_STREAM * sizeof(int8_t);
    // Add a 1D memory copy node to the graph.
    /*
        &memcpyH2D: output parameter, returns the handle of the newly created node.
        graph: target graph.
        NULL, 0: list of predecessor nodes (NULL here means no dependencies — this is the graph entry point).
        dev_mem->d_init_llr[stream_id]: destination address on GPU (each stream has its own buffer).
        host_mem->h_pinned_llr[stream_id]: source address on CPU (uses pinned memory for higher transfer efficiency).
        llr_size: number of bytes to copy.
        cudaMemcpyHostToDevice: direction flag.
    */
    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&memcpyH2D, graph, NULL, 0,
        dev_mem->d_init_llr[stream_id],
        host_mem->h_pinned_llr[stream_id],
        llr_size_per_stream, cudaMemcpyHostToDevice));

    // ----------------------------------- order_kernel -----------------------------------

    // orderArgs: list of kernel arguments (must be passed as an array of void*)
    void* orderArgs[] = {
        (void*)&dev_mem->d_app[stream_id],
        (void*)&dev_mem->d_init_llr[stream_id],
        (void*)&N
    };
    cudaKernelNodeParams orderParams = {};
    orderParams.func = (void*)order_kernel;
    orderParams.gridDim = g->order_grid;
    orderParams.blockDim = g->order_block;
    orderParams.sharedMemBytes = 0;
    // kernelParams: pointer to the argument array
    orderParams.kernelParams = orderArgs;
    // extra: used for advanced features like CUDA Stream Capture; not needed here, set to NULL
    orderParams.extra = NULL;

    // Add kernel node to the graph.
    /*  &orderNode: output node handle.
        &memcpyH2D, 1: depends on 1 predecessor node (i.e., execute after H2D copy completes).
        &orderParams: kernel configuration.
    */
    CUDA_CHECK(cudaGraphAddKernelNode(&orderNode, graph, &memcpyH2D, 1, &orderParams));

    // ------------------------------------------- CNP + VNP -------------------------------------------
    cudaGraphNode_t prevNode = orderNode;
    for (int iter = 0; iter < n_iterations; ++iter) {
        void* cnpArgs[] = {
            (void*)&dev_mem->d_app[stream_id],
            (void*)&dev_mem->d_c2v[stream_id],
            (void*)&dev_mem->d_delta_c2v[stream_id],
            (void*)&Z,
            (void*)&N,
            (void*)&cn
        };
        cudaKernelNodeParams cnpParams = {};
        cnpParams.func = (void*)(iter == 0 ? cnp_kernel_simd4_1st_iter : cnp_kernel_simd4);
        cnpParams.gridDim = g->cnp_grid;
        cnpParams.blockDim = g->cnp_block;
        cnpParams.sharedMemBytes = g->cnp_shared_size;
        cnpParams.kernelParams = cnpArgs;
        cnpParams.extra = NULL;
        CUDA_CHECK(cudaGraphAddKernelNode(&cnpNodes[iter], graph, &prevNode, 1, &cnpParams));

        void* vnpArgs[] = {
            (void*)&dev_mem->d_app[stream_id],
            (void*)&dev_mem->d_delta_c2v[stream_id],
            (void*)&Z,
            (void*)&N,
            (void*)&vn
        };
        cudaKernelNodeParams vnpParams = {};
        vnpParams.func = (void*)vnp_kernel_simd4;
        vnpParams.gridDim = g->vnp_grid;
        vnpParams.blockDim = g->vnp_block;
        vnpParams.sharedMemBytes = 0;
        vnpParams.kernelParams = vnpArgs;
        vnpParams.extra = NULL;
        CUDA_CHECK(cudaGraphAddKernelNode(&vnpNodes[iter], graph, &cnpNodes[iter], 1, &vnpParams));

        prevNode = vnpNodes[iter];
    }

    // ------------------------------------- reorder_kernel --------------------------------------------
    void* reorderArgs[] = {
        (void*)&dev_mem->d_app[stream_id],
        (void*)&dev_mem->d_app_reordered[stream_id],
        (void*)&N
    };
    cudaKernelNodeParams reorderParams = {};
    reorderParams.func = (void*)reorder_kernel;
    reorderParams.gridDim = g->reorder_grid;
    reorderParams.blockDim = g->reorder_block;
    reorderParams.sharedMemBytes = 0;
    reorderParams.kernelParams = reorderArgs;
    reorderParams.extra = NULL;
    CUDA_CHECK(cudaGraphAddKernelNode(&reorderNode, graph, &prevNode, 1, &reorderParams));

    // -------------------------------------- hard_decision_kernel ------------------------------------------
    void* hardArgs[] = {
        (void*)&dev_mem->d_app_reordered[stream_id],
        (void*)&dev_mem->d_hard_bits[stream_id],
        (void*)&Z,
        (void*)&N,
        (void*)&Kb
    };
    cudaKernelNodeParams hardParams = {};
    // hardParams.func = (void*)hard_decision_kernel;
    hardParams.func = (void*)hard_decision_bit_reverse_kernel;
    hardParams.gridDim = g->hard_grid;
    hardParams.blockDim = g->hard_block;
    hardParams.sharedMemBytes = 0;
    hardParams.kernelParams = hardArgs;
    hardParams.extra = NULL;
    CUDA_CHECK(cudaGraphAddKernelNode(&hardNode, graph, &reorderNode, 1, &hardParams));

    // ----------------------------------------- D2H -------------------------------------------
    int hard_bits_size_per_stream = BG1_MAX_INFO_LEN / 8 * CWS_PER_STREAM * sizeof(uint8_t);
    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&memcpyD2H, graph, &hardNode, 1,
        host_mem->h_pinned_hard[stream_id],
        dev_mem->d_hard_bits[stream_id],
        hard_bits_size_per_stream, cudaMemcpyDeviceToHost));

    // Free dynamically allocated memory
    free(cnpNodes);
    free(vnpNodes);
}


void init_graph_BG1_R13_Z384(int n_iterations){
    const int N = 26112;
    const int Z = 384;
    const int mb = 46;
    const int nb = 68;
    const int Kb = 22;
    const int cn = 46;
    const int vn = 68;
    struct cuda_grid* g = setup_cuda_grid(Z, mb, nb, cn, vn, Kb);
    for (int stream_id = 0; stream_id < MAX_STREAMS; ++stream_id) {
        CUDA_CHECK(cudaGraphCreate(&cudaGraphs_R13[stream_id], 0));
        build_ldpc_graph_per_stream(
            cudaGraphs_R13[stream_id],
            g,
            n_iterations,
            stream_id,
            N,
            Z,
            cn,
            vn,
            Kb
        );
        CUDA_CHECK(cudaGraphInstantiate(&cudaGraphExecs_R13[stream_id], cudaGraphs_R13[stream_id], NULL, NULL, 0));
    }
}

void init_graph_BG1_R23_Z384(int n_iterations){
    const int N = 26112;
    const int Z = 384;
    const int mb = 46;
    const int nb = 68;
    const int Kb = 22;
    const int cn = 13;
    const int vn = 35;
    struct cuda_grid* g = setup_cuda_grid(Z, mb, nb, cn, vn, Kb);
    for (int stream_id = 0; stream_id < MAX_STREAMS; ++stream_id) {
        CUDA_CHECK(cudaGraphCreate(&cudaGraphs_R23[stream_id], 0));
        build_ldpc_graph_per_stream(
            cudaGraphs_R23[stream_id],
            g,
            n_iterations,
            stream_id,
            N,
            Z,
            cn,
            vn,
            Kb
        );
        CUDA_CHECK(cudaGraphInstantiate(&cudaGraphExecs_R23[stream_id], cudaGraphs_R23[stream_id], NULL, NULL, 0));
    }
}


extern "C" void ldpc_decoder_cuda_init()
{
    params = (struct ldpc_params*)malloc(sizeof(struct ldpc_params));

    // Decoder parameter configuration
    params->rate = 13;
    params->bg_index = 1;
    params->Z = 384;
    params->N = 26112;
    params->K = 8448;
    params->Kb = 22;
    params->mb = BG1_ROW;
    params->nb = BG1_COL;
    params->H = NULL;   // Parity-check matrix H
    params->compH_cn = NULL;
    params->compH_vn = NULL;  // Compressed representation of H for CN and VN processing
    params->n_iterations = 5;

    // Allocate host and device memory
    // MAX_STREAMS * CWS_PER_STREAM
    setup_host_memory();

    setup_device_memory();

    // Initialize GPU constant memory
    init_decoder_constant();

    // Create CUDA streams
    cudaStreams = (cudaStream_t*)malloc(MAX_STREAMS * sizeof(cudaStream_t));
    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    /*
        cudaGraph_t represents an editable graph (a "capture" graph) that can be modified
        (e.g., by adding nodes or edges) but cannot be executed directly.
        It must be instantiated via cudaGraphInstantiate() to produce a cudaGraphExec_t,
        which is the executable form of the graph.
    */
    cudaGraphs_R13 = (cudaGraph_t*)malloc(MAX_STREAMS * sizeof(cudaGraph_t));
    cudaGraphs_R23 = (cudaGraph_t*)malloc(MAX_STREAMS * sizeof(cudaGraph_t));
    cudaGraphExecs_R13 = (cudaGraphExec_t*)malloc(MAX_STREAMS * sizeof(cudaGraphExec_t));
    cudaGraphExecs_R23 = (cudaGraphExec_t*)malloc(MAX_STREAMS * sizeof(cudaGraphExec_t));

    init_graph_BG1_R13_Z384(params->n_iterations);

    init_graph_BG1_R23_Z384(params->n_iterations);

    printf("Initialize BG1_R13_Z384 and BG1_R23_Z384 CUDA Graph\n");
}

template<typename T>
void free_global_memory(T** d_ptr, T* big_block) {
    if(d_ptr){
        free(d_ptr);
    }
    if(big_block){
        CUDA_CHECK(cudaFree(big_block));
    }
}

// Free pinned host memory for multi-stream usage
template<typename T>
void free_pinned_memory(T** h_ptr, T* big_block) {
    if (h_ptr) {
        free(h_ptr);
    }
    if (big_block) {
        CUDA_CHECK(cudaFreeHost(big_block));
    }
}

// Free all global device memory
void free_device_mem_all(){
    if (dev_mem == NULL) return;

    free_global_memory<int8_t>(dev_mem->d_init_llr, dev_mem->d_big_init_llr);
    free_global_memory<uint32_t>(dev_mem->d_app, dev_mem->d_big_app);  
    free_global_memory<uint32_t>(dev_mem->d_c2v, dev_mem->d_big_c2v);
    free_global_memory<uint32_t>(dev_mem->d_delta_c2v, dev_mem->d_big_delta_c2v);
    free_global_memory<int8_t>(dev_mem->d_app_reordered, dev_mem->d_big_app_reordered);
    free_global_memory<uint8_t>(dev_mem->d_hard_bits, dev_mem->d_big_hard_bits);

    // Free the structure itself
    free(dev_mem);
}

// Free all host pinned memory
void free_host_mem_all() {
    if (host_mem == NULL) return;

    free_pinned_memory<int8_t>(host_mem->h_pinned_llr, host_mem->h_big_pinned_llr);
    free_pinned_memory<uint8_t>(host_mem->h_pinned_hard, host_mem->h_big_hard_bits);

    // Free the structure itself
    free(host_mem);
}

extern "C" void ldpc_decoder_cuda_free(){

    CUDA_CHECK(cudaDeviceSynchronize());
    // destroy graphs and streams
    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(cudaStreams[i]));
        CUDA_CHECK(cudaGraphExecDestroy(cudaGraphExecs_R13[i]));
        CUDA_CHECK(cudaGraphDestroy(cudaGraphs_R13[i]));
        CUDA_CHECK(cudaGraphExecDestroy(cudaGraphExecs_R23[i]));
        CUDA_CHECK(cudaGraphDestroy(cudaGraphs_R23[i]));
    }
    
    free(cudaGraphs_R13);
    free(cudaGraphExecs_R13);
    free(cudaGraphs_R23);
    free(cudaGraphExecs_R23);

    free(cudaStreams);
    // free memory
    free_device_mem_all();
    free_host_mem_all();
    free(params);
}
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
// #include <cooperative_groups.h>
// amespace cg = cooperative_groups;

#include "nrLDPC_CUDA_CnProcKernel_BG1_R13.h"
#include "nrLDPC_CUDA_BnProcKernel_BG1_R13.h"
#include "nrLDPC_CUDA_BnToCnPC_Kernel_BG1_R13.h"

#define Q_SCALE 8.0
#define BG1_GRP0_CN 1
#define ZC 384 // for BG1 test only
#define CPU_ADDRESSING 1 // 0 means copy data into gpu memory, for common gpu; 1 for grace hopper which can read cpu memory directly
#define CUDA_STREAM 0 // 1 means use cudastream to run kernels in parallel;
#define MAX_NUM_DLSCH_SEGMENTS_DL 132

#define BIG_KERNEL 1

// decoder_graphs.cu
#include "decoder_graphs.h"

cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL] = {nullptr};
bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL] = {false};

// 适配 CUDA 11+/12+
static const char *ptrTypeName(cudaMemoryType type)
{
  switch (type) {
    case cudaMemoryTypeUnregistered:
      return "Unregistered/Unknown";
    case cudaMemoryTypeHost:
      return "Host (pinned)";
    case cudaMemoryTypeDevice:
      return "Device";
    case cudaMemoryTypeManaged:
      return "Managed";
    default:
      return "Unknown";
  }
}
//
#define CHECK_CUDA(call)                                                                     \
  do {                                                                                       \
    cudaError_t _e = (call);                                                                 \
    if (_e != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      return;                                                                                \
    }                                                                                        \
  } while (0)

//
extern "C" void check_ptr_host(const void *p, const char *name)
{
  cudaPointerAttributes attr;
  cudaError_t e = cudaPointerGetAttributes(&attr, p);
  if (e != cudaSuccess) {
    printf("Ptr %-24s = %p  <cudaPointerGetAttributes failed: %s>\n", name, p, cudaGetErrorString(e));
    return;
  }
  const char *type = "Unregistered/Unknown";
  if (attr.type == cudaMemoryTypeHost)
    type = "Host";
  if (attr.type == cudaMemoryTypeDevice)
    type = "Device";
  if (attr.type == cudaMemoryTypeManaged)
    type = "Managed";
  printf("Ptr %-24s = %p  type=%s  device=%d  devicePointer=%p  hostPointer=%p\n",
         name,
         p,
         type,
         attr.device,
         attr.devicePointer,
         attr.hostPointer);
}

static void dump_arr8_host(const arr8_t *a, const char *name, int idx)
{
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}
static void dump_arr16_host(const arr16_t *a, const char *name, int idx)
{
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}
static void dump_arr32_host(const arr32_t *a, const char *name, int idx)
{
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}

// check lut
void inspect_lut(const t_nrLDPC_lut *p_lut_dev)
{
  printf("==== Inspect t_nrLDPC_lut(dev) @ %p ====\n", (void *)p_lut_dev);
  check_ptr_host(p_lut_dev, "p_lut_dev");

  // 1)
  t_nrLDPC_lut h = {0};
  CHECK_CUDA(cudaMemcpy(&h, p_lut_dev, sizeof(h), cudaMemcpyDeviceToHost));

  // 2)
  check_ptr_host(h.startAddrCnGroups, "startAddrCnGroups");
  check_ptr_host(h.numCnInCnGroups, "numCnInCnGroups");
  check_ptr_host(h.numBnInBnGroups, "numBnInBnGroups");
  check_ptr_host(h.startAddrBnGroups, "startAddrBnGroups");
  check_ptr_host(h.startAddrBnGroupsLlr, "startAddrBnGroupsLlr");
  check_ptr_host(h.llr2llrProcBufAddr, "llr2llrProcBufAddr");
  check_ptr_host(h.llr2llrProcBufBnPos, "llr2llrProcBufBnPos");

  for (int i = 0; i < NR_LDPC_NUM_CN_GROUPS_BG1; ++i) {
    dump_arr16_host(&h.circShift[i], "circShift", i);
    dump_arr32_host(&h.startAddrBnProcBuf[i], "startAddrBnProcBuf", i);
    dump_arr8_host(&h.bnPosBnProcBuf[i], "bnPosBnProcBuf", i);
    dump_arr8_host(&h.posBnInCnProcBuf[i], "posBnInCnProcBuf", i);
  }
  printf("========================================\n");
}

__global__ void check_ptr_kernel(const void *ptr, int id)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("check_ptr id=%d ptr=%p\n", id, ptr);
  }
}

__global__ void check_ptr_kernel_easy(int id)
{
  printf("hello!\n");
}

__device__ __constant__ uint8_t h_block_group_ids_cnProc[50] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                                2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                                                                4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};

__device__ __constant__ uint8_t h_block_CN_idx_cnProc[50] = {0,  0,  1,  2,  3,  4,  0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                             11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
                                                             2,  3,  4,  0,  1,  0,  1,  0, 0, 0, 1, 1, 2, 2, 3, 3};

__device__ __constant__ uint16_t h_block_thread_counts_cnProc[50] = {
    288, 384, 384, 384, 384, 384, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 576,
    576, 576, 576, 576, 576, 576, 576, 672, 672, 672, 672, 672, 768, 768, 864, 864, 960, 912, 912, 912, 912, 912, 912, 912, 912};

__device__ __constant__ uint32_t h_block_input_offsets_cnProc[50] = {
    0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288, 12672,
    13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080, 61824, 62208,
    62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92160, 92544, 92544, 92928, 92928, 93312, 93312};

__device__ __constant__ uint32_t h_block_output_offsets_cnProc[50] = {
    0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288, 12672,
    13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080, 61824, 62208,
    62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92160, 92544, 92544, 92928, 92928, 93312, 93312};

__device__ __constant__ uint8_t h_block_group_ids_BnToCnPC[46] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                                  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                                                                  4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8};

__device__ __constant__ uint8_t h_block_CN_idx_BnToCnPC[46] = {0,  0,  1,  2,  3,  4,  0,  1,  2, 3, 4, 5, 6, 7, 8, 9,
                                                               10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7,
                                                               0,  1,  2,  3,  4,  0,  1,  0,  1, 0, 0, 1, 2, 3};

__device__ __constant__ uint16_t h_block_thread_counts_BnToCnPC[46] = {
    288, 384, 384, 384, 384, 384, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480,
    480, 576, 576, 576, 576, 576, 576, 576, 576, 672, 672, 672, 672, 672, 768, 768, 864, 864, 960, 912, 912, 912, 912};

__device__ __constant__ uint32_t h_block_input_offsets_BnToCnPC[46] = {
    0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
    12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
    61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};

__device__ __constant__ uint32_t h_block_output_offsets_BnToCnPC[46] = {
    0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
    12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
    61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};

  __device__ __constant__ uint8_t lut_CnGrpIdx_BG1_R13[316] = {
    // Group 1 (3 messages)
    1, 1, 1, 
    // Group 2 (20 messages)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    // Group 3 (90 messages)
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    // Group 4 (48 messages)
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
    4, 4, 4, 4, 4, 4, 4, 4, 
    // Group 5 (35 messages)
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
    // Group 6 (16 messages)
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    // Group 7 (18 messages)
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
    // Group 8 (10 messages)
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
    // Group 9 (76 messages)
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
};
  __device__ __constant__ uint8_t lut_CnMsgIdx_BG1_R13[316] = {
    // Group 1 (Nbn=3, Ncn=1)
    1, 2, 3, 
    // Group 2 (Nbn=4, Ncn=5)
    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 
    // Group 3 (Nbn=5, Ncn=18)
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
    1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 
    // Group 4 (Nbn=6, Ncn=8)
    1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 
    3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 
    5, 6, 1, 2, 3, 4, 5, 6, 
    // Group 5 (Nbn=7, Ncn=5)
    1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 
    7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 
    // Group 6 (Nbn=8, Ncn=2)
    1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 
    // Group 7 (Nbn=9, Ncn=2)
    1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    // Group 8 (Nbn=10, Ncn=1)
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    // Group 9 (Nbn=19, Ncn=4)
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
};
__device__ __constant__ uint8_t lut_CnIdx_BG1_R13[316] = {
    // Group 1 (Nbn=3, Ncn=1) -> 1 repeated 3 times
    1, 1, 1, 
    // Group 2 (Nbn=4, Ncn=5) -> 1,2,3,4,5 each repeated 4 times
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 
    // Group 3 (Nbn=5, Ncn=18) -> 1..18 each repeated 5 times
    1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 
    5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 
    9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 
    13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 
    17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 
    // Group 4 (Nbn=6, Ncn=8) -> 1..8 each repeated 6 times
    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 
    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 
    7, 7, 8, 8, 8, 8, 8, 8, 
    // Group 5 (Nbn=7, Ncn=5) -> 1..5 each repeated 7 times
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 
    3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 
    // Group 6 (Nbn=8, Ncn=2) -> 1,2 each repeated 8 times
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
    // Group 7 (Nbn=9, Ncn=2) -> 1,2 each repeated 9 times
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    // Group 8 (Nbn=10, Ncn=1) -> 1 repeated 10 times
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    // Group 9 (Nbn=19, Ncn=4) -> 1..4 each repeated 19 times
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
};

__constant__ static uint8_t d_lut_numBnInCnGroups_BG1_R13[9];
__constant__ static int d_lut_numThreadsEachCnGroupsNeed_BG1_R13[9];
//__constant__ static uint8_t d_lut_numCnInCnGroups_BG1_R13[9];

// === CUDA Error Checking ===
// Wrap any CUDA API call with CHECK(...) to automatically print error info with file and line number
// Example usage: CHECK(cudaMalloc(&ptr, size));
#define CHECK(call) ErrorCheck((call), __FILE__, __LINE__)

void dump_cnProcBufRes_to_file(const int8_t *cnProcBufRes, const char *filename)
{
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("Failed to open dump file");
    exit(EXIT_FAILURE);
  }
  // printf("\nNR_LDPC_SIZE_CN_PROC_BUF: %d\n", NR_LDPC_SIZE_CN_PROC_BUF);

  for (int i = 0; i < NR_LDPC_SIZE_CN_PROC_BUF; i++) {
    fprintf(fp, "%02x ", (uint8_t)cnProcBufRes[i]);
    if ((i + 1) % 16 == 0)
      fprintf(fp, "\n");
  }

  fclose(fp);
}
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

//------------------------------------------------------------------------
//------------------------------------------------------------------------
//-----------------------CUDA Scheduler Area------------------------------
//------------------------------------------------------------------------
//------------------------------------------------------------------------
__global__ void cnProcKernel_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                             const int8_t *__restrict__ d_cnBufAll,
                                             int8_t *__restrict__ d_cnOutAll,
                                             int8_t *__restrict__ d_bnBufAll,
                                             int Zc,
                                             int8_t *iter_ptr,
                                             int8_t numMaxIter,
                                             int *PC_Flag)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }
  // printf("I'm inside cnProc_kernel\n");

  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...315
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  //uint16_t blockSize = h_block_thread_counts_cnProc[blk];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 30336) //30336 is the total processed 316 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;

  switch (groupIdx) {
    case 0:
      cnProcKernel_BG1_R13_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 1:
      cnProcKernel_BG1_R13_int8_G4 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 2:
      cnProcKernel_BG1_R13_int8_G5 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 3:
      cnProcKernel_BG1_R13_int8_G6 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 4:
      cnProcKernel_BG1_R13_int8_G7 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 5:
      cnProcKernel_BG1_R13_int8_G8 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 6:
      cnProcKernel_BG1_R13_int8_G9 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 7:
      cnProcKernel_BG1_R13_int8_G10 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
    case 8:
      cnProcKernel_BG1_R13_int8_G19 (p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, MsgIdx, lane, groupIdx, CnIdx, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                        int8_t *cnProcBuf,
                                        int8_t *cnProcBufRes,
                                        int8_t *bnProcBuf,
                                        int Z,
                                        int8_t *iter_ptr,
                                        int8_t numMaxIter,
                                        int *PC_Flag,
                                        cudaStream_t *streams,
                                        int8_t CudaStreamIdx)
{
#if BIG_KERNEL
  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);
  int maxBlockSize = 1024; // Maximun threads are 960
  dim3 gridDim(30); // 50
  dim3 blockDim(maxBlockSize);

  cnProcKernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                 cnProcBuf,
                                                                                 cnProcBufRes,
                                                                                 bnProcBuf,
                                                                                 Z,
                                                                                 iter_ptr,
                                                                                 numMaxIter,
                                                                                 PC_Flag);
  // printf("Check point 1001: ");
  // CHECK(cudaGetLastError());

#else
  printf("To be continued ^ ^\n");
#endif
}

__global__ void bnProcPcKernel_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                               int8_t *__restrict__ d_bnProcBufRes,
                                               int8_t *__restrict__ d_llrProcBuf,
                                               int8_t *__restrict__ d_llrRes,
                                               const uint8_t *lut_numBnInBnGroups,
                                               const uint32_t *lut_startAddrBnBuf,
                                               const uint16_t *lut_startAddrBnLlr,
                                               int Zc,
                                               int8_t *iter_ptr,
                                               int8_t numMaxIter,
                                               int *PC_Flag)
{
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  /*if (tid == 0) {
    printf("2: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= 6528) {
    return;
  }
  static const uint8_t lut_GrpIdx[68] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 28, 30,
  };

  static const uint8_t lut_BnIdx[68] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1,  1,  1,  2,
      1,  2,  3,  4,  1,  2,  3,  1,  1,  2,  3,  4,  1,  2,  3,  1,  2,  3,  4,  1,  1,  1,
  };
  //                                          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12,
  //                                          13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30
  static const uint8_t lut_BnToAddrIdx[30] = {1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  12, 0, 13};
  int row = tid / 96; // to decide the inner block
  int lane = tid % 96; // to decide the inner lane

  uint8_t GrpIdx = lut_GrpIdx[row];
  // uint8_t MsgIdx = lut_MsgIdx[row];
  uint8_t BnIdx = lut_BnIdx[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx[GrpIdx - 1];
  uint8_t GrpNum = lut_numBnInBnGroups[GrpIdx - 1];

  const int8_t *p_bnProcBuf_Grp = (const int8_t *)(d_bnProcBuf + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_bnProcBufRes_Grp = (const int8_t *)(d_bnProcBufRes + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_llrProcBuf_Grp = (const int8_t *)(d_llrProcBuf + lut_startAddrBnLlr[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp = (const int8_t *)(d_llrRes + lut_startAddrBnLlr[BnToAddrIdx - 1]);

  bnProcPcKernel_int8_Gn(p_bnProcBuf_Grp, p_bnProcBufRes_Grp, p_llrProcBuf_Grp, p_llrRes_Grp, lane, GrpIdx, BnIdx, GrpNum, Zc);
  // grid);

  // t1:
}

__global__ void bnProcKernel_int8_BIG_stream(const int8_t *__restrict__ d_bnProcBuf,
                                             int8_t *__restrict__ d_bnProcBufRes,
                                             int8_t *__restrict__ d_llrProcBuf,
                                             int8_t *__restrict__ d_llrRes,
                                             const uint8_t *lut_numBnInBnGroups,
                                             const uint32_t *lut_startAddrBnBuf,
                                             const uint16_t *lut_startAddrBnLlr,
                                             int Zc,
                                             int8_t *iter_ptr,
                                             int8_t numMaxIter,
                                             int *PC_Flag)
{
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  /*if (tid == 0) {
    printf("3: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }*/
  if (tid >= 30336) {
    return;
  }
  static const uint8_t lut_GrpIdx[316] = {
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,
      6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
      7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,
      9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 28, 28, 28,
      28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30,
      30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
  };

  static const uint8_t lut_MsgIdx[316] = {
      1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  1,  2,  3,  4,  5,  6,  1,
      2,  3,  4, 5,  6,  7,  1,  2,  3,  4,  5,  6,  7,  1,  2,  3,  4,  5,  6,  7,  1,  2,  3,  4,  5,  6,  7,  1,  2,  3,  4,  5,
      6,  7,  8, 1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
      5,  6,  7, 8,  9,  10, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 1,  2,  3,  4,  5,  6,
      7,  8,  9, 10, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 1,  2,  3,  4,  5,  6,
      7,  8,  9, 10, 11, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 1,  2,  3,
      4,  5,  6, 7,  8,  9,  10, 11, 12, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 1, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 1,  2,
      3,  4,  5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  };

  static const uint8_t lut_BnIdx[316] = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,
      2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,
      4,  4,  4,  4,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,
      3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
      3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
  };
  //                                          1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
  //                                          28,29, 30
  static const uint8_t lut_BnToAddrIdx[30] = {1, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  12, 0, 13};
  int row = tid / 96; // to decide the inner block
  int lane = tid % 96; // to decide the inner lane

  uint8_t GrpIdx = lut_GrpIdx[row];
  uint8_t MsgIdx = lut_MsgIdx[row];
  uint8_t BnIdx = lut_BnIdx[row];
  uint8_t BnToAddrIdx = lut_BnToAddrIdx[GrpIdx - 1];
  uint8_t GrpNum = lut_numBnInBnGroups[GrpIdx - 1];

  const int8_t *p_bnProcBuf_Grp = (const int8_t *)(d_bnProcBuf + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_bnProcBufRes_Grp = (const int8_t *)(d_bnProcBufRes + lut_startAddrBnBuf[BnToAddrIdx - 1]);
  const int8_t *p_llrProcBuf_Grp = (const int8_t *)(d_llrProcBuf + lut_startAddrBnLlr[BnToAddrIdx - 1]);
  const int8_t *p_llrRes_Grp = (const int8_t *)(d_llrRes + lut_startAddrBnLlr[BnToAddrIdx - 1]);

  bnProcKernel_int8_Gn(p_bnProcBuf_Grp,
                       p_bnProcBufRes_Grp,
                       p_llrProcBuf_Grp,
                       p_llrRes_Grp,
                       lane,
                       GrpIdx,
                       MsgIdx,
                       BnIdx,
                       GrpNum,
                       Zc);
}

void nrLDPC_bnProc_BG1_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                        int8_t *bnProcBuf,
                                        int8_t *bnProcBufRes,
                                        int8_t *llrProcBuf,
                                        int8_t *llrRes,
                                        int Z,
                                        int8_t *iter_ptr,
                                        int8_t numMaxIter,
                                        int *PC_Flag,
                                        cudaStream_t *streams,
                                        int8_t CudaStreamIdx)
{
  const uint8_t *lut_numBnInBnGroups;
  const uint32_t *lut_startAddrBnGroups;
  const uint16_t *lut_startAddrBnGroupsLlr;

  lut_numBnInBnGroups = p_lut->numBnInBnGroups;
  lut_startAddrBnGroups = p_lut->startAddrBnGroups;
  lut_startAddrBnGroupsLlr = p_lut->startAddrBnGroupsLlr;

  int8_t *p_bnProcBuf = (int8_t *)bnProcBuf;
  int8_t *p_bnProcBufRes = (int8_t *)bnProcBufRes;
  int8_t *p_llrProcBuf = (int8_t *)llrProcBuf;
  int8_t *p_llrRes = (int8_t *)llrRes;

#if BIG_KERNEL
  int maxBlockSize = 1024; // Z;
  int totalBlocks = 30;

  dim3 gridDim(totalBlocks);
  dim3 blockDim(maxBlockSize);

  bnProcPcKernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_bnProcBuf,
                                                                                   p_bnProcBufRes,
                                                                                   p_llrProcBuf,
                                                                                   p_llrRes,
                                                                                   lut_numBnInBnGroups,
                                                                                   lut_startAddrBnGroups,
                                                                                   lut_startAddrBnGroupsLlr,
                                                                                   Z,
                                                                                   iter_ptr,
                                                                                   numMaxIter,
                                                                                   PC_Flag);
  // printf("In stream %d B: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
  bnProcKernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_bnProcBuf,
                                                                                 p_bnProcBufRes,
                                                                                 p_llrProcBuf,
                                                                                 p_llrRes,
                                                                                 lut_numBnInBnGroups,
                                                                                 lut_startAddrBnGroups,
                                                                                 lut_startAddrBnGroupsLlr,
                                                                                 Z,
                                                                                 iter_ptr,
                                                                                 numMaxIter,
                                                                                 PC_Flag);

#else

  printf("\n *************** To be continued *************** \n");

#endif
}

__global__ void BnToCnPC_Kernel_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                                int8_t *__restrict__ d_bnOutAll,
                                                const int8_t *__restrict__ d_cnBufAll,
                                                int8_t *__restrict__ d_cnOutAll,
                                                int8_t *__restrict__ d_bnBufAll,
                                                int8_t *d_llrRes,
                                                int Zc,
                                                int8_t *iter_ptr,
                                                int8_t numMaxIter,
                                                int *PC_Flag,
                                                e_nrLDPC_outMode outMode,
                                                int8_t *p_out,
                                                int8_t *llrOut,
                                                int8_t *p_llrOut,
                                                uint32_t numLLR)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;


  const uint32_t *lut_startAddrs = p_lut->startAddrCnGroups;

  int row = tid / 96; // to decide the global MsgIdx; row = 0,1,2...315
  int lane = tid % 96; // to decide the inner lane
  // if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupIdx = lut_CnGrpIdx_BG1_R13[row] - 1;
  // if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = lut_CnIdx_BG1_R13[row] - 1;
  uint8_t MsgIdx = lut_CnMsgIdx_BG1_R13[row];
  uint32_t inOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  uint32_t outOffset = lut_startAddrs[groupIdx] + 384 * CnIdx;
  // if(blk == 1&&tid == 0) printf("1.2\n");
  //   __syncthreads();

  if (tid >= 30336) //30336 is the total processed 316 msg * 96
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;
  int8_t *p_bnProcBufRes = (int8_t *)d_bnOutAll;
  int8_t *p_llrRes = (int8_t *)d_llrRes;



  // Early stopping
  if (!(*iter_ptr > numMaxIter || *PC_Flag == 0)) {
    if (tid == 0) {
      *PC_Flag = 0;
      // printf("4: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
    }
    //__syncthreads();

    // uint32_t pcRes = 0; // setting flag for Parity Check

    switch (groupIdx) {
      case 0:
        CnToBnPC_Kernel_int8_G3_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 1:
        CnToBnPC_Kernel_int8_G4_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 2:
        CnToBnPC_Kernel_int8_G5_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 3:
        CnToBnPC_Kernel_int8_G6_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 4:
        CnToBnPC_Kernel_int8_G7_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 5:
        CnToBnPC_Kernel_int8_G8_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 6:
        CnToBnPC_Kernel_int8_G9_Stream(p_lut,
                                       p_bnProcBufRes,
                                       p_cnProcBuf,
                                       p_cnProcBufRes,
                                       p_bnProcBuf,
                                       MsgIdx,
                                       lane,
                                       groupIdx,
                                       CnIdx,
                                       Zc,
                                       PC_Flag);
        break;
      case 7:
        CnToBnPC_Kernel_int8_G10_Stream(p_lut,
                                        p_bnProcBufRes,
                                        p_cnProcBuf,
                                        p_cnProcBufRes,
                                        p_bnProcBuf,
                                        MsgIdx,
                                        lane,
                                        groupIdx,
                                        CnIdx,
                                        Zc,
                                        PC_Flag);
        break;
      case 8:
        CnToBnPC_Kernel_int8_G19_Stream(p_lut,
                                        p_bnProcBufRes,
                                        p_cnProcBuf,
                                        p_cnProcBufRes,
                                        p_bnProcBuf,
                                        MsgIdx,
                                        lane,
                                        groupIdx,
                                        CnIdx,
                                        Zc,
                                        PC_Flag);
        break;
    }
  }

  if (*iter_ptr == numMaxIter) { // output
    llrRes2llrOut_Kernel_int8_BG1(p_lut, p_llrOut, p_llrRes, Zc);
  } else {
    if (tid == 0) {
      // printf("Why you guys not here when iter_ptr = %d???\n",*iter_ptr);
      (*iter_ptr)++;
    }
  }
}

__global__ void OutPut_Kernel_int8_BIG_stream(const t_nrLDPC_lut *p_lut,
                                              int Zc,
                                              int8_t *iter_ptr,
                                              int8_t numMaxIter,
                                              int *PC_Flag,
                                              e_nrLDPC_outMode outMode,
                                              int8_t *p_out,
                                              int8_t *llrOut,
                                              int8_t *p_llrOut,
                                              uint32_t numLLR)
{
  // only activate in the last iteration

  if (*iter_ptr == numMaxIter) {
    if (outMode == nrLDPC_outMode_BIT)
      llr2bitPacked_Kernel_int8_BG1((uint8_t *)p_out, p_llrOut, numLLR);

    else // if (outMode == nrLDPC_outMode_BITINT8)
      llr2bit_Kernel_int8_BG1((uint8_t *)p_out, p_llrOut, numLLR);
  } else
    return;
}

void nrLDPC_BnToCnPC_BG1_cuda_stream_core(const t_nrLDPC_lut *p_lut,
                                          int8_t *bnProcBufRes,
                                          int8_t *cnProcBuf,
                                          int8_t *cnProcBufRes,
                                          int8_t *bnProcBuf,
                                          int8_t *llrRes,
                                          int Z,
                                          int8_t *iter_ptr,
                                          int8_t numMaxIter,
                                          int *PC_Flag,
                                          e_nrLDPC_outMode outMode,
                                          int8_t *p_out,
                                          int8_t *llrOut,
                                          int8_t *p_llrOut,
                                          uint32_t numLLR,
                                          cudaStream_t *streams,
                                          int8_t CudaStreamIdx)
{
  const uint32_t *lut_startAddrCnGroups = p_lut->startAddrCnGroups;

  const int numGroups = 9;

#if BIG_KERNEL
  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  int maxBlockSize = 1024; // Maximun threads are 960
  dim3 gridDim(30);
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  // printf("In stream %d BC: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
  BnToCnPC_Kernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                    bnProcBufRes,
                                                                                    cnProcBuf,
                                                                                    cnProcBufRes,
                                                                                    bnProcBuf,
                                                                                    llrRes,
                                                                                    Z,
                                                                                    iter_ptr,
                                                                                    numMaxIter,
                                                                                    PC_Flag,
                                                                                    outMode,
                                                                                    p_out,
                                                                                    llrOut,
                                                                                    p_llrOut,
                                                                                    numLLR);

  OutPut_Kernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                  Z,
                                                                                  iter_ptr,
                                                                                  numMaxIter,
                                                                                  PC_Flag,
                                                                                  outMode,
                                                                                  p_out,
                                                                                  llrOut,
                                                                                  p_llrOut,
                                                                                  numLLR);

  // printf("Check point 1001: ");

  // CHECK(cudaGetLastError());
#else
  printf("To be continued ^ ^");
#endif
}

__global__ void check_lut_kernel(const t_nrLDPC_lut *p_lut)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("=== Device p_lut->startAddrBnProcBuf dump ===\n");
    for (int i = 0; i < 9; i++) {
      printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
             i,
             (void *)p_lut->startAddrBnProcBuf[i].d,
             p_lut->startAddrBnProcBuf[i].dim1,
             p_lut->startAddrBnProcBuf[i].dim2);
    }

    printf("=== Device p_lut->bnPosBnProcBuf dump ===\n");
    for (int i = 0; i < 9; i++) {
      printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
             i,
             (void *)p_lut->bnPosBnProcBuf[i].d,
             p_lut->bnPosBnProcBuf[i].dim1,
             p_lut->bnPosBnProcBuf[i].dim2);
    }
  }
}

extern "C" void nrLDPC_decoder_scheduler_BG1_cuda_core(const t_nrLDPC_lut *p_lut,
                                                       int8_t *p_out,
                                                       uint32_t numLLR,
                                                       int8_t *cnProcBuf,
                                                       int8_t *cnProcBufRes,
                                                       int8_t *bnProcBuf,
                                                       int8_t *bnProcBufRes,
                                                       int8_t *llrRes,
                                                       int8_t *llrProcBuf,
                                                       int8_t *llrOut,
                                                       int8_t *p_llrOut,
                                                       int Z,
                                                       uint8_t BG,
                                                       uint8_t R,
                                                       uint8_t numMaxIter,
                                                       e_nrLDPC_outMode outMode,
                                                       cudaStream_t *streams,
                                                       uint8_t CudaStreamIdx,
                                                       cudaEvent_t *doneEvent,
                                                       int8_t *iter_ptr,
                                                       int *PC_Flag)
{
#if 1 // CPU_ADDRESSING

  cudaStream_t stream = streams[CudaStreamIdx];
  // cudaEvent_t captureDoneEvent[MAX_NUM_DLSCH_SEGMENTS];
  // cudaEvent_t captureDoneEvent[MAX_NUM_DLSCH_SEGMENTS];

  if (!graphCreated[CudaStreamIdx]) {
    printf("Creating the graph for stream %d\n", CudaStreamIdx);
    if (CudaStreamIdx != 0) {
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }
    // CHECK(cudaGetLastError());
    /*
        // print all the address to see if they are isolated
        printf("Stream %d parameter addresses:\n", CudaStreamIdx);
        printf("  p_lut       = %p\n", (void*)p_lut);
        printf("  p_out       = %p\n", (void*)p_out);
        printf("  cnProcBuf   = %p\n", (void*)cnProcBuf);
        printf("  cnProcBufRes= %p\n", (void*)cnProcBufRes);
        printf("  bnProcBuf   = %p\n", (void*)bnProcBuf);
        printf("  bnProcBufRes= %p\n", (void*)bnProcBufRes);
        printf("  llrRes      = %p\n", (void*)llrRes);
        printf("  llrProcBuf  = %p\n", (void*)llrProcBuf);
        printf("  llrOut      = %p\n", (void*)llrOut);
        printf("  p_llrOut    = %p\n", (void*)p_llrOut);
        printf("  iter_ptr    = %p\n", (void*)iter_ptr);
        printf("  PC_Flag     = %p\n", (void*)PC_Flag);
        fflush(stdout);
    check_lut_kernel<<<1,1,0,streams[CudaStreamIdx]>>>(p_lut);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    */
    // Start graph recording
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // check_ptr_kernel_easy<<<1,10>>>(2);
    // cudaDeviceSynchronize();
    // CHECK(cudaGetLastError());
    for (int i = 0; i <= numMaxIter; i++) {
      // printf("I'm inside the loop i = %d\n", i);
      nrLDPC_cnProc_BG1_cuda_stream_core(p_lut,
                                         cnProcBuf,
                                         cnProcBufRes,
                                         bnProcBuf,
                                         (int)Z,
                                         iter_ptr,
                                         numMaxIter,
                                         PC_Flag,
                                         streams,
                                         CudaStreamIdx);
      CHECK(cudaGetLastError());
      // cd cudaDeviceSynchronize();

      // printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
      nrLDPC_bnProc_BG1_cuda_stream_core(p_lut,
                                         bnProcBuf,
                                         bnProcBufRes,
                                         llrProcBuf,
                                         llrRes,
                                         (int)Z,
                                         iter_ptr,
                                         numMaxIter,
                                         PC_Flag,
                                         streams,
                                         CudaStreamIdx);
      // cudaDeviceSynchronize();

      // printf("In stream %d 2: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
      CHECK(cudaGetLastError());
      // cudaDeviceSynchronize();
      nrLDPC_BnToCnPC_BG1_cuda_stream_core(p_lut,
                                           bnProcBufRes,
                                           cnProcBuf,
                                           cnProcBufRes,
                                           bnProcBuf,
                                           llrRes,
                                           (int)Z,
                                           iter_ptr,
                                           numMaxIter,
                                           PC_Flag,
                                           outMode,
                                           p_out,
                                           llrOut,
                                           p_llrOut,
                                           numLLR,
                                           streams,
                                           CudaStreamIdx);
      CHECK(cudaGetLastError());
      // cudaDeviceSynchronize();
      // printf("In stream %d 3: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
    }

    // stop recording
    cudaStreamEndCapture(stream, &decoderGraphs[CudaStreamIdx]);
    // printf("5\n");
    cudaGraphInstantiate(&decoderGraphExec[CudaStreamIdx], decoderGraphs[CudaStreamIdx], NULL, NULL, 0);
    graphCreated[CudaStreamIdx] = true;

    // Execute （make sure the first trial finish）
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    // cudaDeviceSynchronize();
    // printf("Graphs should be captured\n");
    // cudaStreamSynchronize(stream);
  } else {
    // printf("Are you here???\n");
    //  reuse the graph after
    if (CudaStreamIdx != 0) {
      // uncomment below if you want streams works in sequence
      // cudaStreamWaitEvent(streams[CudaStreamIdx], doneEvent[CudaStreamIdx-1], 0);//cudaEventSynchronize(doneEvent[CudaStreamIdx -
      // 1]);
    }
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    //
  }
  // CHECK(cudaGetLastError());
#else
  printf("To be continued ^ ^\n");
#endif
}

extern "C" bool is_device_pointer(const void *p)
{
  if (p == NULL)
    return false;
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, p);
  if (err != cudaSuccess) {
    // cudaPointerGetAttributes return value might vary in different runtime version
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attrs.type == cudaMemoryTypeDevice);
#else
  return (attrs.memoryType == cudaMemoryTypeDevice);
#endif
}
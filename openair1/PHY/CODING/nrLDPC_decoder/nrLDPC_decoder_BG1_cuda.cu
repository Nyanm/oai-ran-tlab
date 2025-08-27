#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
// #include <cooperative_groups.h>
// amespace cg = cooperative_groups;

#include "nrLDPC_CnProcKernel_BG1_cuda.h"
#include "nrLDPC_BnProcKernel_BG1_cuda.h"
#include "nrLDPC_BnToCnPC_Kernel_BG1_cuda.h"

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
static const char* ptrTypeName(cudaMemoryType type) {
  switch (type) {
    case cudaMemoryTypeUnregistered: return "Unregistered/Unknown";
    case cudaMemoryTypeHost:         return "Host (pinned)";
    case cudaMemoryTypeDevice:       return "Device";
    case cudaMemoryTypeManaged:      return "Managed";
    default:                         return "Unknown";
  }
}
// 简单的错误检查宏
#define CHECK_CUDA(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(_e));           \
    return;                                                        \
  }                                                                \
} while(0)

// 你已有的：打印指针属性（别去解引用）
extern "C" void check_ptr_host(const void *p, const char *name) {
  cudaPointerAttributes attr;
  cudaError_t e = cudaPointerGetAttributes(&attr, p);
  if (e != cudaSuccess) {
    printf("Ptr %-24s = %p  <cudaPointerGetAttributes failed: %s>\n",
           name, p, cudaGetErrorString(e));
    return;
  }
  const char *type = "Unregistered/Unknown";
  if (attr.type == cudaMemoryTypeHost)    type = "Host";
  if (attr.type == cudaMemoryTypeDevice)  type = "Device";
  if (attr.type == cudaMemoryTypeManaged) type = "Managed";
  printf("Ptr %-24s = %p  type=%s  device=%d  devicePointer=%p  hostPointer=%p\n",
         name, p, type, attr.device, attr.devicePointer, attr.hostPointer);
}

static void dump_arr8_host(const arr8_t *a, const char *name, int idx) {
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}
static void dump_arr16_host(const arr16_t *a, const char *name, int idx) {
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}
static void dump_arr32_host(const arr32_t *a, const char *name, int idx) {
  char tag[64];
  snprintf(tag, sizeof(tag), "%s[%d].d", name, idx);
  printf("%s[%d]: dim1=%d dim2=%d\n", name, idx, a->dim1, a->dim2);
  check_ptr_host(a->d, tag);
}

// 用设备指针调用这个函数
void inspect_lut(const t_nrLDPC_lut *p_lut_dev) {
  printf("==== Inspect t_nrLDPC_lut(dev) @ %p ====\n", (void*)p_lut_dev);
  check_ptr_host(p_lut_dev, "p_lut_dev");

  // 1) 先把“头”拷回主机（浅拷贝）
  t_nrLDPC_lut h = {0};
  CHECK_CUDA(cudaMemcpy(&h, p_lut_dev, sizeof(h), cudaMemcpyDeviceToHost));

  // 2) 现在用这份主机副本里的“设备指针值”做属性查询即可
  check_ptr_host(h.startAddrCnGroups,    "startAddrCnGroups");
  check_ptr_host(h.numCnInCnGroups,      "numCnInCnGroups");
  check_ptr_host(h.numBnInBnGroups,      "numBnInBnGroups");
  check_ptr_host(h.startAddrBnGroups,    "startAddrBnGroups");
  check_ptr_host(h.startAddrBnGroupsLlr, "startAddrBnGroupsLlr");
  check_ptr_host(h.llr2llrProcBufAddr,   "llr2llrProcBufAddr");
  check_ptr_host(h.llr2llrProcBufBnPos,  "llr2llrProcBufBnPos");

  for (int i = 0; i < NR_LDPC_NUM_CN_GROUPS_BG1; ++i) {
    dump_arr16_host(&h.circShift[i],          "circShift",          i);
    dump_arr32_host(&h.startAddrBnProcBuf[i], "startAddrBnProcBuf", i);
    dump_arr8_host (&h.bnPosBnProcBuf[i],     "bnPosBnProcBuf",     i);
    dump_arr8_host (&h.posBnInCnProcBuf[i],   "posBnInCnProcBuf",   i);
  }
  printf("========================================\n");
}


 __global__ void check_ptr_kernel(const void* ptr, int id) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("check_ptr id=%d ptr=%p\n", id, ptr);
  }
}

__global__ void check_ptr_kernel_easy(int id) {
    printf("hello!\n");
}

  __device__ __constant__ uint8_t h_block_group_ids_cnProc[50] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};

  __device__ __constant__ uint8_t h_block_CN_idx_cnProc[50] = {0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0,
                                             1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 0, 1, 0,  0,  0,  1,  1,  2,  2,  3,  3};

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

__device__ __constant__ uint8_t h_block_group_ids_BnToCnPC[46] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8};

  __device__ __constant__ uint8_t h_block_CN_idx_BnToCnPC[46] = {0,  0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                             17, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 0,  1,  0,  0,  1,  2,  3};

  __device__ __constant__ uint16_t h_block_thread_counts_BnToCnPC[46] = {288, 384, 384, 384, 384, 384, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480,
                                                     480, 480, 480, 480, 480, 480, 480, 480, 576, 576, 576, 576, 576, 576, 576, 576,
                                                     672, 672, 672, 672, 672, 768, 768, 864, 864, 960, 912, 912, 912, 912};

  __device__ __constant__ uint32_t h_block_input_offsets_BnToCnPC[46] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
      12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
      61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};

  __device__ __constant__ uint32_t h_block_output_offsets_BnToCnPC[46] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
      12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
      61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};



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

__global__ void cnProcKernel_int8_BIG(const t_nrLDPC_lut *p_lut,
                                      const int8_t *__restrict__ d_cnBufAll,
                                      int8_t *__restrict__ d_cnOutAll,
                                      int8_t *__restrict__ d_bnBufAll,
                                      const uint8_t *__restrict__ block_group_ids,
                                      const uint8_t *__restrict__ block_CN_idx,
                                      const uint16_t *__restrict__ block_thread_counts,
                                      const uint32_t *__restrict__ block_input_offsets,
                                      const uint32_t *__restrict__ block_output_offsets,
                                      int Zc)
{
  int blk = blockIdx.x;
  int tid = threadIdx.x;
  if(blk == 0 && tid == 0) printf("kernel launched\n");
  uint8_t groupId = block_group_ids[blk];
  uint8_t CnIdx = block_CN_idx[blk];
  uint16_t blockSize = block_thread_counts[blk];
  uint32_t inOffset = block_input_offsets[blk];
  uint32_t outOffset = block_output_offsets[blk];

  if (tid >= blockSize)
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;
  // if(blk == 45 && tid == 64){
  // printf("d_cnBufAll = %p, d_cnOutAll = %p, p_cnProcBuf = %p, p_cnProcBufRes = %p, inOffset = %d, outOffset = %d \n", d_cnBufAll,
  // d_cnOutAll, p_cnProcBuf, p_cnProcBufRes, inOffset, outOffset);
  //}
  /*
if(tid == 65 && blk == 24){
    printf("=== cnProcKernel_int8_G3 INPUTS ===\n");
    printf("p_lut = %p\n", p_lut);
    printf("d_cnBufAll = %p\n", d_cnBufAll);
    printf("d_cnOutAll = %p\n", d_cnOutAll);
    printf("d_bnBufAll = %p\n", d_bnBufAll);
    printf("tid = %d\n", tid);
    printf("Zc = %d\n", Zc);
}*/

  switch (groupId) {
    case 0:
      cnProcKernel_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 1:
      cnProcKernel_int8_G4(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 2:
      cnProcKernel_int8_G5(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 3:
      cnProcKernel_int8_G6(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 4:
      cnProcKernel_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 5:
      cnProcKernel_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 6:
      cnProcKernel_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 7:
      cnProcKernel_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 8:
      cnProcKernel_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
  }
}

void nrLDPC_cnProc_BG1_cuda_core(const t_nrLDPC_lut *p_lut, int8_t *cnProcBuf, int8_t *cnProcBufRes, int8_t *bnProcBuf, int Z)
{
  // const uint8_t h_lut_numBnInCnGroups_BG1_R13[] = {3, 4, 5, 6, 7, 8, 9, 10, 19};
  // const int h_lut_numThreadsEachCnGroupsNeed_BG1_R13[] = {288, 384, 480, 576, 672, 768, 864, 960, 1824};
  // const uint8_t h_lut_numCnInCnGroups_BG1_R13[] = {1, 5, 18, 8, 5, 2, 2, 1, 4};

  // const uint8_t *lut_numCnInCnGroups = (const uint8_t *)p_lut->numCnInCnGroups;
  const uint32_t *lut_startAddrCnGroups = lut_startAddrCnGroups_BG1;

  const int numGroups = 9;

#if BIG_KERNEL

  static const uint8_t h_block_group_ids[50] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};

  static const uint8_t h_block_CN_idx[50] = {0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0,
                                             1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 0, 1, 0,  0,  0,  1,  1,  2,  2,  3,  3};

  static const uint16_t h_block_thread_counts[50] = {
      288, 384, 384, 384, 384, 384, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 576,
      576, 576, 576, 576, 576, 576, 576, 672, 672, 672, 672, 672, 768, 768, 864, 864, 960, 912, 912, 912, 912, 912, 912, 912, 912};

  static const uint32_t h_block_input_offsets[50] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288, 12672,
      13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080, 61824, 62208,
      62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92160, 92544, 92544, 92928, 92928, 93312, 93312};

  static const uint32_t h_block_output_offsets[50] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288, 12672,
      13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080, 61824, 62208,
      62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92160, 92544, 92544, 92928, 92928, 93312, 93312};

  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  int maxBlockSize = 960; // Maximun threads are 960
  dim3 gridDim(50);
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  cnProcKernel_int8_BIG<<<gridDim, blockDim>>>(p_lut,
                                               cnProcBuf,
                                               cnProcBufRes,
                                               bnProcBuf,
                                               h_block_group_ids,
                                               h_block_CN_idx,
                                               h_block_thread_counts,
                                               h_block_input_offsets,
                                               h_block_output_offsets,
                                               Z);
  // printf("Check point 1001: ");
  // CHECK(cudaGetLastError());

#else
#if !CUDA_STREAM
  // No cuda stream using
  for (int i = 0; i < numGroups; ++i) {
    p_cnProcBuf = cnProcBuf + lut_startAddrCnGroups[i];
    p_cnProcBufRes = cnProcBufRes + lut_startAddrCnGroups[i];
    // printf("\nlut_startAddrCnGroups[%d]: %d\n", i, (int)lut_startAddrCnGroups[i]);

    // printf("In i = %d, p_cnProcBuf = %p, p_cnProcBufRes = %p", i, (void *)p_cnProcBuf, (void *)p_cnProcBufRes);

    switch (i) {
      case 0:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);

        // print p_cnProcBuf和p_cnProcBufRes firt int8_t element
        /*printf("BG3: p_cnProcBuf first all elements: ");
        for (int idx = 0; idx < 1152; idx++)
        {
            printf("%x ", p_cnProcBuf[idx]);
        }*/
        // printf("\n");

        // cudaPointerAttributes attr;
        // cudaPointerGetAttributes(&attr, p_cnProcBuf);
        // printf("p_cnProcBuf is on %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");
        cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);

        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        /*
                    printf("p_cnProcBufRes first 10 elements: ");
                    for (int idx = 0; idx < 1152; idx++)
                    {
                        printf("%d ", p_cnProcBufRes[idx]);
                    }
                    printf("\n");*/

        break;
      case 1:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        // h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 2:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 3:
        //("launching kernel[%d]: grid=%d, block=%d\n", i,
        //      h_lut_numCnInCnGroups_BG1_R13[i],
        //      h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 4:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 5:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 6:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                p_cnProcBufRes,
                                                                                                                Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 7:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]>>>(p_cnProcBuf,
                                                                                                                 p_cnProcBufRes,
                                                                                                                 Z);
        /*printf("BG3: p_cnProcBuf first all elements: ");
for (int idx = 0; idx < 1152; idx++)
{
printf("%x ", p_cnProcBuf[idx]);
}
printf("\n");*/
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
      case 8:
        // printf("launching kernel[%d]: grid=%d, block=%d\n", i,
        //        h_lut_numCnInCnGroups_BG1_R13[i],
        //        h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i]);
        //  Group 19: split into 2x blocks, half threads
        cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2, h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        CHECK(cudaGetLastError());
        // cudaDeviceSynchronize();
        break;
    }
  }
#else
  // Create CUDA streams for concurrent kernel execution
  cudaStream_t streams[numGroups];
  for (int i = 0; i < numGroups; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  // Launch each group kernel on a separate stream
  for (int i = 0; i < numGroups; ++i) {
    p_cnProcBuf = &cnProcBuf[lut_startAddrCnGroups[i]];
    p_cnProcBufRes = &cnProcBufRes[lut_startAddrCnGroups[i]];

    switch (i) {
      case 0:
        cnProcKernel_int8_G3<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 1:
        cnProcKernel_int8_G4<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 2:
        cnProcKernel_int8_G5<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 3:
        cnProcKernel_int8_G6<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 4:
        cnProcKernel_int8_G7<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 5:
        cnProcKernel_int8_G8<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 6:
        cnProcKernel_int8_G9<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 7:
        cnProcKernel_int8_G10<<<h_lut_numCnInCnGroups_BG1_R13[i], h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i], 0, streams[i]>>>(
            p_cnProcBuf,
            p_cnProcBufRes,
            Z);
        break;
      case 8:
        // Group 19 requires more than 1024 threads, so split into 2x blocks, half threads
        cnProcKernel_int8_G19<<<h_lut_numCnInCnGroups_BG1_R13[i] * 2,
                                h_lut_numThreadsEachCnGroupsNeed_BG1_R13[i] / 2,
                                0,
                                streams[i]>>>(p_cnProcBuf, p_cnProcBufRes, Z);
        break;
    }
  }

  // Wait for all streams to finish
  for (int i = 0; i < numGroups; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]); // Release stream resources
  }

#endif
#endif
  // CHECK(cudaGetLastError());
  // CHECK(cudaDeviceSynchronize());
}

// CUDA wrapper function: external interface identical to the original C version
extern "C" void nrLDPC_cnProc_BG1_cuda(const t_nrLDPC_lut *p_lut,
                                       int8_t *cnProcBuf,
                                       int8_t *cnProcBufRes,
                                       int8_t *bnProcBuf,
                                       uint16_t Z)
{
  // printf("CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
#if CPU_ADDRESSING
  // printf("\nVery very first cnProcBuf = %p, cnProcBufRes = %p \n", cnProcBuf, cnProcBufRes);
  nrLDPC_cnProc_BG1_cuda_core(p_lut, cnProcBuf, cnProcBufRes, bnProcBuf, (int)Z);

#else
  // printf("Here CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
  size_t cnProcBuf_size = 200000 /* buffer size for cnProcBuf  */;
  size_t cnProcBufRes_size = 200000 /* buffer size for cnProcBufRes  */;

  // use Unified Memory
  int8_t *d_cnProcBuf = nullptr;
  int8_t *d_cnProcBufRes = nullptr;

  cudaError_t err;

  // ask for unified memory
  err = cudaMallocManaged(&d_cnProcBuf, cnProcBuf_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBuf failed: %s\n", cudaGetErrorString(err));
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBuf success, d_cnProcBuf = %p\n", (void *)d_cnProcBuf);
  }

  err = cudaMallocManaged(&d_cnProcBufRes, cnProcBufRes_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBufRes failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_cnProcBuf);
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBufRes success, d_cnProcBufRes = %p\n", (void *)d_cnProcBufRes);
  }

  //
  memcpy(d_cnProcBuf, cnProcBuf, cnProcBuf_size);
  memset(d_cnProcBufRes, 0, cnProcBufRes_size);

  // kernel function
  nrLDPC_cnProc_BG1_cuda_core(p_lut, d_cnProcBuf, d_cnProcBufRes, (int)Z);

  memcpy(cnProcBufRes, d_cnProcBufRes, cnProcBufRes_size);

  // free memory
  cudaFree(d_cnProcBuf);
  cudaFree(d_cnProcBufRes);
#endif
  // cudaPointerAttributes attr;
  // cudaPointerGetAttributes(&attr, d_cnProcBuf);
  // printf("d_cnProcBuf is on %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");
  cudaDeviceSynchronize();
}

__global__ void bnProcPcKernel_int8_BIG(const int8_t *__restrict__ d_bnProcBuf,
                                        int8_t *__restrict__ d_bnProcBufRes,
                                        int8_t *__restrict__ d_llrProcBuf,
                                        int8_t *__restrict__ d_llrRes,
                                        const uint8_t *lut_numBnInBnGroups,
                                        const uint32_t *lut_startAddrBnBuf,
                                        const uint16_t *lut_startAddrBnLlr,
                                        int Zc)
{
  // cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void bnProcKernel_int8_BIG(const int8_t *__restrict__ d_bnProcBuf,
                                      int8_t *__restrict__ d_bnProcBufRes,
                                      int8_t *__restrict__ d_llrProcBuf,
                                      int8_t *__restrict__ d_llrRes,
                                      const uint8_t *lut_numBnInBnGroups,
                                      const uint32_t *lut_startAddrBnBuf,
                                      const uint16_t *lut_startAddrBnLlr,
                                      int Zc)
{
  // cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

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
  // grid);

  // t1:
}

__global__ void bnProcKernel_int8_BIG_United(const int8_t *__restrict__ d_bnProcBuf,
                                             int8_t *__restrict__ d_bnProcBufRes,
                                             int8_t *__restrict__ d_llrProcBuf,
                                             int8_t *__restrict__ d_llrRes,
                                             const uint8_t *lut_numBnInBnGroups,
                                             const uint32_t *lut_startAddrBnBuf,
                                             const uint16_t *lut_startAddrBnLlr,
                                             int Zc)
{
  // cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

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

  bnProcKernel_int8_Gn_United(p_bnProcBuf_Grp,
                              p_bnProcBufRes_Grp,
                              p_llrProcBuf_Grp,
                              p_llrRes_Grp,
                              lane,
                              GrpIdx,
                              MsgIdx,
                              BnIdx,
                              GrpNum,
                              Zc);
  // grid);

  // t1:
}

void nrLDPC_bnProc_BG1_cuda_core(const t_nrLDPC_lut *p_lut,
                                 int8_t *bnProcBuf,
                                 int8_t *bnProcBufRes,
                                 int8_t *llrProcBuf,
                                 int8_t *llrRes,
                                 int Z)
{
  const uint8_t *lut_numBnInBnGroups;
  const uint32_t *lut_startAddrBnGroups;
  const uint16_t *lut_startAddrBnGroupsLlr;

  lut_numBnInBnGroups = lut_numBnInBnGroups_BG1_R13;
  lut_startAddrBnGroups = lut_startAddrBnGroups_BG1_R13;
  lut_startAddrBnGroupsLlr = lut_startAddrBnGroupsLlr_BG1_R13;

  int8_t *p_bnProcBuf = (int8_t *)bnProcBuf;
  int8_t *p_bnProcBufRes = (int8_t *)bnProcBufRes;
  int8_t *p_llrProcBuf = (int8_t *)llrProcBuf;
  int8_t *p_llrRes = (int8_t *)llrRes;
/*
  // --- compute totalBlocks ---
  int totalBlocks = 0;
  for (int k = 1; k <= 30; k++) {
      int numBn = lut_numBnInBnGroups[k - 1];
      totalBlocks += numBn * k;
  }
  printf("Total blocks required = %d\n", totalBlocks);
*/
#if BIG_KERNEL
  int maxBlockSize = 1024; // Z;
  int totalBlocks = 30;

  dim3 gridDim(totalBlocks);
  dim3 blockDim(maxBlockSize);

  /*
    bnProcKernel_int8_BIG_United<<<gridDim, blockDim>>>(p_bnProcBuf,
                                                 p_bnProcBufRes,
                                                 p_llrProcBuf,
                                                 p_llrRes,
                                                 lut_numBnInBnGroups,
                                                 lut_startAddrBnGroups,
                                                 lut_startAddrBnGroupsLlr,
                                                 Z);
  */
  bnProcPcKernel_int8_BIG<<<gridDim, blockDim>>>(p_bnProcBuf,
                                                 p_bnProcBufRes,
                                                 p_llrProcBuf,
                                                 p_llrRes,
                                                 lut_numBnInBnGroups,
                                                 lut_startAddrBnGroups,
                                                 lut_startAddrBnGroupsLlr,
                                                 Z);

  bnProcKernel_int8_BIG<<<gridDim, blockDim>>>(p_bnProcBuf,
                                               p_bnProcBufRes,
                                               p_llrProcBuf,
                                               p_llrRes,
                                               lut_numBnInBnGroups,
                                               lut_startAddrBnGroups,
                                               lut_startAddrBnGroupsLlr,
                                               Z);

  /*

     // check device cooperative capability
     cudaDeviceProp props;
     cudaGetDeviceProperties(&props, 0);
     if (!props.cooperativeLaunch) {
         printf("ERROR: Device does not support cooperative launch required for grid.sync.\n");
         return;
     }

     // set up kernel arguments
     void *kernelArgs[] = {
         (void *)&p_bnProcBuf,
         (void *)&p_bnProcBufRes,
         (void *)&p_llrProcBuf,
         (void *)&p_llrRes,
         (void *)&lut_numBnInBnGroups,
         (void *)&lut_startAddrBnGroups,
         (void *)&lut_startAddrBnGroupsLlr,
         (void *)&Z
     };

     // cooperative launch
     cudaError_t err = cudaLaunchCooperativeKernel(
         (void*)bnProcKernel_int8_BIG,
         gridDim,
         blockDim,
         kernelArgs,
         0,
         nullptr);

     if (err != cudaSuccess) {
         printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(err));
     }
        */
  // printf("Check point 1101: ");
  // CHECK(cudaGetLastError());

#else

  printf("\n *************** To be continued *************** \n");

#endif
}

extern "C" void nrLDPC_bnProc_BG1_cuda(const t_nrLDPC_lut *p_lut,
                                       int8_t *bnProcBuf,
                                       int8_t *bnProcBufRes,
                                       int8_t *llrProcBuf,
                                       int8_t *llrRes,
                                       uint16_t Z)
{
#if CPU_ADDRESSING
  // printf("\nVery very first bnProcBuf = %p, bnProcBufRes = %p,  llrProcBuf = %p, llrRes = %p \n", bnProcBuf, bnProcBufRes,
  // llrProcBuf, llrRes);

  nrLDPC_bnProc_BG1_cuda_core(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, (int)Z);

#else
  // printf("Here CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
  size_t cnProcBuf_size = 200000 /* buffer size for cnProcBuf  */;
  size_t cnProcBufRes_size = 200000 /* buffer size for cnProcBufRes  */;

  // use Unified Memory
  int8_t *d_cnProcBuf = nullptr;
  int8_t *d_cnProcBufRes = nullptr;

  cudaError_t err;

  // ask for unified memory
  err = cudaMallocManaged(&d_cnProcBuf, cnProcBuf_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBuf failed: %s\n", cudaGetErrorString(err));
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBuf success, d_cnProcBuf = %p\n", (void *)d_cnProcBuf);
  }

  err = cudaMallocManaged(&d_cnProcBufRes, cnProcBufRes_size);
  if (err != cudaSuccess) {
    printf("cudaMallocManaged d_cnProcBufRes failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_cnProcBuf);
    return;
  } else {
    // printf("cudaMallocManaged d_cnProcBufRes success, d_cnProcBufRes = %p\n", (void *)d_cnProcBufRes);
  }

  //
  memcpy(d_cnProcBuf, cnProcBuf, cnProcBuf_size);
  memset(d_cnProcBufRes, 0, cnProcBufRes_size);

  // kernel function
  nrLDPC_cnProc_BG1_cuda_core(p_lut, d_cnProcBuf, d_cnProcBufRes, (int)Z);

  memcpy(cnProcBufRes, d_cnProcBufRes, cnProcBufRes_size);

  // free memory
  cudaFree(d_cnProcBuf);
  cudaFree(d_cnProcBufRes);
#endif
  // cudaPointerAttributes attr;
  // cudaPointerGetAttributes(&attr, d_cnProcBuf);
  // printf("d_cnProcBuf is on %s memory\n", attr.type == cudaMemoryTypeDevice ? "device" : "host");
  cudaDeviceSynchronize();
}

__global__ void BnToCnPC_Kernel_int8_BIG(const t_nrLDPC_lut *p_lut,
                                         int8_t *__restrict__ d_bnOutAll,
                                         const int8_t *__restrict__ d_cnBufAll,
                                         int8_t *__restrict__ d_cnOutAll,
                                         int8_t *__restrict__ d_bnBufAll,
                                         const uint8_t *__restrict__ block_group_ids,
                                         const uint8_t *__restrict__ block_CN_idx,
                                         const uint16_t *__restrict__ block_thread_counts,
                                         const uint32_t *__restrict__ block_input_offsets,
                                         const uint32_t *__restrict__ block_output_offsets,
                                         int Zc,
                                         int *PC_Flag)
{
  int blk = blockIdx.x;
  int tid = threadIdx.x;

  uint8_t groupId = block_group_ids[blk];
  uint8_t CnIdx = block_CN_idx[blk];
  uint16_t blockSize = block_thread_counts[blk];
  uint32_t inOffset = block_input_offsets[blk];
  uint32_t outOffset = block_output_offsets[blk];

  if (tid >= blockSize)
    return;

  int8_t *p_bnProcBufRes = (int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;

  switch (groupId) {
    case 0:
      CnToBnPC_Kernel_int8_G3(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 1:
      CnToBnPC_Kernel_int8_G4(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 2:
      CnToBnPC_Kernel_int8_G5(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 3:
      CnToBnPC_Kernel_int8_G6(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 4:
      CnToBnPC_Kernel_int8_G7(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 5:
      CnToBnPC_Kernel_int8_G8(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 6:
      CnToBnPC_Kernel_int8_G9(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 7:
      CnToBnPC_Kernel_int8_G10(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
    case 8:
      CnToBnPC_Kernel_int8_G19(p_lut, p_bnProcBufRes, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc, PC_Flag);
      break;
  }
}
void nrLDPC_BnToCnPC_BG1_cuda_core(const t_nrLDPC_lut *p_lut,
                                   int8_t *bnProcBufRes,
                                   int8_t *cnProcBuf,
                                   int8_t *cnProcBufRes,
                                   int8_t *bnProcBuf,
                                   int Z,
                                   int *PC_Flag)
{
  // const uint8_t h_lut_numBnInCnGroups_BG1_R13[] = {3, 4, 5, 6, 7, 8, 9, 10, 19};
  // const int h_lut_numThreadsEachCnGroupsNeed_BG1_R13[] = {288, 384, 480, 576, 672, 768, 864, 960, 1824};
  // const uint8_t h_lut_numCnInCnGroups_BG1_R13[] = {1, 5, 18, 8, 5, 2, 2, 1, 4};

  // const uint8_t *lut_numCnInCnGroups = (const uint8_t *)p_lut->numCnInCnGroups;
  const uint32_t *lut_startAddrCnGroups = lut_startAddrCnGroups_BG1;

  const int numGroups = 9;

#if BIG_KERNEL

  static const uint8_t h_block_group_ids[46] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8};

  static const uint8_t h_block_CN_idx[46] = {0,  0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                             17, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 0,  1,  0,  0,  1,  2,  3};

  static const uint16_t h_block_thread_counts[46] = {288, 384, 384, 384, 384, 384, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480,
                                                     480, 480, 480, 480, 480, 480, 480, 480, 576, 576, 576, 576, 576, 576, 576, 576,
                                                     672, 672, 672, 672, 672, 768, 768, 864, 864, 960, 912, 912, 912, 912};

  static const uint32_t h_block_input_offsets[46] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
      12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
      61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};

  static const uint32_t h_block_output_offsets[46] = {
      0,     1152,  1536,  1920,  2304,  2688,  8832,  9216,  9600,  9984,  10368, 10752, 11136, 11520, 11904, 12288,
      12672, 13056, 13440, 13824, 14208, 14592, 14976, 15360, 43392, 43776, 44160, 44544, 44928, 45312, 45696, 46080,
      61824, 62208, 62592, 62976, 63360, 75264, 75648, 81408, 81792, 88320, 92160, 92544, 92928, 93312};

  // printf("\nInitial addr : cnProcBuf = %p, cnProcBufRes = %p\n", cnProcBuf, cnProcBufRes);

  int maxBlockSize = 960; // Maximun threads are 960
  dim3 gridDim(46);
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  BnToCnPC_Kernel_int8_BIG<<<gridDim, blockDim>>>(p_lut,
                                                  bnProcBufRes,
                                                  cnProcBuf,
                                                  cnProcBufRes,
                                                  bnProcBuf,
                                                  h_block_group_ids,
                                                  h_block_CN_idx,
                                                  h_block_thread_counts,
                                                  h_block_input_offsets,
                                                  h_block_output_offsets,
                                                  Z,
                                                  PC_Flag);
  // printf("Check point 1001: ");

  // CHECK(cudaGetLastError());
#else
  printf("To be continued ^ ^");
#endif
}

extern "C" void nrLDPC_BnToCnPC_BG1_cuda(const t_nrLDPC_lut *p_lut,
                                         int8_t *bnProcBufRes,
                                         int8_t *cnProcBuf,
                                         int8_t *cnProcBufRes,
                                         int8_t *bnProcBuf,
                                         uint16_t Z,
                                         int *PC_Flag)
{
  // printf("CPU_ADDRESSING: %d\n", CPU_ADDRESSING);
#if CPU_ADDRESSING
  // printf("\nVery very first cnProcBuf = %p, cnProcBufRes = %p \n", cnProcBuf, cnProcBufRes);
  nrLDPC_BnToCnPC_BG1_cuda_core(p_lut, bnProcBufRes, cnProcBuf, cnProcBufRes, bnProcBuf, (int)Z, PC_Flag);

#else
  printf("To be continued ^ ^\n");

#endif
  cudaDeviceSynchronize();
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
    int blk = blockIdx.x;
  int tid = threadIdx.x;
  /*
  if (threadIdx.x == 0 && blockIdx.x == 1) {
        printf("=== Kernel Parameter Dump ===\n");
        printf("p_lut       = %p\n", (void*)p_lut);
        printf("p_cnBuf        = %p\n", (void*)d_cnBufAll);
        printf("p_cnOut= %p\n", (void*)d_cnOutAll);
        printf("bnProcBuf   = %p\n", (void*)d_bnBufAll);
        printf("Zc          = %d\n", Zc);
        printf("PC_Flag     = %p\n", (int)PC_Flag);
        printf("*PC_Flag     = %d\n", (int)*PC_Flag);
        printf("iter_ptr    = %p\n", (void*)iter_ptr);
        if (iter_ptr) {
            printf("  *iter_ptr = %d\n", *iter_ptr);
        }
        printf("=============================\n");
    }
  if (blk == 1 && tid == 0) {
  printf("=== p_lut dump ===\n");
  printf("  startAddrCnGroups   = %p\n", (void*)p_lut->startAddrCnGroups);
  printf("  numCnInCnGroups     = %p\n", (void*)p_lut->numCnInCnGroups);
  printf("  numBnInBnGroups     = %p\n", (void*)p_lut->numBnInBnGroups);
  printf("  startAddrBnGroups   = %p\n", (void*)p_lut->startAddrBnGroups);
  printf("  startAddrBnGroupsLlr= %p\n", (void*)p_lut->startAddrBnGroupsLlr);
  printf("  llr2llrProcBufAddr  = %p\n", (void*)p_lut->llr2llrProcBufAddr);
  printf("  llr2llrProcBufBnPos = %p\n", (void*)p_lut->llr2llrProcBufBnPos);

  // 如果需要确认这些数组不是全 0，可以打印其中几个元素
  if (p_lut->startAddrCnGroups)
    printf("   startAddrCnGroups[0] = %u\n", p_lut->startAddrCnGroups[0]);
  if (p_lut->numCnInCnGroups)
    printf("   numCnInCnGroups[0] = %u\n", p_lut->numCnInCnGroups[0]);
  if (p_lut->numBnInBnGroups)
    printf("   numBnInBnGroups[0] = %u\n", p_lut->numBnInBnGroups[0]);
  if (p_lut->startAddrBnGroups)
    printf("   startAddrBnGroups[0] = %u\n", p_lut->startAddrBnGroups[0]);
  if (p_lut->startAddrBnGroupsLlr)
    printf("   startAddrBnGroupsLlr[0] = %u\n", p_lut->startAddrBnGroupsLlr[0]);
  if (p_lut->llr2llrProcBufAddr)
    printf("   llr2llrProcBufAddr[0] = %u\n", p_lut->llr2llrProcBufAddr[0]);
  if (p_lut->llr2llrProcBufBnPos)
    printf("   llr2llrProcBufBnPos[0] = %u\n", p_lut->llr2llrProcBufBnPos[0]);
  printf("====================\n");
}

    __syncthreads();
 */
    //if (*iter_ptr == 0)
    //*PC_Flag = 1;
  // Early stopping
  if (*iter_ptr > numMaxIter || *PC_Flag == 0) {
    return;
  }
//printf("I'm inside cnProc_kernel\n");

//if(blk == 1&&tid == 0) printf("I'm inside cnProc_kernel\n");
  uint8_t groupId = h_block_group_ids_cnProc[blk];
  //if(blk == 1&&tid == 0) printf("1.1\n");
  uint8_t CnIdx = h_block_CN_idx_cnProc[blk];
  uint16_t blockSize = h_block_thread_counts_cnProc[blk];
  uint32_t inOffset = h_block_input_offsets_cnProc[blk];
  uint32_t outOffset = h_block_output_offsets_cnProc[blk];
//if(blk == 1&&tid == 0) printf("1.2\n");
  //  __syncthreads();

  if (tid >= blockSize)
    return;

  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;

  switch (groupId) {
    case 0:
      cnProcKernel_int8_G3(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 1:
      cnProcKernel_int8_G4(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 2:
      cnProcKernel_int8_G5(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 3:
      cnProcKernel_int8_G6(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 4:
      cnProcKernel_int8_G7(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 5:
      cnProcKernel_int8_G8(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 6:
      cnProcKernel_int8_G9(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 7:
      cnProcKernel_int8_G10(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
      break;
    case 8:
      cnProcKernel_int8_G19(p_lut, p_cnProcBuf, p_cnProcBufRes, p_bnProcBuf, tid, groupId, CnIdx, Zc);
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
  int maxBlockSize = 960; // Maximun threads are 960
  dim3 gridDim(50);//50
  dim3 blockDim(maxBlockSize);

  cnProcKernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                 cnProcBuf,
                                                                                 cnProcBufRes,
                                                                                 bnProcBuf,
                                                                                 Z,
                                                                                 iter_ptr,
                                                                                 numMaxIter,
                                                                                 PC_Flag);
   //printf("Check point 1001: ");
   //CHECK(cudaGetLastError());

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
//printf("In stream %d B: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
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
                                                const uint8_t *__restrict__ block_group_ids,
                                                const uint8_t *__restrict__ block_CN_idx,
                                                const uint16_t *__restrict__ block_thread_counts,
                                                const uint32_t *__restrict__ block_input_offsets,
                                                const uint32_t *__restrict__ block_output_offsets,
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
  int blk = blockIdx.x;
  int tid = threadIdx.x;

  uint8_t groupId = h_block_group_ids_BnToCnPC[blk];
  uint8_t CnIdx = h_block_CN_idx_BnToCnPC[blk];
  uint16_t blockSize = h_block_thread_counts_BnToCnPC[blk];
  uint32_t inOffset = h_block_input_offsets_BnToCnPC[blk];
  uint32_t outOffset = h_block_output_offsets_BnToCnPC[blk];

  if (tid >= blockSize)
    return;

  int8_t *p_bnProcBufRes = (int8_t *)d_bnOutAll;
  const int8_t *p_cnProcBuf = (const int8_t *)(d_cnBufAll + inOffset);
  int8_t *p_cnProcBufRes = (int8_t *)(d_cnOutAll + outOffset);
  int8_t *p_bnProcBuf = (int8_t *)d_bnBufAll;
  int8_t *p_llrRes = (int8_t *)d_llrRes;

  // Early stopping
  if (!(*iter_ptr > numMaxIter || *PC_Flag == 0)) {
 
  


  if (tid == 0 && blk == 0) {
    *PC_Flag = 0;

    // printf("4: Iter = %d, PC_Flag = %d\n", *iter_ptr, *PC_Flag);
  }
  //__syncthreads();

  // uint32_t pcRes = 0; // setting flag for Parity Check


  switch (groupId) {
    case 0:
      CnToBnPC_Kernel_int8_G3_Stream(p_lut,
                                     p_bnProcBufRes,
                                     p_cnProcBuf,
                                     p_cnProcBufRes,
                                     p_bnProcBuf,
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                     tid,
                                     groupId,
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
                                      tid,
                                      groupId,
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
                                      tid,
                                      groupId,
                                      CnIdx,
                                      Zc,
                                      PC_Flag);
      break;
  }
}

if (*iter_ptr == numMaxIter) { // output
    llrRes2llrOut_Kernel_int8_BG1(p_lut, p_llrOut, p_llrRes, Zc);
}   else {
    if (tid == 0 && blk == 0) {
      //printf("Why you guys not here when iter_ptr = %d???\n",*iter_ptr);
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

  int maxBlockSize = 960; // Maximun threads are 960
  dim3 gridDim(46);
  dim3 blockDim(maxBlockSize);
  // printf("bnProcBuf =  %p\n", bnProcBuf);
  //printf("In stream %d BC: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
  BnToCnPC_Kernel_int8_BIG_stream<<<gridDim, blockDim, 0, streams[CudaStreamIdx]>>>(p_lut,
                                                                                    bnProcBufRes,
                                                                                    cnProcBuf,
                                                                                    cnProcBufRes,
                                                                                    bnProcBuf,
                                                                                    llrRes,
                                                                                    h_block_group_ids_BnToCnPC,
                                                                                    h_block_CN_idx_BnToCnPC,
                                                                                    h_block_thread_counts_BnToCnPC,
                                                                                    h_block_input_offsets_BnToCnPC,
                                                                                    h_block_output_offsets_BnToCnPC,
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

   //CHECK(cudaGetLastError());
#else
  printf("To be continued ^ ^");
#endif
}

__global__ void check_lut_kernel(const t_nrLDPC_lut *p_lut) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== Device p_lut->startAddrBnProcBuf dump ===\n");
        for (int i = 0; i < 9; i++) {
            printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
                   i,
                   (void*)p_lut->startAddrBnProcBuf[i].d,
                   p_lut->startAddrBnProcBuf[i].dim1,
                   p_lut->startAddrBnProcBuf[i].dim2);
        }

        printf("=== Device p_lut->bnPosBnProcBuf dump ===\n");
        for (int i = 0; i < 9; i++) {
            printf("[%d] .d=%p, .dim1=%d, .dim2=%d\n",
                   i,
                   (void*)p_lut->bnPosBnProcBuf[i].d,
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
                                                       int8_t* iter_ptr,
                                                       int* PC_Flag)
{ 
#if 1//CPU_ADDRESSING

  cudaStream_t stream = streams[CudaStreamIdx];
  //cudaEvent_t captureDoneEvent[MAX_NUM_DLSCH_SEGMENTS];
  //cudaEvent_t captureDoneEvent[MAX_NUM_DLSCH_SEGMENTS];

  if (!graphCreated[CudaStreamIdx]) {
    printf("Creating the graph for stream %d\n", CudaStreamIdx);
    if(CudaStreamIdx != 0){
      cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]);
    }
    //CHECK(cudaGetLastError());
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
  /////cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
//check_ptr_kernel_easy<<<1,10>>>(2);
//cudaDeviceSynchronize();
//CHECK(cudaGetLastError());
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
    //cd cudaDeviceSynchronize();

     //printf("In stream %d 1: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
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
     //cudaDeviceSynchronize();

     //printf("In stream %d 2: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
    CHECK(cudaGetLastError());
    //cudaDeviceSynchronize();
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
    //cudaDeviceSynchronize();
    //printf("In stream %d 3: Iter = %d, PC_Flag = %d\n", CudaStreamIdx, *iter_ptr, *PC_Flag);
     
    
     }

      // stop recording
    ////cudaStreamEndCapture(stream, &decoderGraphs[CudaStreamIdx]);
    //printf("5\n");
    ////cudaGraphInstantiate(&decoderGraphExec[CudaStreamIdx], decoderGraphs[CudaStreamIdx], NULL, NULL, 0);
    ////graphCreated[CudaStreamIdx] = true;

    // Execute （make sure the first trial finish）
    ////cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    cudaDeviceSynchronize();
    printf("Graphs should be captured\n");
    //cudaStreamSynchronize(stream);
  } else {
    //printf("Are you here???\n");
    // reuse the graph after
    if(CudaStreamIdx != 0){
      //uncomment below if you want streams works in sequence
      cudaStreamWaitEvent(streams[CudaStreamIdx], doneEvent[CudaStreamIdx-1], 0);//cudaEventSynchronize(doneEvent[CudaStreamIdx - 1]); 
    }
    cudaGraphLaunch(decoderGraphExec[CudaStreamIdx], stream);
    cudaEventRecord(doneEvent[CudaStreamIdx], stream);
    //
  }
  //CHECK(cudaGetLastError());
#else
  printf("To be continued ^ ^\n");
#endif
}

extern "C" bool is_device_pointer(const void* p) {
    if (p == NULL) return false;
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
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "nrLDPC_types.h"
// #include <cooperative_groups.h>
// amespace cg = cooperative_groups;

#include "nrLDPC_CnProcKernel_BG1_cuda.h"
#include "nrLDPC_BnProcKernel_BG1_cuda.h"

#define Q_SCALE 8.0
#define BG1_GRP0_CN 1
#define ZC 384 // for BG1 test only
#define CPU_ADDRESSING 1 // 0 means copy data into gpu memory, for common gpu; 1 for grace hopper which can read cpu memory directly
#define CUDA_STREAM \
  0 // 1 means use cudastream to run kernels in parallel; for grace hopper, GPU automatically run kernels in parallel
//  so 0 is enough.
#define BIG_KERNEL 1

__constant__ static uint8_t d_lut_numBnInCnGroups_BG1_R13[9];
__constant__ static int d_lut_numThreadsEachCnGroupsNeed_BG1_R13[9];
//__constant__ static uint8_t d_lut_numCnInCnGroups_BG1_R13[9];

// === CUDA Error Checking ===
// Wrap any CUDA API call with CHECK(...) to automatically print error info with file and line number
// Example usage: CHECK(cudaMalloc(&ptr, size));
#define CHECK(call) ErrorCheck((call), __FILE__, __LINE__)

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
  //const uint8_t h_lut_numBnInCnGroups_BG1_R13[] = {3, 4, 5, 6, 7, 8, 9, 10, 19};
  //const int h_lut_numThreadsEachCnGroupsNeed_BG1_R13[] = {288, 384, 480, 576, 672, 768, 864, 960, 1824};
  //const uint8_t h_lut_numCnInCnGroups_BG1_R13[] = {1, 5, 18, 8, 5, 2, 2, 1, 4};

  // const uint8_t *lut_numCnInCnGroups = (const uint8_t *)p_lut->numCnInCnGroups;
  const uint32_t *lut_startAddrCnGroups = lut_startAddrCnGroups_BG1;

  const int numGroups = 9;

#if BIG_KERNEL

  static const uint8_t h_block_group_ids[50] = {0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8};

  static const uint8_t h_block_CN_idx[50] =    {0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17, 0,
                                                1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3};
  
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
  //printf("bnProcBuf =  %p\n", bnProcBuf);
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
   //CHECK(cudaGetLastError());

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
extern "C" void nrLDPC_cnProc_BG1_cuda(const t_nrLDPC_lut *p_lut, int8_t *cnProcBuf, int8_t *cnProcBufRes, int8_t *bnProcBuf, uint16_t Z)
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
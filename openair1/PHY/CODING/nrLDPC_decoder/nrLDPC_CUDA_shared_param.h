#pragma once
#include <cuda_runtime.h>
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#ifdef __cplusplus
extern "C" {
#endif

#define num_TotalThreads_BG1_R13 30336
#define num_TotalThreads_BG1_R23 13824
#define num_TotalThreads_llr_llrRes 6528
#define RowLength 96 //Zc = 384/4 = 96

#define num_TotalBlocks_BG1_R13 316
#define num_TotalBlocks_BG1_R23 144
#define num_TotalBlocks_llr_llrRes 68

#define DUMPCNBN 0

extern cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS_DL];
extern cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS_DL];
extern bool graphCreated[MAX_NUM_DLSCH_SEGMENTS_DL];

#ifdef __cplusplus
}
#endif

typedef struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
}KernelLaunchConfig;


typedef struct {
    int idxBn;
    int idxCn;
    int preBuf;
    int circShift;
    int8_t dd;
} DumpEntry;

__device__ DumpEntry dumpBuf[316];
__device__ DumpEntry dumpBn[316];

typedef struct {
    int8_t* p_llr_ptr;     
    int8_t* p_out_ptr;      
} ldpc_cuda_bridge_t;
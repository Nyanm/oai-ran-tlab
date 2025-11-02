#pragma once
#include <cuda_runtime.h>
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#ifdef __cplusplus
extern "C" {
#endif

#define num_TotalThreads_BG1_R13 30336
#define num_TotalThreads_BG1_R23 13824
#define RowLength 96 //Zc = 384/4 = 96

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

typedef struct SegmentPack {
    int packIdx;         // pack index 0..packCount-1
    int startSeg;        // global start segment index
    int nSeg;            // number of segments in this pack
    cudaStream_t stream; // stream for this pack
    cudaEvent_t doneEvt; // event to signal when pack finishes
} SegmentPack;

typedef struct ThreadSize {
    int NumBlocks;
    int NumThreads;
}ThreadSize;

extern SegmentPack segmentPacks[MAX_NUM_DLSCH_SEGMENTS_DL];
extern ThreadSize threadSize;
#pragma once
#include <cuda_runtime.h>
#define MAX_NUM_DLSCH_SEGMENTS_DL 132
#ifdef __cplusplus
extern "C" {
#endif

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

extern SegmentPack segmentPacks[MAX_NUM_DLSCH_SEGMENTS_DL];
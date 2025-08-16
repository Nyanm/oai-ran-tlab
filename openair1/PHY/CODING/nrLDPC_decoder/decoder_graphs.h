#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaGraph_t decoderGraphs[MAX_NUM_DLSCH_SEGMENTS];
extern cudaGraphExec_t decoderGraphExec[MAX_NUM_DLSCH_SEGMENTS];
extern bool graphCreated[MAX_NUM_DLSCH_SEGMENTS];

#ifdef __cplusplus
}
#endif
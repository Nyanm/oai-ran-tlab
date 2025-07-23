#ifndef __OAI_CUDA_H__
#define __OAI_CUDA_H__
#include "SIMULATION/TOOLS/sim.h"

// The __cplusplus macro is defined by C++ compilers.
// We use it to add the extern "C" block only when compiling with C++.
// A pure C compiler will ignore this block completely, fixing the error.
#ifdef __cplusplus
extern "C" {
#endif

// Original function for reference (can be removed later)
void multipath_channel_cuda(channel_desc_t *desc,
                            float **tx_sig_re,
                            float **tx_sig_im,
                            float **rx_sig_re,
                            float **rx_sig_im,
                            uint32_t length);

// NEW: High-performance version that uses pre-allocated GPU memory
void multipath_channel_cuda_fast(channel_desc_t *desc,
                                 float **tx_sig_re,
                                 float **tx_sig_im,
                                 float **rx_sig_re,
                                 float **rx_sig_im,
                                 uint32_t length,
                                 void *d_tx_sig,
                                 void *d_channel,
                                 void *d_rx_sig);


#ifdef __cplusplus
}
#endif

#endif

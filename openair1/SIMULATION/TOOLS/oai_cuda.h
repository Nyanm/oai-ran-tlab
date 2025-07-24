#ifndef __OAI_CUDA_H__
#define __OAI_CUDA_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NEW, DECOUPLED FUNCTION SIGNATURE
void multipath_channel_cuda_fast(
    // Input Signal Buffers (Host Pointers)
    float **tx_sig_re,
    float **tx_sig_im,
    // Output Signal Buffers (Host Pointers)
    float **rx_sig_re,
    float **rx_sig_im,
    // Simulation & Channel Parameters
    int nb_tx,
    int nb_rx,
    int channel_length,
    uint32_t length,
    uint64_t channel_offset,
    float path_loss,
    // Host pointer to the flattened channel coefficients
    float *h_channel_coeffs,
    // Pre-allocated GPU memory (Device Pointers)
    void *d_tx_sig,
    void *d_rx_sig
);


#ifdef __cplusplus
}
#endif

#endif // __OAI_CUDA_H__

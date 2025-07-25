#ifndef __OAI_CUDA_H__
#define __OAI_CUDA_H__

#include <stdint.h>

// The __NVCC__ macro is defined ONLY when the CUDA compiler (nvcc) is running.
// It is NOT defined for standard C/C++ compilers like gcc/g++.
#ifdef __NVCC__
    // If we are compiling a .cu file, provide the minimal c16_t definition.
    // This is needed for the host-side wrapper functions within the .cu file.
    typedef struct complex16 {
      int16_t r;
      int16_t i;
    } c16_t;
#else
    // If we are compiling a .c file (like dlsim.c), just include the official
    // OAI header that provides the c16_t type. This avoids redefinition.
    #include "PHY/TOOLS/tools_defs.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

// --- Multipath Channel Function ---
void multipath_channel_cuda_fast(
    float **tx_sig_re, float **tx_sig_im,
    float **rx_sig_re, float **rx_sig_im,
    int nb_tx, int nb_rx, int channel_length,
    uint32_t length, uint64_t channel_offset,
    float path_loss,
    float *h_channel_coeffs,
    void *d_tx_sig, void *d_rx_sig
);

// --- AWGN and Phase Noise Function ---
void add_noise_cuda_fast(
    const float **r_re,
    const float **r_im,
    c16_t **output_signal,
    int num_samples,
    int nb_rx,
    float sigma2,
    double ts,
    int slot_offset,
    int delay,
    uint16_t pdu_bit_map,
    uint16_t ptrs_bit_map,
    void *d_r_sig,
    void *d_output_sig,
    void *d_curand_states
);

// --- Helper functions to manage cuRAND states from C code ---
void* create_and_init_curand_states_cuda(int num_elements, unsigned long long seed);
void destroy_curand_states_cuda(void* d_curand_states);


#ifdef __cplusplus
}
#endif

#endif // __OAI_CUDA_H__

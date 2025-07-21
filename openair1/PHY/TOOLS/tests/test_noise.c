/**
 * Corrected and Optimized Benchmark for add_noise_float (CPU) vs. add_noise_cuda_fast (GPU)
 *
 * This test harness accurately measures and compares the performance of noise generation
 * on the CPU and the GPU. It pre-allocates all required memory (host, device, and pinned)
 * outside the timing loop to ensure the benchmark measures only the function execution time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// OAI Includes
#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/TOOLS/oai_cuda.h" // The clean C interface for our CUDA functions
#include "common/utils/LOG/log.h"
#include "common/utils/utils.h"

// CUDA Runtime API (needed for cudaMalloc, cudaFree, etc.)
#include <cuda_runtime.h>

// OAI boilerplate for configuration
configmodule_interface_t *uniqCfg = NULL;

// OAI boilerplate for exit handling
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

// --- Helper Function ---
// Generates a random float signal to serve as input for the noise functions.
void generate_random_signal(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            // Use simple integer math for speed; statistical properties are not critical for a performance benchmark.
            sig_re[i][j] = (float)((rand() % 20000) - 10000);
            sig_im[i][j] = (float)((rand() % 20000) - 10000);
        }
    }
}

// ====================================================================================
// Main Benchmark Function
// ====================================================================================
int main(int argc, char **argv) {
    
    logInit();
    randominit(0);
    
    // --- Test Parameters ---
    int nb_rx_configs[] = {1, 2, 4, 8};
    int num_samples_configs[] = {30720, 61440, 122880};
    int num_trials = 100;
    float snr_db = 10.0f;

    printf("Starting Noise Generation Benchmark (CPU vs. CUDA)\n");
    printf("Averaging each test case over %d trials.\n", num_trials);
    printf("---------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-15s | %-15s | %-15s | %-15s\n", "Antennas", "Signal Length", "CPU Time (us)", "CUDA Time (us)", "Speedup");
    printf("---------------------------------------------------------------------------------------------\n");

    // Iterate over all test configurations
    for (int r = 0; r < sizeof(nb_rx_configs)/sizeof(int); r++) {
    for (int s = 0; s < sizeof(num_samples_configs)/sizeof(int); s++) {
        
        int nb_rx = nb_rx_configs[r];
        int num_samples = num_samples_configs[s];

        // --- Allocate Host Memory ---
        // Input signals (float)
        float **r_re = malloc(nb_rx * sizeof(float *));
        float **r_im = malloc(nb_rx * sizeof(float *));
        for (int i = 0; i < nb_rx; i++) {
            r_re[i] = malloc(num_samples * sizeof(float));
            r_im[i] = malloc(num_samples * sizeof(float));
        }
        
        // Output signals (c16_t). Allocate as one contiguous block for better memory management.
        c16_t **output_cpu = malloc(nb_rx * sizeof(c16_t *));
        c16_t **output_gpu = malloc(nb_rx * sizeof(c16_t *));
        output_cpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        for (int i = 1; i < nb_rx; i++) {
            output_cpu[i] = output_cpu[0] + i * num_samples;
            output_gpu[i] = output_gpu[0] + i * num_samples;
        }

        // --- Allocate GPU & Pinned Memory ---
        void *d_r_sig, *d_output_sig, *d_curand_states;
        void *h_r_sig_pinned, *h_output_sig_pinned;

        // Device memory
        cudaMalloc(&d_r_sig,   nb_rx * num_samples * sizeof(float2));
        cudaMalloc(&d_output_sig,   nb_rx * num_samples * sizeof(short2));

        // Pinned host memory (for fast, asynchronous transfers)
        cudaMallocHost(&h_r_sig_pinned, nb_rx * num_samples * sizeof(float2));
        cudaMallocHost(&h_output_sig_pinned, nb_rx * num_samples * sizeof(short2));

        // Initialize cuRAND states on the GPU using our C-callable helper function
        int num_rand_elements = nb_rx * num_samples;
        d_curand_states = create_and_init_curand_states_cuda(num_rand_elements, time(NULL));

        // --- Timing variables ---
        double total_cpu_ns = 0;
        double total_gpu_ns = 0;

        // --- Simulation Parameters ---
        double ts = 1.0 / 30.72e6;
        float signal_power = 1.0f; // Assume normalized signal power
        float sigma2 = signal_power / powf(10.0f, snr_db / 10.0f);

        // --- Run Benchmark Trials ---
        for (int t = 0; t < num_trials; t++) {
            generate_random_signal(r_re, r_im, nb_rx, num_samples);
            
            struct timespec start, end;

            // --- Time CPU Run ---
            clock_gettime(CLOCK_MONOTONIC, &start);
            add_noise_float(output_cpu, (const float **)r_re, (const float **)r_im, sigma2, num_samples, 0, ts, 0, 0, 0, nb_rx);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

            // --- Time GPU Run ---
            clock_gettime(CLOCK_MONOTONIC, &start);
            // Call the updated function signature, passing all pre-allocated buffers
            add_noise_cuda_fast(
                (const float **)r_re, (const float **)r_im, output_gpu, 
                num_samples, nb_rx, sigma2, ts, 0, 0, 0, 0,
                d_r_sig, d_output_sig, d_curand_states,
                h_r_sig_pinned, h_output_sig_pinned
            );
            // Block until the GPU has finished all work for accurate timing
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        }

        // --- Calculate and Print Results ---
        double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
        double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
        double speedup = (avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
        
        printf("%-15d | %-15d | %-15.2f | %-15.2f | %-15.2fx\n", 
               nb_rx, num_samples, avg_cpu_us, avg_gpu_us, speedup);

        // --- Cleanup All Allocated Memory ---
        for (int i = 0; i < nb_rx; i++) { 
            free(r_re[i]); 
            free(r_im[i]); 
        }
        free(r_re); 
        free(r_im);
        free(output_cpu[0]); 
        free(output_gpu[0]);
        free(output_cpu); 
        free(output_gpu);
        
        cudaFree(d_r_sig); 
        cudaFree(d_output_sig);
        cudaFreeHost(h_r_sig_pinned);
        cudaFreeHost(h_output_sig_pinned);
        destroy_curand_states_cuda(d_curand_states);
    }
    }

    printf("---------------------------------------------------------------------------------------------\n");
    printf("Benchmark finished.\n");

    return 0;
}
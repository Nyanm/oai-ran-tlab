/**
 * @file test_multipath.c
 * @brief Standalone benchmark for comparing CPU and CUDA multipath channel implementations.
 * Updated to use standard Linux timers to be fully independent of the OAI
 * configuration and timing libraries.
 * @author Nika Ghaderi & Gemini
 * @date July 24, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// OAI Includes (only what is absolutely necessary)
#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/TOOLS/oai_cuda.h"
#include "common/utils/LOG/log.h"
#include "common/utils/utils.h" // For randominit

// CUDA Runtime API
#include <cuda_runtime.h>

// The OAI config system is not used, so this can be NULL.
configmodule_interface_t *uniqCfg = NULL;

// Provide a definition for the exit_function that some OAI libraries might still require on link.
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

// ====================================================================================
// Helper Functions
// ====================================================================================

void generate_random_signal(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
            sig_im[i][j] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        }
    }
}

// Manually create a channel descriptor to avoid OAI config dependencies
channel_desc_t* create_manual_channel_desc(int nb_tx, int nb_rx, int channel_length) {
    channel_desc_t* desc = (channel_desc_t*)calloc(1, sizeof(channel_desc_t));
    if (!desc) return NULL;

    desc->nb_tx = nb_tx;
    desc->nb_rx = nb_rx;
    desc->channel_length = channel_length;
    desc->path_loss_dB = 10.0;
    desc->channel_offset = 0;

    int num_links = nb_tx * nb_rx;
    desc->ch = (struct complexd**)malloc(num_links * sizeof(struct complexd*));
    for (int i = 0; i < num_links; i++) {
        desc->ch[i] = (struct complexd*)malloc(channel_length * sizeof(struct complexd));
        for (int l = 0; l < channel_length; l++) {
            desc->ch[i][l].r = (double)rand() / (double)RAND_MAX;
            desc->ch[i][l].i = (double)rand() / (double)RAND_MAX;
        }
    }
    return desc;
}

void free_manual_channel_desc(channel_desc_t* desc) {
    if (!desc) return;
    int num_links = desc->nb_tx * desc->nb_rx;
    for (int i = 0; i < num_links; i++) {
        free(desc->ch[i]);
    }
    free(desc->ch);
    free(desc);
}


int verify_results(float **re_cpu, float **im_cpu, float **re_gpu, float **im_gpu, int nb_rx, int num_samples) {
    double total_error = 0.0;
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            double err_re = re_cpu[i][j] - re_gpu[i][j];
            double err_im = im_cpu[i][j] - im_gpu[i][j];
            total_error += (err_re * err_re) + (err_im * err_im);
        }
    }
    double mse = total_error / (nb_rx * num_samples);
    return (mse < 1e-9) ? 0 : 1;
}


// ====================================================================================
// Main Benchmark Function
// ====================================================================================

int main(int argc, char **argv) {
    
    logInit();
    randominit(0);
    
    // --- Test Parameters ---
    int nb_tx_configs[] = {1, 2, 4};
    int nb_rx_configs[] = {1, 2, 4};
    int num_samples_configs[] = {30720, 61440, 122880};
    int channel_length_configs[] = {16, 32};
    char* channel_type_names[] = {"Short Channel", "Long Channel"};
    int num_trials = 100;

    printf("Starting Multipath Channel Benchmark (CPU vs. CUDA)\n");
    printf("Averaging each test case over %d trials.\n", num_trials);
    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-15s | %-15s | %-15s | %-15s | %-15s | %-10s\n", "Channel Type", "MIMO Config", "Signal Length", "CPU Time (us)", "CUDA Time (us)", "Speedup", "Verification");
    printf("----------------------------------------------------------------------------------------------------------------------\n");

    for (int c = 0; c < sizeof(channel_length_configs)/sizeof(int); c++) {
        for (int s = 0; s < sizeof(num_samples_configs)/sizeof(int); s++) {
            for (int m = 0; m < sizeof(nb_tx_configs)/sizeof(int); m++) {
                
                int nb_tx = nb_tx_configs[m];
                int nb_rx = nb_rx_configs[m];
                int num_samples = num_samples_configs[s];
                int channel_length = channel_length_configs[c];

                char mimo_str[16];
                sprintf(mimo_str, "%dx%d", nb_tx, nb_rx);

                channel_desc_t *chan_desc = create_manual_channel_desc(nb_tx, nb_rx, channel_length);

                float **s_re = malloc(nb_tx * sizeof(float *));
                float **s_im = malloc(nb_tx * sizeof(float *));
                float **r_re_cpu = malloc(nb_rx * sizeof(float *));
                float **r_im_cpu = malloc(nb_rx * sizeof(float *));
                float **r_re_gpu = malloc(nb_rx * sizeof(float *));
                float **r_im_gpu = malloc(nb_rx * sizeof(float *));

                for (int i=0; i<nb_tx; i++) { s_re[i] = malloc(num_samples * sizeof(float)); s_im[i] = malloc(num_samples * sizeof(float)); }
                for (int i=0; i<nb_rx; i++) { r_re_cpu[i] = malloc(num_samples * sizeof(float)); r_im_cpu[i] = malloc(num_samples * sizeof(float)); r_re_gpu[i] = malloc(num_samples * sizeof(float)); r_im_gpu[i] = malloc(num_samples * sizeof(float)); }

                void *d_tx_sig, *d_rx_sig;
                int num_conv_samples = num_samples - chan_desc->channel_offset;
                cudaMalloc(&d_tx_sig,   nb_tx * num_conv_samples * sizeof(float2));
                cudaMalloc(&d_rx_sig,   nb_rx * num_conv_samples * sizeof(float2));

                double total_cpu_ns = 0;
                double total_gpu_ns = 0;
                int verification_passed = 1;

                float path_loss = (float)pow(10, chan_desc->path_loss_dB / 20.0);
                int num_links = chan_desc->nb_tx * chan_desc->nb_rx;
                int channel_size_bytes = num_links * chan_desc->channel_length * sizeof(float2);
                float* h_channel_coeffs = (float*)malloc(channel_size_bytes);
                for (int link = 0; link < num_links; link++) {
                    for (int l = 0; l < chan_desc->channel_length; l++) {
                        int idx = link * chan_desc->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)chan_desc->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)chan_desc->ch[link][l].i;
                    }
                }

                for (int t = 0; t < num_trials; t++) {
                    struct timespec start, end;
                    generate_random_signal(s_re, s_im, nb_tx, num_samples);
                    
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    multipath_channel_float(chan_desc, s_re, s_im, r_re_cpu, r_im_cpu, num_samples, 1, 0);
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
                    
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    multipath_channel_cuda_fast(s_re, s_im, r_re_gpu, r_im_gpu, nb_tx, nb_rx, channel_length, num_samples, chan_desc->channel_offset, path_loss, h_channel_coeffs, d_tx_sig, d_rx_sig);
                    cudaDeviceSynchronize();
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
                }

                if (verify_results(r_re_cpu, r_im_cpu, r_re_gpu, r_im_gpu, nb_rx, num_samples) != 0) {
                    verification_passed = 0;
                }

                double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
                double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
                double speedup = (avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
                
                printf("%-15s | %-15s | %-15d | %-15.2f | %-15.2f | %-15.2fx | %-10s\n", 
                       channel_type_names[c], mimo_str, num_samples, avg_cpu_us, avg_gpu_us, speedup, 
                       (verification_passed ? "PASSED" : "FAILED"));

                free(h_channel_coeffs);
                for (int i=0; i<nb_tx; i++) { free(s_re[i]); free(s_im[i]); }
                for (int i=0; i<nb_rx; i++) { free(r_re_cpu[i]); free(r_im_cpu[i]); free(r_re_gpu[i]); free(r_im_gpu[i]); }
                free(s_re); free(s_im); free(r_re_cpu); free(r_im_cpu); free(r_re_gpu); free(r_im_gpu);
                cudaFree(d_tx_sig); cudaFree(d_rx_sig);
                free_manual_channel_desc(chan_desc);
            }
        }
    }

    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("Benchmark finished.\n");

    return 0;
}

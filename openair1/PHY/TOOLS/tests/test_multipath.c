/**
 * @file test_multipath.c
 * @brief Standalone benchmark for comparing CPU and CUDA multipath channel implementations.
 * @author Nika Ghaderi & Gemini
 * @date July 23, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// OAI Includes
#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/TOOLS/oai_cuda.h"
#include "common/utils/utils.h"
#include "common/utils/threadPool/thread-pool.h"
#include "common/utils/T/T.h"
#include "common/utils/LOG/log.h"
#include "common/config/config_load_configmodule.h"

// CUDA Runtime API
#include <cuda_runtime.h>

// Define the global config pointer that the OAI libraries expect
configmodule_interface_t *uniqCfg = NULL;

// Provide a definition for the exit_function that the OAI libraries require
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    printf("Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}



// ====================================================================================
// Helper Functions
// ====================================================================================

void generate_random_signal(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (float)rand() / (float)RAND_MAX;
            sig_im[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
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
    T_Config_Init();
    if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == NULL) {
        return -1;
    }
    randominit(0);
    
    // --- Comprehensive Test Parameters ---
    int nb_tx_configs[] = {1, 2, 4};
    int nb_rx_configs[] = {1, 2, 4};
    int num_samples_configs[] = {30720, 61440, 122880}; // 1, 2, 4 slots
    SCM_t channel_models[] = {AWGN, TDL_A};
    char* channel_names[] = {"AWGN", "TDL-A"};
    int num_trials = 100;

    printf("Starting Multipath Channel Benchmark (CPU vs. CUDA)\n");
    printf("Averaging each test case over %d trials.\n", num_trials);
    printf("------------------------------------------------------------------------------------------------------------------\n");
    printf("%-12s | %-15s | %-15s | %-15s | %-15s | %-15s | %-10s\n", "Channel", "MIMO Config", "Signal Length", "CPU Time (us)", "CUDA Time (us)", "Speedup", "Verification");
    printf("------------------------------------------------------------------------------------------------------------------\n");

    for (int c = 0; c < sizeof(channel_models)/sizeof(SCM_t); c++) {
        for (int s = 0; s < sizeof(num_samples_configs)/sizeof(int); s++) {
            for (int m = 0; m < sizeof(nb_tx_configs)/sizeof(int); m++) {
                
                int nb_tx = nb_tx_configs[m];
                int nb_rx = nb_rx_configs[m];
                int num_samples = num_samples_configs[s];

                char mimo_str[16];
                sprintf(mimo_str, "%dx%d", nb_tx, nb_rx);

                channel_desc_t *chan_desc = new_channel_desc_scm(nb_tx, nb_rx, channel_models[c], 30.72, 0, 0, 0, 0, 0, 0, 0, 0, 0);

                float **s_re = malloc(nb_tx * sizeof(float *));
                float **s_im = malloc(nb_tx * sizeof(float *));
                float **r_re_cpu = malloc(nb_rx * sizeof(float *));
                float **r_im_cpu = malloc(nb_rx * sizeof(float *));
                float **r_re_gpu = malloc(nb_rx * sizeof(float *));
                float **r_im_gpu = malloc(nb_rx * sizeof(float *));

                for (int i=0; i<nb_tx; i++) { s_re[i] = malloc(num_samples * sizeof(float)); s_im[i] = malloc(num_samples * sizeof(float)); }
                for (int i=0; i<nb_rx; i++) { r_re_cpu[i] = malloc(num_samples * sizeof(float)); r_im_cpu[i] = malloc(num_samples * sizeof(float)); r_re_gpu[i] = malloc(num_samples * sizeof(float)); r_im_gpu[i] = malloc(num_samples * sizeof(float)); }

                void *d_tx_sig, *d_channel, *d_rx_sig;
                int num_conv_samples = num_samples - chan_desc->channel_offset;
                cudaMalloc(&d_tx_sig,   nb_tx * num_conv_samples * sizeof(float) * 2);
                cudaMalloc(&d_channel,  nb_tx * nb_rx * chan_desc->channel_length * sizeof(float) * 2);
                cudaMalloc(&d_rx_sig,   nb_rx * num_conv_samples * sizeof(float) * 2);

                time_stats_t cpu_stats = {0}, gpu_stats = {0};
                int verification_passed = 1;

                for (int t = 0; t < num_trials; t++) {
                    generate_random_signal(s_re, s_im, nb_tx, num_samples);
                    start_meas(&cpu_stats);
                    multipath_channel_float(chan_desc, s_re, s_im, r_re_cpu, r_im_cpu, num_samples, 0, 0);
                    stop_meas(&cpu_stats);
                    start_meas(&gpu_stats);
                    multipath_channel_cuda_fast(chan_desc, s_re, s_im, r_re_gpu, r_im_gpu, num_samples, d_tx_sig, d_channel, d_rx_sig);
                    stop_meas(&gpu_stats);
                    if (t == num_trials - 1) {
                        if (verify_results(r_re_cpu, r_im_cpu, r_re_gpu, r_im_gpu, nb_rx, num_samples) != 0) {
                            verification_passed = 0;
                        }
                    }
                }

                double avg_cpu_time = cpu_stats.p_time / cpu_stats.trials;
                double avg_gpu_time = gpu_stats.p_time / gpu_stats.trials;
                double speedup = avg_cpu_time / avg_gpu_time;
                
                printf("%-12s | %-15s | %-15d | %-15.2f | %-15.2f | %-15.2fx | %-10s\n", 
                       channel_names[c], mimo_str, num_samples, avg_cpu_time, avg_gpu_time, speedup, 
                       (verification_passed ? "PASSED" : "FAILED"));

                for (int i=0; i<nb_tx; i++) { free(s_re[i]); free(s_im[i]); }
                for (int i=0; i<nb_rx; i++) { free(r_re_cpu[i]); free(r_im_cpu[i]); free(r_re_gpu[i]); free(r_im_gpu[i]); }
                free(s_re); free(s_im); free(r_re_cpu); free(r_im_cpu); free(r_re_gpu); free(r_im_gpu);
                cudaFree(d_tx_sig); cudaFree(d_channel); cudaFree(d_rx_sig);
                free_channel_desc_scm(chan_desc);
            }
        }
    }

    printf("------------------------------------------------------------------------------------------------------------------\n");
    printf("Benchmark finished.\n");

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include "common/utils/LOG/log.h"
#include "common/utils/utils.h"
#include "SIMULATION/TOOLS/sim.h"

#ifdef ENABLE_CUDA
#include "SIMULATION/TOOLS/oai_cuda.h"
#include <cuda_runtime.h>
#endif

configmodule_interface_t *uniqCfg = NULL;

typedef enum {
    MODE_GPU_ONLY,
    MODE_CPU_ONLY,
    MODE_ALL
} run_mode_t;

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

void generate_random_float_signal(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (float)((rand() % 2000) - 1000);
            sig_im[i][j] = (float)((rand() % 2000) - 1000);
        }
    }
}

int main(int argc, char **argv) {
    int num_channels = 1;
    int n_tx = 4;
    int n_rx = 4;
    double fs = 122.88e6;
    int mu = 1;
    run_mode_t mode = MODE_ALL;
    int num_trials = 50;

    struct option long_options[] = {
        {"num-channels", required_argument, 0, 'c'},
        {"n-tx",         required_argument, 0, 't'},
        {"n-rx",         required_argument, 0, 'r'},
        {"fs",           required_argument, 0, 'f'},
        {"mu",           required_argument, 0, 'm'},
        {"mode",         required_argument, 0, 'd'},
        {"trials",       required_argument, 0, 'n'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:t:r:f:m:d:n:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'c': num_channels = atoi(optarg); break;
            case 't': n_tx = atoi(optarg); break;
            case 'r': n_rx = atoi(optarg); break;
            case 'f': fs = atof(optarg); break;
            case 'm': mu = atoi(optarg); break;
            case 'd':
                if (strcmp(optarg, "cpu") == 0) mode = MODE_CPU_ONLY;
                else if (strcmp(optarg, "gpu") == 0) mode = MODE_GPU_ONLY;
                else if (strcmp(optarg, "all") == 0) mode = MODE_ALL;
                break;
            case 'n': num_trials = atoi(optarg); break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  -c, --num-channels <N>   Number of parallel channels to simulate (Default: 1)\n");
                printf("  -t, --n-tx <N>           Number of transmit antennas per channel (Default: 4)\n");
                printf("  -r, --n-rx <N>           Number of receive antennas per channel (Default: 4)\n");
                printf("  -f, --fs <Hz>            Sampling frequency in Hz (Default: 122.88e6)\n");
                printf("  -m, --mu <N>             5G NR Numerology (Default: 1)\n");
                printf("  -d, --mode <cpu|gpu|all> Benchmark mode (Default: all)\n");
                printf("  -n, --trials <N>         Number of trials for averaging (Default: 50)\n");
                printf("  -h, --help               Show this help message\n");
                return 0;
            default: exit(1);
        }
    }

    logInit();
    randominit(0);
    int num_samples = (int)(pow(2.0, -mu) * fs * 0.001);

    printf("--- Channel Scalability Benchmark ---\n");
    printf("Averaging over %d trials.\n", num_trials);
    printf("Configuration per trial:\n");
    printf("  Run Mode          : %s\n", (mode == MODE_CPU_ONLY) ? "CPU Only" : (mode == MODE_GPU_ONLY) ? "GPU Only" : "CPU vs GPU");
    printf("  Parallel Channels : %d\n", num_channels);
    printf("  MIMO Configuration: %dx%d\n", n_tx, n_rx);
    printf("  Samples per Slot  : %d\n", num_samples);
    printf("---------------------------------------\n");

    // --- Common Memory Allocation & Setup ---
    float **s_re = malloc(n_tx * sizeof(float *));
    float **s_im = malloc(n_tx * sizeof(float *));
    for (int i = 0; i < n_tx; i++) {
        s_re[i] = malloc(num_samples * sizeof(float));
        s_im[i] = malloc(num_samples * sizeof(float));
    }
    
    channel_desc_t **channels = malloc(num_channels * sizeof(channel_desc_t*));
    for (int c = 0; c < num_channels; c++) {
        channels[c] = new_channel_desc_scm(n_tx, n_rx, TDL_A, fs/1e6, 0, 0, 0.03, 0, 0, 0, 0, 0, 0);
    }

    double total_cpu_ns = 0;
    double total_gpu_ns = 0;
    struct timespec start, end;

#ifdef ENABLE_CUDA
    // Declare all GPU pointers and initialize to NULL
    void *d_tx_sig = NULL, *d_intermediate_sig = NULL, *d_final_output = NULL, 
         *d_curand_states = NULL, *h_tx_sig_pinned = NULL, *h_final_output_pinned = NULL,
         *d_channel_coeffs_gpu = NULL;
    float *h_channel_coeffs = NULL;
    c16_t ***output_gpu = NULL;

    if (mode == MODE_GPU_ONLY || mode == MODE_ALL) {
        // --- GPU ONE-TIME SETUP ---
        output_gpu = malloc(num_channels * sizeof(c16_t**));
        for(int c = 0; c < num_channels; ++c) {
            output_gpu[c] = malloc(n_rx * sizeof(c16_t*));
            for (int i = 0; i < n_rx; i++) {
                output_gpu[c][i] = malloc(num_samples * sizeof(c16_t));
            }
        }
        
        const int max_taps = 256;
        h_channel_coeffs = malloc(n_tx * n_rx * max_taps * sizeof(float) * 2);
        
        #if defined(USE_UNIFIED_MEMORY)
            cudaMallocManaged(&h_tx_sig_pinned, n_tx * num_samples * sizeof(float2), cudaMemAttachGlobal);
            cudaMallocManaged(&d_intermediate_sig, n_rx * num_samples * sizeof(float2), cudaMemAttachGlobal);
            cudaMallocManaged(&h_final_output_pinned, n_rx * num_samples * sizeof(short2), cudaMemAttachGlobal);
            d_tx_sig = h_tx_sig_pinned; d_final_output = h_final_output_pinned;
        #elif defined(USE_ATS_MEMORY)
            h_tx_sig_pinned = malloc(n_tx * num_samples * sizeof(float2));
            cudaMalloc(&d_intermediate_sig, n_rx * num_samples * sizeof(float2));
            cudaMalloc(&d_final_output, n_rx * num_samples * sizeof(short2));
            h_final_output_pinned = malloc(n_rx * num_samples * sizeof(short2));
            d_tx_sig = NULL;
        #else // Default explicit copy method
            cudaMalloc(&d_tx_sig, n_tx * num_samples * sizeof(float2));
            cudaMalloc(&d_intermediate_sig, n_rx * num_samples * sizeof(float2));
            cudaMalloc(&d_final_output, n_rx * num_samples * sizeof(short2));
            cudaMallocHost(&h_tx_sig_pinned, n_tx * num_samples * sizeof(float2));
            cudaMallocHost(&h_final_output_pinned, n_rx * num_samples * sizeof(short2));
        #endif
        cudaMalloc(&d_channel_coeffs_gpu, n_tx * n_rx * max_taps * sizeof(float2));
        d_curand_states = create_and_init_curand_states_cuda(n_rx * num_samples, time(NULL));
    }
#endif

    // --- MAIN TIMING LOOP ---
    for (int t = 0; t < num_trials; t++) {
        generate_random_float_signal(s_re, s_im, n_tx, num_samples);
        for (int c = 0; c < num_channels; c++) {
            random_channel(channels[c], 0);
        }

        if (mode == MODE_CPU_ONLY || mode == MODE_ALL) {
            float **r_re_cpu = malloc(n_rx * sizeof(float *));
            float **r_im_cpu = malloc(n_rx * sizeof(float *));
            c16_t ***output_cpu = malloc(num_channels * sizeof(c16_t**));
            for(int i=0; i < n_rx; ++i) { r_re_cpu[i] = malloc(num_samples * sizeof(float)); r_im_cpu[i] = malloc(num_samples * sizeof(float)); }
            for(int c=0; c < num_channels; ++c) {
                output_cpu[c] = malloc(n_rx * sizeof(c16_t*));
                for (int i = 0; i < n_rx; i++) {
                    output_cpu[c][i] = malloc(num_samples * sizeof(c16_t));
                }
            }
            
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int c = 0; c < num_channels; c++) {
                multipath_channel_float(channels[c], s_re, s_im, r_re_cpu, r_im_cpu, num_samples, 1, 0);
                add_noise_float(output_cpu[c], (const float **)r_re_cpu, (const float **)r_im_cpu, 1.0f, num_samples, 0, 1.0/fs, 0, 0, 0, n_rx);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_cpu_ns += ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec));

            for(int i=0; i < n_rx; ++i) { free(r_re_cpu[i]); free(r_im_cpu[i]); }
            for(int c=0; c < num_channels; ++c) { for (int i=0; i<n_rx; i++) free(output_cpu[c][i]); free(output_cpu[c]); }
            free(r_re_cpu); free(r_im_cpu); free(output_cpu);
        }

#ifdef ENABLE_CUDA
        if (mode == MODE_GPU_ONLY || mode == MODE_ALL) {
            #if defined(USE_UNIFIED_MEMORY)
                float2* managed_tx_sig = (float2*)h_tx_sig_pinned;
                for (int aa = 0; aa < n_tx; aa++) {
                    for (int i = 0; i < num_samples; i++) {
                        managed_tx_sig[aa * num_samples + i] = make_float2(s_re[aa][i], s_im[aa][i]);
                    }
                }
            #elif defined(USE_ATS_MEMORY)
                float2* ats_input_buffer = (float2*)h_tx_sig_pinned;
                for (int aa = 0; aa < n_tx; aa++) {
                    for (int i = 0; i < num_samples; i++) {
                        ats_input_buffer[aa * num_samples + i] = make_float2(s_re[aa][i], s_im[aa][i]);
                    }
                }
            #endif

            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int c = 0; c < num_channels; c++) {
                float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                for (int link = 0; link < n_tx * n_rx; link++) {
                    for (int l = 0; l < channels[c]->channel_length; l++) {
                        int idx = link * channels[c]->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
                run_channel_pipeline_cuda(s_re, s_im, output_gpu[c], n_tx, n_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 1.0f, 1.0/fs, 0, 0, 0, 0, d_tx_sig, d_intermediate_sig, d_final_output, d_curand_states, h_tx_sig_pinned, h_final_output_pinned, d_channel_coeffs_gpu);
            }
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_gpu_ns += ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec));
        }
#endif
    }
    
    // --- FINAL REPORT ---
    double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
    double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
    
    printf("\n--- Results ---\n");
    if (mode == MODE_CPU_ONLY || mode == MODE_ALL) {
        printf("Avg Total CPU Time for %d channels: %.2f us\n", num_channels, avg_cpu_us);
    }
#ifdef ENABLE_CUDA
    if (mode == MODE_GPU_ONLY || mode == MODE_ALL) {
        double avg_gpu_per_chan_us = avg_gpu_us / num_channels;
        printf("Avg Total GPU Time for %d channels: %.2f us\n", num_channels, avg_gpu_us);
        printf("Avg GPU Time per Channel:         %.2f us\n", avg_gpu_per_chan_us);
        printf("Real-time Target (< 500 us):      %s\n", (avg_gpu_per_chan_us < 500.0) ? "PASS" : "FAIL");
    }
    if (mode == MODE_ALL) {
        printf("Speedup (Total CPU/Total GPU):    %.2fx\n", avg_cpu_us / avg_gpu_us);
    }
#endif
    printf("---------------------------------------\n");

    // --- FINAL CLEANUP ---
#ifdef ENABLE_CUDA
    if (mode == MODE_GPU_ONLY || mode == MODE_ALL) {
        #if defined(USE_UNIFIED_MEMORY)
            cudaFree(h_tx_sig_pinned); cudaFree(d_intermediate_sig); cudaFree(h_final_output_pinned);
        #elif defined(USE_ATS_MEMORY)
            free(h_tx_sig_pinned); cudaFree(d_intermediate_sig); cudaFree(d_final_output); free(h_final_output_pinned);
        #else
            cudaFree(d_tx_sig); cudaFree(d_intermediate_sig); cudaFree(d_final_output); cudaFreeHost(h_tx_sig_pinned); cudaFreeHost(h_final_output_pinned);
        #endif
        cudaFree(d_channel_coeffs_gpu);
        destroy_curand_states_cuda(d_curand_states);
        
        for(int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < n_rx; i++) {
                free(output_gpu[c][i]);
            }
            free(output_gpu[c]);
        }
        free(output_gpu);
        free(h_channel_coeffs);
    }
#endif
    for (int i = 0; i < n_tx; i++) { free(s_re[i]); free(s_im[i]); }
    free(s_re); free(s_im);
    for(int i=0; i < num_channels; ++i) free_channel_desc_scm(channels[i]);
    free(channels);
    
    return 0;
}
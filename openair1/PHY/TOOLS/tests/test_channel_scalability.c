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

    struct option long_options[] = {
        {"num-channels", required_argument, 0, 'c'},
        {"n-tx",         required_argument, 0, 't'},
        {"n-rx",         required_argument, 0, 'r'},
        {"fs",           required_argument, 0, 'f'},
        {"mu",           required_argument, 0, 'm'},
        {"mode",         required_argument, 0, 'd'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:t:r:f:m:d:h", long_options, NULL)) != -1) {
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
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -c, --num-channels <N>   Number of parallel channels to simulate (Default: 1)\n");
                printf("  -t, --n-tx <N>           Number of transmit antennas per channel (Default: 4)\n");
                printf("  -r, --n-rx <N>           Number of receive antennas per channel (Default: 4)\n");
                printf("  -f, --fs <Hz>            Sampling frequency in Hz (Default: 122.88e6)\n");
                printf("  -m, --mu <N>             5G NR Numerology (Default: 1)\n");
                printf("  -d, --mode <cpu|gpu|all> Benchmark mode (Default: all)\n");
                printf("  -h, --help               Show this help message\n");
                return 0;
            default: exit(1);
        }
    }

    logInit();
    randominit(0);

    int num_samples = (int)(pow(2.0, -mu) * fs * 0.001);

    printf("--- Channel Scalability Benchmark ---\n");
    printf("Configuration:\n");
    printf("  Run Mode          : %s\n", (mode == MODE_CPU_ONLY) ? "CPU Only" : (mode == MODE_GPU_ONLY) ? "GPU Only" : "CPU vs GPU");
    printf("  Parallel Channels : %d\n", num_channels);
    printf("  MIMO Configuration: %dx%d\n", n_tx, n_rx);
    printf("  Samples per Slot  : %d\n", num_samples);
    printf("---------------------------------------\n");

    float **s_re = malloc(n_tx * sizeof(float *));
    float **s_im = malloc(n_tx * sizeof(float *));
    for (int i = 0; i < n_tx; i++) {
        s_re[i] = malloc(num_samples * sizeof(float));
        s_im[i] = malloc(num_samples * sizeof(float));
    }
    generate_random_float_signal(s_re, s_im, n_tx, num_samples);
    
    channel_desc_t **channels = malloc(num_channels * sizeof(channel_desc_t*));
    for (int c = 0; c < num_channels; c++) {
        channels[c] = new_channel_desc_scm(n_tx, n_rx, TDL_A, fs/1e6, 0, 0, 0.03, 0, 0, 0, 0, 0, 0);
        random_channel(channels[c], 0);
    }

    double gpu_parallel_total_us = -1.0;

    double gpu_sequential_total_us = -1.0;
    double cpu_total_us = -1.0;
    struct timespec start, end;

#ifdef ENABLE_CUDA
    if (mode == MODE_GPU_ONLY || mode == MODE_ALL) {
        c16_t ***gpu_output_signals = malloc(num_channels * sizeof(c16_t**));
        for(int c = 0; c < num_channels; ++c) {
            gpu_output_signals[c] = malloc(n_rx * sizeof(c16_t*));
            for (int i = 0; i < n_rx; i++) {
                gpu_output_signals[c][i] = malloc(num_samples * sizeof(c16_t));
            }
        }
        
        const int max_taps = 256;
        float *h_channel_coeffs = malloc(n_tx * n_rx * max_taps * sizeof(float) * 2);
        void *d_channel_coeffs_gpu;
        cudaMalloc(&d_channel_coeffs_gpu, n_tx * n_rx * max_taps * sizeof(float2));


        void *d_tx_sig, *d_intermediate_sig, *d_final_output, *d_curand_states, *h_tx_sig_pinned, *h_final_output_pinned;
        // #if defined(USE_UNIFIED_MEMORY)
        //     cudaMallocManaged(&h_tx_sig_pinned, n_tx * num_samples * sizeof(float2), cudaMemAttachGlobal);
        //     cudaMallocManaged(&d_intermediate_sig, n_rx * num_samples * sizeof(float2), cudaMemAttachGlobal);
        //     cudaMallocManaged(&h_final_output_pinned, n_rx * num_samples * sizeof(short2), cudaMemAttachGlobal);
        //     d_tx_sig = h_tx_sig_pinned; d_final_output = h_final_output_pinned;
        // #elif defined(USE_ATS_MEMORY)
        //     h_tx_sig_pinned = malloc(n_tx * num_samples * sizeof(float2));
        //     cudaMalloc(&d_intermediate_sig, n_rx * num_samples * sizeof(float2));
        //     cudaMalloc(&d_final_output, n_rx * num_samples * sizeof(short2));
        //     h_final_output_pinned = malloc(n_rx * num_samples * sizeof(short2));
        //     d_tx_sig = NULL;
        // #else // Default explicit copy method
        //     cudaMalloc(&d_tx_sig, n_tx * num_samples * sizeof(float2));
        //     cudaMalloc(&d_intermediate_sig, n_rx * num_samples * sizeof(float2));
        //     cudaMalloc(&d_final_output, n_rx * num_samples * sizeof(short2));
        //     cudaMallocHost(&h_tx_sig_pinned, n_tx * num_samples * sizeof(float2));
        //     cudaMallocHost(&h_final_output_pinned, n_rx * num_samples * sizeof(short2));
        // #endif



            cudaMalloc(&d_tx_sig, n_tx * num_samples * sizeof(float2));
            cudaMalloc(&d_intermediate_sig, n_rx * num_samples * sizeof(float2));
            cudaMalloc(&d_final_output, n_rx * num_samples * sizeof(short2));
            cudaMallocHost(&h_tx_sig_pinned, n_tx * num_samples * sizeof(float2));
            cudaMallocHost(&h_final_output_pinned, n_rx * num_samples * sizeof(short2));




        d_curand_states = create_and_init_curand_states_cuda(n_rx * num_samples, time(NULL));

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int c = 0; c < num_channels; c++) {
            float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
            AssertFatal(channels[c]->channel_length <= max_taps, "Channel length exceeds allocated buffer size!");
            for (int link = 0; link < n_tx * n_rx; link++) {
                for (int l = 0; l < channels[c]->channel_length; l++) {
                    int idx = link * channels[c]->channel_length + l;
                    ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                    ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                }
            }
            run_channel_pipeline_cuda(s_re, s_im, gpu_output_signals[c], n_tx, n_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 1.0f, 1.0/fs, 0, 0, 0, 0, d_tx_sig, d_intermediate_sig, d_final_output, d_curand_states, h_tx_sig_pinned, h_final_output_pinned, d_channel_coeffs_gpu);
        }
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        gpu_parallel_total_us = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / 1000.0;






// --- 2. Sequential GPU Benchmark ---
        double total_sequential_ns = 0;
        for (int c = 0; c < num_channels; c++) {
            float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
            clock_gettime(CLOCK_MONOTONIC, &start);
            run_channel_pipeline_cuda(s_re, s_im, gpu_output_signals[c], n_tx, n_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 1.0f, 1.0/fs, 0, 0, 0, 0, d_tx_sig, d_intermediate_sig, d_final_output, d_curand_states, h_tx_sig_pinned, h_final_output_pinned, d_channel_coeffs_gpu);
            cudaDeviceSynchronize(); // Synchronize after EACH launch
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_sequential_ns += ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec));
        }
        gpu_sequential_total_us = total_sequential_ns / 1000.0;













        for(int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < n_rx; i++) {
                free(gpu_output_signals[c][i]);
            }
            free(gpu_output_signals[c]);
        }
        cudaFree(d_channel_coeffs_gpu);
        free(gpu_output_signals); 
        free(h_channel_coeffs);
        
        // #if defined(USE_UNIFIED_MEMORY)
        //     cudaFree(h_tx_sig_pinned); cudaFree(d_intermediate_sig); cudaFree(h_final_output_pinned);
        // #elif defined(USE_ATS_MEMORY)
        //     free(h_tx_sig_pinned); cudaFree(d_intermediate_sig); cudaFree(d_final_output); free(h_final_output_pinned);
        // #else
        //     cudaFree(d_tx_sig); cudaFree(d_intermediate_sig); cudaFree(d_final_output); cudaFreeHost(h_tx_sig_pinned); cudaFreeHost(h_final_output_pinned);
        // #endif




cudaFree(d_tx_sig); cudaFree(d_intermediate_sig); cudaFree(d_final_output); cudaFreeHost(h_tx_sig_pinned); cudaFreeHost(h_final_output_pinned);
    



        destroy_curand_states_cuda(d_curand_states);
    }
#endif

    if (mode == MODE_CPU_ONLY || mode == MODE_ALL) {
        float **r_re = malloc(n_rx * sizeof(float *));
        float **r_im = malloc(n_rx * sizeof(float *));
        c16_t ***cpu_output_signals = malloc(num_channels * sizeof(c16_t**));
        for(int i=0; i < n_rx; ++i) { r_re[i] = malloc(num_samples * sizeof(float)); r_im[i] = malloc(num_samples * sizeof(float)); }

        for(int c=0; c < num_channels; ++c) {
            cpu_output_signals[c] = malloc(n_rx * sizeof(c16_t*));
            for (int i = 0; i < n_rx; i++) {
                cpu_output_signals[c][i] = malloc(num_samples * sizeof(c16_t));
            }
        }        

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int c = 0; c < num_channels; c++) {
            multipath_channel_float(channels[c], s_re, s_im, r_re, r_im, num_samples, 1, 0);
            add_noise_float(cpu_output_signals[c], (const float **)r_re, (const float **)r_im, 1.0f, num_samples, 0, 1.0/fs, 0, 0, 0, n_rx);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        cpu_total_us = ((end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec)) / 1000.0;

        for(int i=0; i < n_rx; ++i) { free(r_re[i]); free(r_im[i]); }
        for(int c = 0; c < num_channels; ++c) {
            for (int i = 0; i < n_rx; i++) {
                free(cpu_output_signals[c][i]);
            }
            free(cpu_output_signals[c]);
        }
        free(r_re); free(r_im); free(cpu_output_signals);
    }

// --- Final Report ---
    printf("\n--- Results ---\n");
    if (cpu_total_us > 0) {
        printf("Total CPU Time for %d channels:     %.2f us\n", num_channels, cpu_total_us);
    }
#ifdef ENABLE_CUDA
    if (gpu_parallel_total_us > 0) {
        double avg_concurrent_us = gpu_parallel_total_us / num_channels;
        printf("Total GPU Concurrent Time for %d channels: %.2f us\n", num_channels, gpu_parallel_total_us);
        printf("Avg GPU Concurrent Time per chan:   %.2f us\n", avg_concurrent_us);
        printf("Real-time Target (< 500 us):      %s\n", (avg_concurrent_us < 500.0) ? "PASS" : "FAIL");
    }
    if (gpu_sequential_total_us > 0) {
        double avg_sequential_us = gpu_sequential_total_us / num_channels;
        printf("Avg GPU Sequential Time per chan:   %.2f us\n", avg_sequential_us);
    }
    if (cpu_total_us > 0 && gpu_parallel_total_us > 0) {
        printf("Speedup (vs Concurrent GPU):        %.2fx\n", cpu_total_us / gpu_parallel_total_us);
    }
#endif
    printf("---------------------------------------\n");

    // --- Final Cleanup ---
    for (int i = 0; i < n_tx; i++) { free(s_re[i]); free(s_im[i]); }
    free(s_re); free(s_im);
    for(int i=0; i < num_channels; ++i) free_channel_desc_scm(channels[i]);
    free(channels);
    
    return 0;
}
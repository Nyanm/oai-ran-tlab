#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <getopt.h>

#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/TOOLS/oai_cuda.h"
#include "common/utils/LOG/log.h"
#include "common/utils/utils.h"
#include <cuda_runtime.h>

configmodule_interface_t *uniqCfg = NULL;

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

void generate_random_signal(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (float)((rand() % 2000) - 1000);
            sig_im[i][j] = (float)((rand() % 2000) - 1000);
        }
    }
}

int main(int argc, char **argv) {
    
    logInit();
    randominit(0);
    
    // --- Configuration with Defaults ---
    int num_channels = 1;
    int nb_tx = 4;
    int nb_rx = 4;
    int num_samples = 30720;
    int channel_length = 32;
    int num_trials = 50;
    float snr_db = 15.0f;

    // --- Argument Parsing ---
    struct option long_options[] = {
        {"num-channels", required_argument, 0, 'c'},
        {"nb-tx",     required_argument, 0, 't'},
        {"nb-rx",     required_argument, 0, 'r'},
        {"num-samples", required_argument, 0, 's'},
        {"ch-len",    required_argument, 0, 'l'},
        {"trials",    required_argument, 0, 'n'},
        {"snr",       required_argument, 0, 'S'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:t:r:s:l:n:S:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'c': num_channels = atoi(optarg); break;
            case 't': nb_tx = atoi(optarg); break;
            case 'r': nb_rx = atoi(optarg); break;
            case 's': num_samples = atoi(optarg); break;
            case 'l': channel_length = atoi(optarg); break;
            case 'n': num_trials = atoi(optarg); break;
            case 'S': snr_db = atof(optarg); break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  -c, --num-channels <N>   Number of parallel channels to simulate (Default: 1)\n");
                printf("  -t, --nb-tx <N>          Number of transmit antennas (Default: 4)\n");
                printf("  -r, --nb-rx <N>          Number of receive antennas (Default: 4)\n");
                printf("  -s, --num-samples <N>    Number of samples (Default: 30720)\n");
                printf("  -l, --ch-len <N>         Channel length (Default: 32)\n");
                printf("  -n, --trials <N>         Number of trials for averaging (Default: 50)\n");
                printf("  -S, --snr <dB>           Signal to Noise Ratio in dB (Default: 15.0)\n");
                printf("  -h, --help               Show this help message\n");
                return 0;
            default: exit(1);
        }
    }

    printf("--- Scalable Channel Pipeline Benchmark ---\n");
    printf("Averaging over %d trials.\n", num_trials);
    printf("Configuration per trial:\n");
    printf("  Parallel Channels : %d\n", num_channels);
    printf("  MIMO Configuration: %dx%d\n", nb_tx, nb_rx);
    printf("  Signal Length     : %d\n", num_samples);
    printf("  Channel Length    : %d\n", channel_length);
    printf("-----------------------------------------------------------\n");

    // --- ONE-TIME MEMORY ALLOCATION ---
    float **tx_sig_re = malloc(nb_tx * sizeof(float *));
    float **tx_sig_im = malloc(nb_tx * sizeof(float *));
    for (int i=0; i<nb_tx; i++) { tx_sig_re[i] = malloc(num_samples * sizeof(float)); tx_sig_im[i] = malloc(num_samples * sizeof(float)); }

    float **rx_multipath_re_cpu = malloc(nb_rx * sizeof(float *));
    float **rx_multipath_im_cpu = malloc(nb_rx * sizeof(float *));
    for (int i=0; i<nb_rx; i++) {
        rx_multipath_re_cpu[i] = malloc(num_samples * sizeof(float));
        rx_multipath_im_cpu[i] = malloc(num_samples * sizeof(float));
    }
    
    channel_desc_t **channels = malloc(num_channels * sizeof(channel_desc_t*));
    for (int c=0; c<num_channels; c++) {
        channels[c] = new_channel_desc_scm(nb_tx, nb_rx, TDL_A, 30.72, 0,0,0.03,0,0,0,0,0,0);
        channels[c]->channel_length = channel_length;
    }

    c16_t ***output_cpu = malloc(num_channels * sizeof(c16_t**));
    c16_t ***output_gpu = malloc(num_channels * sizeof(c16_t**));
    for(int c=0; c<num_channels; c++){
        output_cpu[c] = malloc(nb_rx * sizeof(c16_t*));
        output_gpu[c] = malloc(nb_rx * sizeof(c16_t*));
        for(int i=0; i<nb_rx; i++) {
            output_cpu[c][i] = malloc(num_samples * sizeof(c16_t));
            output_gpu[c][i] = malloc(num_samples * sizeof(c16_t));
        }
    }
    
    void *d_tx_sig=NULL, *d_rx_sig=NULL, *d_output_noise=NULL, *d_curand_states=NULL, 
         *h_tx_sig_pinned=NULL, *h_output_sig_pinned=NULL, *d_channel_coeffs_gpu=NULL;
    float* h_channel_coeffs = NULL;

    const int max_taps = 256;
    cudaMalloc(&d_channel_coeffs_gpu, nb_tx * nb_rx * max_taps * sizeof(float2));
    h_channel_coeffs = malloc(nb_tx * nb_rx * max_taps * sizeof(float2));

    #if defined(USE_UNIFIED_MEMORY)
        cudaMallocManaged(&d_tx_sig, nb_tx * num_samples * sizeof(float2), cudaMemAttachGlobal);
        cudaMallocManaged(&d_rx_sig, nb_rx * num_samples * sizeof(float2), cudaMemAttachGlobal);
        cudaMallocManaged(&d_output_noise, nb_rx * num_samples * sizeof(short2), cudaMemAttachGlobal);
        h_tx_sig_pinned = d_tx_sig;
        h_output_sig_pinned = d_output_noise;
    #else 
        cudaMalloc(&d_tx_sig, nb_tx * num_samples * sizeof(float2));
        cudaMalloc(&d_rx_sig, nb_rx * num_samples * sizeof(float2));
        cudaMalloc(&d_output_noise, nb_rx * num_samples * sizeof(short2));
        cudaMallocHost(&h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2));
        cudaMallocHost(&h_output_sig_pinned, nb_rx * num_samples * sizeof(short2));
    #endif
    d_curand_states = create_and_init_curand_states_cuda(nb_rx * num_samples, time(NULL));

    // --- WARM-UP RUN ---
    printf("Performing GPU warm-up run...\n");
    run_channel_pipeline_cuda(tx_sig_re, tx_sig_im, output_gpu[0], nb_tx, nb_rx, channel_length, num_samples, 0, h_channel_coeffs, 0, 0, 0, 0, 0, 0, d_tx_sig, d_rx_sig, d_output_noise, d_curand_states, h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu);
    cudaDeviceSynchronize();
    printf("Warm-up complete. Starting benchmark...\n\n");
    
    double total_cpu_ns = 0;
    double total_gpu_ns = 0;

    // --- MAIN TIMING LOOP ---
    for (int t = 0; t < num_trials; t++) {
        generate_random_signal(tx_sig_re, tx_sig_im, nb_tx, num_samples);
        for(int c=0; c<num_channels; c++) random_channel(channels[c], 0);

        struct timespec start, end;

        // --- CPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        for(int c=0; c<num_channels; c++){
            multipath_channel_float(channels[c], tx_sig_re, tx_sig_im, rx_multipath_re_cpu, rx_multipath_im_cpu, num_samples, 1, 0);
            add_noise_float(output_cpu[c], (const float **)rx_multipath_re_cpu, (const float **)rx_multipath_im_cpu, 0.1, num_samples, 0, 0, 0, 0, 0, nb_rx);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

        // --- GPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        for(int c=0; c<num_channels; c++){
            float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
            for (int link = 0; link < nb_tx * nb_rx; link++) {
                for (int l = 0; l < channels[c]->channel_length; l++) {
                    int idx = link * channels[c]->channel_length + l;
                    ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                    ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                }
            }
            run_channel_pipeline_cuda(tx_sig_re, tx_sig_im, output_gpu[c], nb_tx, nb_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 0.1, 0, 0, 0, 0, 0, d_tx_sig, d_rx_sig, d_output_noise, d_curand_states, h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu);
        }
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    }
    
    // --- FINAL REPORT ---
    double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
    double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
    double speedup = (avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
    
    printf("\n--- Final Benchmark Results ---\n");
    printf("%-25s | %-25s | %-25s | %-15s\n", "Avg CPU Time (us)", "Avg GPU Time (us)", "Avg GPU Time per Channel (us)", "Overall Speedup");
    printf("------------------------------------------------------------------------------------------------------------------\n");
    printf("%-25.2f | %-25.2f | %-25.2f | %-15.2fx\n", avg_cpu_us, avg_gpu_us, avg_gpu_us / num_channels, speedup);
    

        free(h_channel_coeffs);
        for (int i=0; i<nb_tx; i++) { free(tx_sig_re[i]); free(tx_sig_im[i]); }
        for (int i=0; i<nb_rx; i++) { free(rx_multipath_re_cpu[i]); free(rx_multipath_im_cpu[i]); }
        free(tx_sig_re); free(tx_sig_im);
        free(rx_multipath_re_cpu); free(rx_multipath_im_cpu);
        free(output_cpu[0]); free(output_gpu[0]);
        free(output_cpu); free(output_gpu);

        #if defined(USE_UNIFIED_MEMORY)
            cudaFree(d_tx_sig);
            cudaFree(d_rx_sig);
            cudaFree(d_output_noise);
        #else
            cudaFree(d_tx_sig);
            cudaFree(d_rx_sig);
            cudaFreeHost(h_tx_sig_pinned);
            cudaFree(d_output_noise);
            cudaFreeHost(h_output_sig_pinned);
        #endif
        cudaFree(d_channel_coeffs_gpu); // <-- FREE
        destroy_curand_states_cuda(d_curand_states);
        
        // free_manual_channel_desc(chan_desc);


    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("Pipeline benchmark finished.\n");

    return 0;
}
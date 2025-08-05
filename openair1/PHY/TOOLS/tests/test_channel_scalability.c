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
    int sum_outputs = 0;

    // --- Argument Parsing ---
    struct option long_options[] = {
        {"num-channels", required_argument, 0, 'c'},
        {"nb-tx",     required_argument, 0, 't'},
        {"nb-rx",     required_argument, 0, 'r'},
        {"num-samples", required_argument, 0, 's'},
        {"ch-len",    required_argument, 0, 'l'},
        {"trials",    required_argument, 0, 'n'},
        {"sum-outputs",  no_argument,       0, 'S'},
        {"help",         no_argument,       0, 'h'},
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
            case 'S': sum_outputs = 1; break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  -c, --num-channels <N>   Number of parallel channels to simulate (Default: 1)\n");
                printf("  -t, --nb-tx <N>          Number of transmit antennas (Default: 4)\n");
                printf("  -r, --nb-rx <N>          Number of receive antennas (Default: 4)\n");
                printf("  -s, --num-samples <N>    Number of samples (Default: 30720)\n");
                printf("  -l, --ch-len <N>         Channel length (Default: 32)\n");
                printf("  -n, --trials <N>         Number of trials for averaging (Default: 50)\n");
                printf("  -S, --sum-outputs        Enable summation of outputs for interference simulation (Default: Disabled)\n");
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
    if (sum_outputs) printf("  Mode              : Interference Simulation (Summing Outputs)\n");
    printf("-----------------------------------------------------------\n");

    // --- ONE-TIME MEMORY ALLOCATION ---
    // If sum_outputs is enabled, we need a unique input signal per channel
    int num_tx_signals = sum_outputs ? num_channels : 1;
    float ***tx_sig_re = malloc(num_tx_signals * sizeof(float**));
    float ***tx_sig_im = malloc(num_tx_signals * sizeof(float**));
    for(int i=0; i<num_tx_signals; i++){
        tx_sig_re[i] = malloc(nb_tx * sizeof(float*));
        tx_sig_im[i] = malloc(nb_tx * sizeof(float*));
        for(int j=0; j<nb_tx; j++){
            tx_sig_re[i][j] = malloc(num_samples * sizeof(float));
            tx_sig_im[i][j] = malloc(num_samples * sizeof(float));
        }
    }
    
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
        // Allocate one contiguous block for all antennas
        output_cpu[c][0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        output_gpu[c][0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        // Set pointers for other antennas
        for(int i=1; i<nb_rx; i++) {
            output_cpu[c][i] = output_cpu[c][0] + i * num_samples;
            output_gpu[c][i] = output_gpu[c][0] + i * num_samples;
        }
    }
    
    void *d_tx_sig=NULL, *d_rx_sig=NULL, *d_output_noise=NULL, *d_curand_states=NULL, 
         *h_tx_sig_pinned=NULL, *h_output_sig_pinned=NULL, *d_channel_coeffs_gpu=NULL,
         **d_individual_gpu_outputs = NULL, *d_summed_gpu_output = NULL;
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

    if (sum_outputs) {
        d_individual_gpu_outputs = malloc(num_channels * sizeof(void*));
        for (int c=0; c<num_channels; c++) {
            cudaMalloc(&d_individual_gpu_outputs[c], nb_rx * num_samples * sizeof(short2));
        }
        cudaMalloc(&d_summed_gpu_output, nb_rx * num_samples * sizeof(short2));
    }
    d_curand_states = create_and_init_curand_states_cuda(nb_rx * num_samples, time(NULL));



    // // --- WARM-UP RUN ---
    // printf("Performing GPU warm-up run...\n");
    // // A warm-up just needs to execute the kernels; input data content is not critical.
    // // We pass tx_sig_re[0] and tx_sig_im[0] to match the expected float** type.
    // run_channel_pipeline_cuda(
    //     tx_sig_re[0], tx_sig_im[0], output_gpu[0],
    //     nb_tx, nb_rx, channel_length, num_samples,
    //     0, h_channel_coeffs, 0.1, 0, 0, 0, 0, 0,
    //     d_tx_sig, d_rx_sig, d_output_noise, d_curand_states,
    //     h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu
    // );
    // cudaDeviceSynchronize();
    // printf("Warm-up complete. Starting benchmark...\n\n");


    double total_cpu_ns = 0;
    double total_gpu_ns = 0;

    // --- MAIN TIMING LOOP ---
    for (int t = 0; t < num_trials; t++) {
        // Generate unique signals if summing, otherwise one shared signal
        for(int i=0; i<num_tx_signals; i++) {
            generate_random_signal(tx_sig_re[i], tx_sig_im[i], nb_tx, num_samples);
        }
        for(int c=0; c<num_channels; c++) random_channel(channels[c], 0);

        struct timespec start, end;

        // --- CPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        for(int c=0; c<num_channels; c++){
            // Use the correct input signal for the current channel
            float** current_tx_re = sum_outputs ? tx_sig_re[c] : tx_sig_re[0];
            float** current_tx_im = sum_outputs ? tx_sig_im[c] : tx_sig_im[0];
            multipath_channel_float(channels[c], current_tx_re, current_tx_im, rx_multipath_re_cpu, rx_multipath_im_cpu, num_samples, 1, 0);
            add_noise_float(output_cpu[c], (const float **)rx_multipath_re_cpu, (const float **)rx_multipath_im_cpu, 0.1, num_samples, 0, 0, 0, 0, 0, nb_rx);
        }
        if (sum_outputs) {
            // Allocate a temporary buffer for the sum, similar to the GPU side
            c16_t* final_sum_cpu = calloc(nb_rx * num_samples, sizeof(c16_t));
            for (int c = 0; c < num_channels; c++) {
                // The output_cpu array is structured as [channel][rx_antenna][sample]
                // For a contiguous block, we access it via output_cpu[c][0]
                for (int i = 0; i < nb_rx * num_samples; i++) {
                    final_sum_cpu[i].r += output_cpu[c][0][i].r;
                    final_sum_cpu[i].i += output_cpu[c][0][i].i;
                }
            }
            free(final_sum_cpu); // We just perform the sum for timing, no need to store
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

        // --- GPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (sum_outputs) {
            for(int c=0; c<num_channels; c++){
                float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                 for (int link = 0; link < nb_tx * nb_rx; link++) {
                    for (int l = 0; l < channels[c]->channel_length; l++) {
                        int idx = link * channels[c]->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
                // Call pipeline with NULL for host output pointer to prevent D->H copy
                // and provide the per-channel device output buffer instead.
                run_channel_pipeline_cuda(tx_sig_re[c], tx_sig_im[c], NULL, nb_tx, nb_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 0.1, 0, 0, 0, 0, 0, d_tx_sig, d_rx_sig, d_individual_gpu_outputs[c], d_curand_states, h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu);
            }
            // All channels processed, now sum their outputs on the GPU
            sum_channel_outputs_cuda(d_individual_gpu_outputs, d_summed_gpu_output, num_channels, nb_rx, num_samples);
        } else {
            // Original benchmark logic
            for(int c=0; c<num_channels; c++){
                float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                for (int link = 0; link < nb_tx * nb_rx; link++) {
                    for (int l = 0; l < channels[c]->channel_length; l++) {
                        int idx = link * channels[c]->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
                // Correctly pass the first signal (float**) from the (float***) array
                run_channel_pipeline_cuda(tx_sig_re[0], tx_sig_im[0], output_gpu[c], nb_tx, nb_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 0.1, 0, 0, 0, 0, 0, d_tx_sig, d_rx_sig, d_output_noise, d_curand_states, h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu);
            }
        }
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    }
    
    // --- FINAL REPORT ---
    double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
    double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
    double speedup = (avg_cpu_us > 0 && avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
    
    // --- New Metrics ---
    double avg_cpu_per_channel_us = avg_cpu_us / num_channels;
    double avg_gpu_per_channel_us = avg_gpu_us / num_channels;
    double total_samples_processed = (double)num_channels * nb_rx * num_samples;
    double gpu_throughput_gsps = total_samples_processed / (avg_gpu_us * 1000.0);

    char val_str[30];

    printf("\n--- Final Benchmark Results ---\n\n");
    printf("+----------------------------------+--------------------------+\n");
    printf("| %-32s | %-24s |\n", "Configuration", "Value");
    printf("+----------------------------------+--------------------------+\n");
    printf("| %-32s | %-24d |\n", "Parallel Channels", num_channels);

    snprintf(val_str, sizeof(val_str), "%d x %d", nb_tx, nb_rx);
    printf("| %-32s | %-24s |\n", "MIMO Configuration (Tx x Rx)", val_str);

    printf("| %-32s | %-24d |\n", "Signal Length (Samples)", num_samples);
    printf("| %-32s | %-24d |\n", "Trials per configuration", num_trials);
    printf("+----------------------------------+--------------------------+\n");
    printf("| %-32s | %-24s |\n", "Performance Metric", "Value");
    printf("+----------------------------------+--------------------------+\n");
    printf("| %-32s | %-24.2f |\n", "Total CPU Time (us)", avg_cpu_us);
    printf("| %-32s | %-24.2f |\n", "Total GPU Time (us)", avg_gpu_us);
    printf("| %-32s | %-24.2f |\n", "Avg Time per Channel - CPU (us)", avg_cpu_per_channel_us);
    printf("| %-32s | %-24.2f |\n", "Avg Time per Channel - GPU (us)", avg_gpu_per_channel_us);

    snprintf(val_str, sizeof(val_str), "%.2fx", speedup);
    printf("| %-32s | %-24s |\n", "Speedup (CPU/GPU)", val_str);

    printf("| %-32s | %-24.3f |\n", "GPU Throughput (GSPS)", gpu_throughput_gsps);
  
    printf("+----------------------------------+--------------------------+\n");
            // --- Cleanup ---
        free(h_channel_coeffs);
        for(int i=0; i<num_tx_signals; i++){
            for(int j=0; j<nb_tx; j++){
                free(tx_sig_re[i][j]);
                free(tx_sig_im[i][j]);
            }
            free(tx_sig_re[i]);
            free(tx_sig_im[i]);
        }
        free(tx_sig_re); free(tx_sig_im);

        for (int i=0; i<nb_rx; i++) { free(rx_multipath_re_cpu[i]); free(rx_multipath_im_cpu[i]); }
        free(rx_multipath_re_cpu); free(rx_multipath_im_cpu);

        for (int c=0; c<num_channels; c++) {
            free(output_cpu[c][0]);
            free(output_gpu[c][0]);
            free(output_cpu[c]);
            free(output_gpu[c]);
        }
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
        cudaFree(d_channel_coeffs_gpu);

        if (sum_outputs) {
            for (int c=0; c<num_channels; c++) {
                cudaFree(d_individual_gpu_outputs[c]);
            }
            free(d_individual_gpu_outputs);
            cudaFree(d_summed_gpu_output);
        }

        destroy_curand_states_cuda(d_curand_states);
        
    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("Pipeline benchmark finished.\n");

    return 0;
}
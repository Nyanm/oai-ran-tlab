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

channel_desc_t* create_manual_channel_desc(int nb_tx, int nb_rx, int channel_length) {
    channel_desc_t* desc = (channel_desc_t*)calloc(1, sizeof(channel_desc_t));
    desc->nb_tx = nb_tx;
    desc->nb_rx = nb_rx;
    desc->channel_length = channel_length;
    desc->path_loss_dB = 0.0;
    desc->channel_offset = 0;
    int num_links = nb_tx * nb_rx;
    desc->ch = (struct complexd**)malloc(num_links * sizeof(struct complexd*));
    for (int i = 0; i < num_links; i++) {
        desc->ch[i] = (struct complexd*)malloc(channel_length * sizeof(struct complexd));
        for (int l = 0; l < channel_length; l++) {
            desc->ch[i][l].r = (double)rand() / (double)RAND_MAX * 0.1;
            desc->ch[i][l].i = (double)rand() / (double)RAND_MAX * 0.1;
        }
    }
    return desc;
}

void free_manual_channel_desc(channel_desc_t* desc) {
    if (!desc) return;
    int num_links = desc->nb_tx * desc->nb_rx;
    for (int i = 0; i < num_links; i++) free(desc->ch[i]);
    free(desc->ch);
    free(desc);
}


int main(int argc, char **argv) {
    
    logInit();
    randominit(0);
    
    // Default values for our new command-line parameters
    int nb_tx = 4;
    int nb_rx = 4;
    int num_samples = 30720;
    int channel_length = 16;
    int num_trials = 50;
    float snr_db = 15.0f;



    // int nb_tx_configs[] = {1, 2, 4, 8};
    // int nb_rx_configs[] = {1, 2, 4, 8};
    // int num_samples_configs[] = {30720, 61440, 122880};
    // int channel_length_configs[] = {16, 32};
    // char* channel_type_names[] = {"Short Channel", "Long Channel"};
    // int num_trials = 50;
    // float snr_db = 15.0f;

    // --- Argument Parsing ---
    struct option long_options[] = {
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
    while ((opt = getopt_long(argc, argv, "t:r:s:l:n:S:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 't': nb_tx = atoi(optarg); break;
            case 'r': nb_rx = atoi(optarg); break;
            case 's': num_samples = atoi(optarg); break;
            case 'l': channel_length = atoi(optarg); break;
            case 'n': num_trials = atoi(optarg); break;
            case 'S': snr_db = atof(optarg); break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -t, --nb-tx <N>          Number of transmit antennas (Default: 4)\n");
                printf("  -r, --nb-rx <N>          Number of receive antennas (Default: 4)\n");
                printf("  -s, --num-samples <N>    Number of samples (Default: 30720)\n");
                printf("  -l, --ch-len <N>         Channel length (Default: 16)\n");
                printf("  -n, --trials <N>         Number of trials for averaging (Default: 50)\n");
                printf("  -S, --snr <dB>           Signal to Noise Ratio in dB (Default: 15.0)\n");
                printf("  -h, --help               Show this help message\n");
                return 0;
            default: exit(1);
        }
    }

    printf("Starting Full Channel Pipeline Benchmark (Multipath + Noise)\n");
    printf("Averaging each test case over %d trials.\n", num_trials);
    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-15s | %-15s | %-20s | %-20s | %-15s\n", "Channel Type", "MIMO Config", "Signal Length", "CPU Pipeline (us)", "GPU Pipeline (us)", "Overall Speedup");
    printf("----------------------------------------------------------------------------------------------------------------------\n");


        char mimo_str[16];
        sprintf(mimo_str, "%dx%d", nb_tx, nb_rx);

        channel_desc_t *chan_desc = create_manual_channel_desc(nb_tx, nb_rx, channel_length);

        float **tx_sig_re = malloc(nb_tx * sizeof(float *));
        float **tx_sig_im = malloc(nb_tx * sizeof(float *));
        float **rx_multipath_re_cpu = malloc(nb_rx * sizeof(float *));
        float **rx_multipath_im_cpu = malloc(nb_rx * sizeof(float *));
        c16_t **output_cpu = malloc(nb_rx * sizeof(c16_t *));
        c16_t **output_gpu = malloc(nb_rx * sizeof(c16_t *));

        for (int i=0; i<nb_tx; i++) { tx_sig_re[i] = malloc(num_samples * sizeof(float)); tx_sig_im[i] = malloc(num_samples * sizeof(float)); }
        for (int i=0; i<nb_rx; i++) {
            rx_multipath_re_cpu[i] = malloc(num_samples * sizeof(float));
            rx_multipath_im_cpu[i] = malloc(num_samples * sizeof(float));
        }
        output_cpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        for (int i=1; i<nb_rx; i++) {
            output_cpu[i] = output_cpu[0] + i * num_samples;
            output_gpu[i] = output_gpu[0] + i * num_samples;
        }
 
        void *d_tx_sig, *d_rx_sig, *d_output_noise, *d_curand_states, *h_tx_sig_pinned, *h_output_sig_pinned, *d_channel_coeffs_gpu; // <-- ADDED
        size_t channel_buffer_size = nb_tx * nb_rx * channel_length * sizeof(float) * 2;

        #if defined(USE_UNIFIED_MEMORY)
            cudaMallocManaged(&d_tx_sig, nb_tx * num_samples * sizeof(float) * 2, cudaMemAttachGlobal);
            cudaMallocManaged(&d_rx_sig, nb_rx * num_samples * sizeof(float) * 2, cudaMemAttachGlobal);
            cudaMallocManaged(&d_output_noise, nb_rx * num_samples * sizeof(short) * 2, cudaMemAttachGlobal);
            cudaMallocManaged(&d_channel_coeffs_gpu, channel_buffer_size, cudaMemAttachGlobal); // <-- ALLOCATE
            h_tx_sig_pinned = d_tx_sig;
            h_output_sig_pinned = d_output_noise;
        #else 
            cudaMalloc(&d_tx_sig, nb_tx * num_samples * sizeof(float) * 2);
            cudaMalloc(&d_rx_sig, nb_rx * num_samples * sizeof(float) * 2);
            cudaMalloc(&d_output_noise, nb_rx * num_samples * sizeof(short) * 2);
            cudaMallocHost(&h_tx_sig_pinned, nb_tx * num_samples * sizeof(float) * 2);
            cudaMallocHost(&h_output_sig_pinned, nb_rx * num_samples * sizeof(short) * 2);
            cudaMalloc(&d_channel_coeffs_gpu, channel_buffer_size); // <-- ALLOCATE
        #endif
        d_curand_states = create_and_init_curand_states_cuda(nb_rx * num_samples, time(NULL));

        double ts = 1.0 / 30.72e6;
        float sigma2 = 1.0f / powf(10.0f, snr_db / 10.0f);
        float path_loss = (float)pow(10, chan_desc->path_loss_dB / 20.0);
        int num_links = nb_tx * nb_rx;
        float* h_channel_coeffs = (float*)malloc(num_links * channel_length * sizeof(float2));
        for (int link = 0; link < num_links; link++) {
            for (int l = 0; l < channel_length; l++) {
                int idx = link * channel_length + l;
                ((float2*)h_channel_coeffs)[idx].x = (float)chan_desc->ch[link][l].r;
                ((float2*)h_channel_coeffs)[idx].y = (float)chan_desc->ch[link][l].i;
            }
        }
        
        double total_cpu_ns = 0;
        double total_gpu_ns = 0;

        for (int t = 0; t < num_trials; t++) {
            generate_random_signal(tx_sig_re, tx_sig_im, nb_tx, num_samples);
            struct timespec start, end;

            clock_gettime(CLOCK_MONOTONIC, &start);
            multipath_channel_float(chan_desc, tx_sig_re, tx_sig_im, rx_multipath_re_cpu, rx_multipath_im_cpu, num_samples, 1, 0);
            add_noise_float(output_cpu, (const float **)rx_multipath_re_cpu, (const float **)rx_multipath_im_cpu, sigma2, num_samples, 0, ts, 0, 0, 0, nb_rx);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

            clock_gettime(CLOCK_MONOTONIC, &start);
            run_channel_pipeline_cuda(
                tx_sig_re, tx_sig_im, output_gpu,
                nb_tx, nb_rx, channel_length, num_samples,
                path_loss, h_channel_coeffs,
                sigma2, ts,
                0, 0, 0, 0, 
                d_tx_sig, d_rx_sig, d_output_noise, d_curand_states,
                h_tx_sig_pinned, h_output_sig_pinned, d_channel_coeffs_gpu
            );
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        }
        
        double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
        double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
        double speedup = (avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
        
        printf("%-15s | %-15d | %-20.2f | %-20.2f | %-15.2fx\n", 
               mimo_str, num_samples, avg_cpu_us, avg_gpu_us, speedup);

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
        
        free_manual_channel_desc(chan_desc);


    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("Pipeline benchmark finished.\n");

    return 0;
}
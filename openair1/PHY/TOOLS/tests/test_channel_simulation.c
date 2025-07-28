/**
 * Benchmark for the complete channel simulation pipeline (Multipath + Noise).
 *
 * This test measures the end-to-end performance of running the multipath channel
 * simulation followed immediately by the noise generation, mimicking the sequence in nr-dlsim/nr-ulsim.
 * It compares the total execution time of the original CPU functions against the
 * complete CUDA-accelerated pipeline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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
    
    int nb_tx_configs[] = {1, 2, 4, 8};
    int nb_rx_configs[] = {1, 2, 4, 8};
    int num_samples_configs[] = {30720, 61440, 122880};
    int channel_length_configs[] = {16, 32};
    char* channel_type_names[] = {"Short Channel", "Long Channel"};
    int num_trials = 50;
    float snr_db = 15.0f;

    printf("Starting Full Channel Pipeline Benchmark (Multipath + Noise)\n");
    printf("Averaging each test case over %d trials.\n", num_trials);
    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("%-15s | %-15s | %-15s | %-20s | %-20s | %-15s\n", "Channel Type", "MIMO Config", "Signal Length", "CPU Pipeline (us)", "GPU Pipeline (us)", "Overall Speedup");
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

        // --- Allocate ALL Memory Buffers ---
        // Transmit Signal (input to pipeline)
        float **tx_sig_re = malloc(nb_tx * sizeof(float *));
        float **tx_sig_im = malloc(nb_tx * sizeof(float *));

        // Intermediate Buffers (output of multipath, input to noise)
        float **rx_multipath_re_cpu = malloc(nb_rx * sizeof(float *));
        float **rx_multipath_im_cpu = malloc(nb_rx * sizeof(float *));
        float **rx_multipath_re_gpu = malloc(nb_rx * sizeof(float *));
        float **rx_multipath_im_gpu = malloc(nb_rx * sizeof(float *));

        // Final Output Buffers
        c16_t **output_cpu = malloc(nb_rx * sizeof(c16_t *));
        c16_t **output_gpu = malloc(nb_rx * sizeof(c16_t *));

        for (int i=0; i<nb_tx; i++) { tx_sig_re[i] = malloc(num_samples * sizeof(float)); tx_sig_im[i] = malloc(num_samples * sizeof(float)); }
        for (int i=0; i<nb_rx; i++) {
            rx_multipath_re_cpu[i] = malloc(num_samples * sizeof(float));
            rx_multipath_im_cpu[i] = malloc(num_samples * sizeof(float));
            rx_multipath_re_gpu[i] = malloc(num_samples * sizeof(float));
            rx_multipath_im_gpu[i] = malloc(num_samples * sizeof(float));
        }
        output_cpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        for (int i=1; i<nb_rx; i++) {
            output_cpu[i] = output_cpu[0] + i * num_samples;
            output_gpu[i] = output_gpu[0] + i * num_samples;
        }

        // --- GPU Memory Allocation ---
        // For multipath_channel_cuda_fast
        void *d_tx_sig, *d_rx_sig;
        cudaMalloc(&d_tx_sig,   nb_tx * num_samples * sizeof(float2));
        cudaMalloc(&d_rx_sig,   nb_rx * num_samples * sizeof(float2));
        
        // For add_noise_cuda
        void *d_r_sig_noise, *d_output_noise, *d_curand_states;
        void *h_r_sig_pinned, *h_output_sig_pinned;
        cudaMalloc(&d_r_sig_noise,   nb_rx * num_samples * sizeof(float2));
        cudaMalloc(&d_output_noise,  nb_rx * num_samples * sizeof(short2));
        cudaMallocHost(&h_r_sig_pinned, nb_rx * num_samples * sizeof(float2));
        void* h_tx_sig_pinned;
        cudaMallocHost(&h_tx_sig_pinned, nb_tx * num_samples * sizeof(float2));
        cudaMallocHost(&h_output_sig_pinned, nb_rx * num_samples * sizeof(short2));
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
                0, 0, // pdu_bit_map, ptrs_bit_map (default for test)
                0, 0, // slot_offset, delay (default for test)
                d_tx_sig,           
                d_rx_sig,           
                d_output_noise,     
                d_curand_states,
                h_tx_sig_pinned,    
                h_output_sig_pinned 
            );
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        }
        
        double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
        double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
        double speedup = (avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
        
        printf("%-15s | %-15s | %-15d | %-20.2f | %-20.2f | %-15.2fx\n", 
               channel_type_names[c], mimo_str, num_samples, avg_cpu_us, avg_gpu_us, speedup);

        free(h_channel_coeffs);
        for (int i=0; i<nb_tx; i++) { free(tx_sig_re[i]); free(tx_sig_im[i]); }
        for (int i=0; i<nb_rx; i++) { free(rx_multipath_re_cpu[i]); free(rx_multipath_im_cpu[i]); free(rx_multipath_re_gpu[i]); free(rx_multipath_im_gpu[i]); }
        free(tx_sig_re); free(tx_sig_im);
        free(rx_multipath_re_cpu); free(rx_multipath_im_cpu);
        free(rx_multipath_re_gpu); free(rx_multipath_im_gpu);
        free(output_cpu[0]); free(output_gpu[0]);
        free(output_cpu); free(output_gpu);

        cudaFree(d_tx_sig); cudaFree(d_rx_sig);
        cudaFreeHost(h_tx_sig_pinned);
        cudaFree(d_r_sig_noise); cudaFree(d_output_noise);
        cudaFreeHost(h_r_sig_pinned); cudaFreeHost(h_output_sig_pinned);
        destroy_curand_states_cuda(d_curand_states);
        
        free_manual_channel_desc(chan_desc);
    }
    }
    }

    printf("----------------------------------------------------------------------------------------------------------------------\n");
    printf("Pipeline benchmark finished.\n");

    return 0;
}
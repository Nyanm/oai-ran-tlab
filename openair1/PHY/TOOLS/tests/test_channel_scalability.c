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
    for (int i = 0; i < num_links; i++) {
        if (desc->ch[i]) free(desc->ch[i]);
    }
    if (desc->ch) free(desc->ch);
    free(desc);
}

int main(int argc, char **argv) {
    
    logInit();
    randominit(0);
    
    int num_channels = 1;
    int nb_tx = 4;
    int nb_rx = 4;
    int num_samples = 122880;
    int channel_length = 32;
    int num_trials = 50;
    int sum_outputs = 0;
    char mode_str[10] = "batch";

    struct option long_options[] = {
        {"num-channels", required_argument, 0, 'c'},
        {"nb-tx",        required_argument, 0, 't'},
        {"nb-rx",        required_argument, 0, 'r'},
        {"num-samples",  required_argument, 0, 's'},
        {"ch-len",       required_argument, 0, 'l'},
        {"trials",       required_argument, 0, 'n'},
        {"sum-outputs",  no_argument,       0, 'S'},
        {"mode",         required_argument, 0, 'm'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:t:r:s:l:n:Sm:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'c': num_channels = atoi(optarg); break;
            case 't': nb_tx = atoi(optarg); break;
            case 'r': nb_rx = atoi(optarg); break;
            case 's': num_samples = atoi(optarg); break;
            case 'l': channel_length = atoi(optarg); break;
            case 'n': num_trials = atoi(optarg); break;
            case 'S': sum_outputs = 1; break;
            case 'm': strncpy(mode_str, optarg, sizeof(mode_str)-1); break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  -c, --num-channels <N>   Number of parallel channels to simulate (Default: 1)\n");
                printf("  -t, --nb-tx <N>          Number of transmit antennas (Default: 4)\n");
                printf("  -r, --nb-rx <N>          Number of receive antennas (Default: 4)\n");
                printf("  -s, --num-samples <N>    Number of samples (Default: 30720)\n");
                printf("  -l, --ch-len <N>         Channel length (Default: 32)\n");
                printf("  -n, --trials <N>         Number of trials for averaging (Default: 50)\n");
                printf("  -S, --sum-outputs        Enable summation of outputs for interference simulation.\n");
                printf("  -m, --mode <serial|stream|batch>  GPU execution mode (Default: batch)\n");
                printf("  -h, --help               Show this help message\n");
                return 0;
            default: exit(1);
        }
    }
    
    // --- MEMORY ALLOCATION ---
    // HOST MEMORY
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
    
    c16_t ***output_cpu = malloc(num_channels * sizeof(c16_t**));
    for(int c=0; c<num_channels; c++){
        output_cpu[c] = malloc(nb_rx * sizeof(c16_t*));
        output_cpu[c][0] = malloc(nb_rx * num_samples * sizeof(c16_t));
        for(int i=1; i<nb_rx; i++) {
            output_cpu[c][i] = output_cpu[c][0] + i * num_samples;
        }
    }
    
    channel_desc_t **channels = malloc(num_channels * sizeof(channel_desc_t*));
    for (int c=0; c<num_channels; c++) {
        channels[c] = create_manual_channel_desc(nb_tx, nb_rx, channel_length);
    }
    
    // DEVICE MEMORY
    void *d_tx_sig=NULL, *d_rx_sig=NULL, *d_curand_states=NULL, 
         *h_tx_sig_pinned=NULL, *h_output_sig_pinned=NULL, *d_channel_coeffs_gpu=NULL,
         **d_individual_gpu_outputs = NULL, *d_summed_gpu_output = NULL;
    c16_t **output_gpu = NULL;
    
    void *d_tx_sig_batch = NULL, *d_intermediate_sig_batch = NULL, *d_final_output_batch = NULL,
         *d_channel_coeffs_batch = NULL, *d_path_loss_batch = NULL;
    float2 *h_channel_coeffs_batch = NULL;
    float *h_path_loss_batch = NULL;
    float *h_channel_coeffs = NULL;

    const int max_taps = 256;

    if (strcmp(mode_str, "batch") == 0) {

        size_t tx_batch_bytes = num_channels * nb_tx * num_samples * sizeof(float2);
        size_t intermediate_batch_bytes = num_channels * nb_rx * num_samples * sizeof(float2);
        size_t final_batch_bytes = num_channels * nb_rx * num_samples * sizeof(short2);
        size_t channel_batch_bytes = num_channels * nb_tx * nb_rx * channel_length * sizeof(float2);
        
        h_channel_coeffs_batch = malloc(channel_batch_bytes);
        h_path_loss_batch = malloc(num_channels * sizeof(float));

    #if defined(USE_UNIFIED_MEMORY)
            printf("Memory Mode: Unified Memory\n");
            cudaMallocManaged(&d_tx_sig_batch, tx_batch_bytes, cudaMemAttachGlobal);
            cudaMallocManaged(&d_intermediate_sig_batch, intermediate_batch_bytes, cudaMemAttachGlobal);
            cudaMallocManaged(&d_final_output_batch, final_batch_bytes, cudaMemAttachGlobal);
            cudaMallocManaged(&d_channel_coeffs_batch, channel_batch_bytes, cudaMemAttachGlobal);
            cudaMallocManaged(&d_path_loss_batch, num_channels * sizeof(float), cudaMemAttachGlobal);
    #elif defined(USE_ATS_MEMORY)
            printf("Memory Mode: ATS\n");
            d_tx_sig_batch = malloc(tx_batch_bytes);
            cudaMalloc(&d_intermediate_sig_batch, intermediate_batch_bytes);
            cudaMalloc(&d_final_output_batch, final_batch_bytes);
            cudaMalloc(&d_channel_coeffs_batch, channel_batch_bytes);
            cudaMalloc(&d_path_loss_batch, num_channels * sizeof(float));
    #else
            printf("Memory Mode: Explicit Copy\n");
            cudaMalloc(&d_tx_sig_batch, tx_batch_bytes);
            cudaMalloc(&d_intermediate_sig_batch, intermediate_batch_bytes);
            cudaMalloc(&d_final_output_batch, final_batch_bytes);
            cudaMalloc(&d_channel_coeffs_batch, channel_batch_bytes);
            cudaMalloc(&d_path_loss_batch, num_channels * sizeof(float));
    #endif


    } else { 
        // --- SERIAL & STREAM MODE MEMORY ALLOCATION ---
        size_t tx_bytes = nb_tx * num_samples * sizeof(float2);
        size_t rx_bytes = nb_rx * num_samples * sizeof(float2);
        size_t output_bytes = nb_rx * num_samples * sizeof(short2);

        #if defined(USE_UNIFIED_MEMORY)
                printf("Memory Mode: Unified Memory\n");
                cudaMallocManaged(&d_channel_coeffs_gpu, nb_tx * nb_rx * max_taps * sizeof(float2), cudaMemAttachGlobal);
                cudaMallocManaged(&d_tx_sig, tx_bytes, cudaMemAttachGlobal);
                cudaMallocManaged(&d_rx_sig, rx_bytes, cudaMemAttachGlobal);
                d_individual_gpu_outputs = malloc(num_channels * sizeof(void*));
                for (int c = 0; c < num_channels; c++) {
                    cudaMallocManaged(&d_individual_gpu_outputs[c], output_bytes, cudaMemAttachGlobal);
                }
                if (sum_outputs) {
                    cudaMallocManaged(&d_summed_gpu_output, output_bytes, cudaMemAttachGlobal);
                }
                h_tx_sig_pinned = d_tx_sig;

                if (strcmp(mode_str, "serial") == 0) {
                    cudaMallocHost(&h_output_sig_pinned, output_bytes);
                    output_gpu = malloc(nb_rx * sizeof(c16_t*));
                    output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
                    for(int i=1; i<nb_rx; i++) output_gpu[i] = output_gpu[0] + i*num_samples;
                }
        #elif defined(USE_ATS_MEMORY)
                printf("Memory Mode: ATS\n");
                cudaMalloc(&d_channel_coeffs_gpu, nb_tx * nb_rx * max_taps * sizeof(float2));
                cudaMalloc(&d_rx_sig, rx_bytes);
                h_tx_sig_pinned = malloc(tx_bytes);
                d_tx_sig = NULL; 

                d_individual_gpu_outputs = malloc(num_channels * sizeof(void*));
                for (int c = 0; c < num_channels; c++) {
                    cudaMalloc(&d_individual_gpu_outputs[c], output_bytes);
                }
                if (sum_outputs) {
                    cudaMalloc(&d_summed_gpu_output, output_bytes);
                }
                if (strcmp(mode_str, "serial") == 0) {
                    h_output_sig_pinned = malloc(output_bytes);
                    output_gpu = malloc(nb_rx * sizeof(c16_t*));
                    output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
                    for(int i=1; i<nb_rx; i++) output_gpu[i] = output_gpu[0] + i*num_samples;
                }
        #else
                printf("Memory Mode: Explicit Copy\n");
                cudaMalloc(&d_channel_coeffs_gpu, nb_tx * nb_rx * max_taps * sizeof(float2));
                cudaMalloc(&d_tx_sig, tx_bytes);
                cudaMalloc(&d_rx_sig, rx_bytes);
                cudaMallocHost(&h_tx_sig_pinned, tx_bytes); // Pinned for performance

                d_individual_gpu_outputs = malloc(num_channels * sizeof(void*));
                for (int c = 0; c < num_channels; c++) {
                    cudaMalloc(&d_individual_gpu_outputs[c], output_bytes);
                }
                if (sum_outputs) {
                    cudaMalloc(&d_summed_gpu_output, output_bytes);
                }
                if (strcmp(mode_str, "serial") == 0) {
                    cudaMallocHost(&h_output_sig_pinned, output_bytes); // Pinned for performance
                    output_gpu = malloc(nb_rx * sizeof(c16_t*));
                    output_gpu[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
                    for(int i=1; i<nb_rx; i++) output_gpu[i] = output_gpu[0] + i*num_samples;
                }
        #endif

        h_channel_coeffs = malloc(nb_tx * nb_rx * max_taps * sizeof(float2));
    }

    // todo: stabilize curand states
    d_curand_states = create_and_init_curand_states_cuda(nb_rx * num_samples, time(NULL));
    // d_curand_states = create_and_init_curand_states_cuda(num_channels * nb_rx * num_samples, time(NULL));
    
    double total_cpu_ns = 0;
    double total_gpu_ns = 0;

    // --- MAIN TIMING LOOP ---
    for (int t = 0; t < num_trials; t++) {
        for(int i=0; i<num_tx_signals; i++) {
            generate_random_signal(tx_sig_re[i], tx_sig_im[i], nb_tx, num_samples);
        }
        for(int c=0; c<num_channels; c++) random_channel(channels[c], 0);

        struct timespec start, end;

        // --- CPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);
        for(int c=0; c<num_channels; c++){
            float** current_tx_re = sum_outputs ? tx_sig_re[c] : tx_sig_re[0];
            float** current_tx_im = sum_outputs ? tx_sig_im[c] : tx_sig_im[0];
            multipath_channel_float(channels[c], current_tx_re, current_tx_im, rx_multipath_re_cpu, rx_multipath_im_cpu, num_samples, 1, 0);
            add_noise_float(output_cpu[c], (const float **)rx_multipath_re_cpu, (const float **)rx_multipath_im_cpu, 0.1, num_samples, 0, 0, 0, 0, 0, nb_rx);
        }
        if (sum_outputs) {
            c16_t* final_sum_cpu = calloc(nb_rx * num_samples, sizeof(c16_t));
            for (int c = 0; c < num_channels; c++) {
                for (int i = 0; i < nb_rx * num_samples; i++) {
                    final_sum_cpu[i].r += output_cpu[c][0][i].r;
                    final_sum_cpu[i].i += output_cpu[c][0][i].i;
                }
            }
            free(final_sum_cpu);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_cpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

        // --- GPU RUN ---
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (strcmp(mode_str, "batch") == 0) {
            for (int c = 0; c < num_channels; c++) {
                h_path_loss_batch[c] = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                for (int link = 0; link < nb_tx * nb_rx; link++) {
                    for (int l = 0; l < channel_length; l++) {
                        int batch_idx = (c * nb_tx * nb_rx + link) * channel_length + l;
                        h_channel_coeffs_batch[batch_idx].x = (float)channels[c]->ch[link][l].r;
                        h_channel_coeffs_batch[batch_idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
            }

            #if defined(USE_UNIFIED_MEMORY)
                    float2* tx_batch_ptr = (float2*)d_tx_sig_batch;
                    for (int c = 0; c < num_channels; c++) {
                        float** current_tx_re = sum_outputs ? tx_sig_re[c] : tx_sig_re[0];
                        float** current_tx_im = sum_outputs ? tx_sig_im[c] : tx_sig_im[0];
                        for (int j = 0; j < nb_tx; j++) {
                            for (int i = 0; i < num_samples; i++) {
                                int batch_idx = (c * nb_tx + j) * num_samples + i;
                                tx_batch_ptr[batch_idx].x = current_tx_re[j][i];
                                tx_batch_ptr[batch_idx].y = current_tx_im[j][i];
                            }
                        }
                    }
                    memcpy(d_channel_coeffs_batch, h_channel_coeffs_batch, num_channels * nb_tx * nb_rx * channel_length * sizeof(float2));
                    memcpy(d_path_loss_batch, h_path_loss_batch, num_channels * sizeof(float));
            #elif defined(USE_ATS_MEMORY)
                    float2* tx_batch_ptr_ats = (float2*)d_tx_sig_batch;
                    for (int c = 0; c < num_channels; c++) {
                        float** current_tx_re = sum_outputs ? tx_sig_re[c] : tx_sig_re[0];
                        float** current_tx_im = sum_outputs ? tx_sig_im[c] : tx_sig_im[0];
                        for (int j = 0; j < nb_tx; j++) {
                            for (int i = 0; i < num_samples; i++) {
                                int batch_idx = (c * nb_tx + j) * num_samples + i;
                                tx_batch_ptr_ats[batch_idx].x = current_tx_re[j][i];
                                tx_batch_ptr_ats[batch_idx].y = current_tx_im[j][i];
                            }
                         }
                    }
                cudaMemcpy(d_channel_coeffs_batch, h_channel_coeffs_batch, num_channels * nb_tx * nb_rx * channel_length * sizeof(float2), cudaMemcpyHostToDevice);
                cudaMemcpy(d_path_loss_batch, h_path_loss_batch, num_channels * sizeof(float), cudaMemcpyHostToDevice);
            #else // EXPLICIT COPY
                    float2* h_tx_sig_batch_interleaved = (float2*)malloc(num_channels * nb_tx * num_samples * sizeof(float2));
                    for (int c = 0; c < num_channels; c++) {
                        float** current_tx_re = sum_outputs ? tx_sig_re[c] : tx_sig_re[0];
                        float** current_tx_im = sum_outputs ? tx_sig_im[c] : tx_sig_im[0];
                        for (int j = 0; j < nb_tx; j++) {
                            for (int i = 0; i < num_samples; i++) {
                                int batch_idx = (c * nb_tx + j) * num_samples + i;
                                h_tx_sig_batch_interleaved[batch_idx].x = current_tx_re[j][i];
                                h_tx_sig_batch_interleaved[batch_idx].y = current_tx_im[j][i];
                            }
                        }
                    }
                    cudaMemcpy(d_tx_sig_batch, h_tx_sig_batch_interleaved, num_channels * nb_tx * num_samples * sizeof(float2), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_channel_coeffs_batch, h_channel_coeffs_batch, num_channels * nb_tx * nb_rx * channel_length * sizeof(float2), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_path_loss_batch, h_path_loss_batch, num_channels * sizeof(float), cudaMemcpyHostToDevice);
                    free(h_tx_sig_batch_interleaved);
            #endif
                        
            run_channel_pipeline_cuda_batched(num_channels, nb_tx, nb_rx, channel_length, num_samples,
                d_path_loss_batch, d_channel_coeffs_batch, 0.1, 0, 0xFFFF, 0xFFFF,
                d_tx_sig_batch, d_intermediate_sig_batch, d_final_output_batch, d_curand_states);      
            cudaDeviceSynchronize();

            if (sum_outputs) {
                // This doesn't allocate new memory. It creates an array of host-side pointers
                // that will point to locations inside the big GPU output buffer.
                d_individual_gpu_outputs = malloc(num_channels * sizeof(void*));
                for (int c = 0; c < num_channels; c++) {
                    d_individual_gpu_outputs[c] = d_final_output_batch + c * nb_rx * num_samples * sizeof(short2);
                }
                sum_channel_outputs_cuda(d_individual_gpu_outputs, d_summed_gpu_output, num_channels, nb_rx, num_samples);
            }

        } else if (strcmp(mode_str, "stream") == 0) {
            cudaStream_t streams[num_channels];
            for(int c=0; c<num_channels; c++) {
                cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking);
            }

            for(int c=0; c<num_channels; c++){
                float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                for (int link = 0; link < nb_tx * nb_rx; link++) {
                    for (int l = 0; l < channels[c]->channel_length; l++) {
                        int idx = link * channels[c]->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
                
                run_channel_pipeline_cuda_streamed(
                    sum_outputs ? tx_sig_re[c] : tx_sig_re[0],
                    sum_outputs ? tx_sig_im[c] : tx_sig_im[0],
                    nb_tx, nb_rx, channels[c]->channel_length, num_samples,
                    path_loss, h_channel_coeffs, 0.1, 0, 0xFFFF, 0xFFFF,
                    d_tx_sig, d_rx_sig, d_individual_gpu_outputs[c], d_curand_states,
                    h_tx_sig_pinned, d_channel_coeffs_gpu, (void*)streams[c]
                );
            }

            if (sum_outputs) {
                sum_channel_outputs_cuda(d_individual_gpu_outputs, d_summed_gpu_output, num_channels, nb_rx, num_samples);
            }
            cudaDeviceSynchronize();

            for(int c=0; c<num_channels; c++) {
                cudaStreamDestroy(streams[c]);
            }

        } else if (strcmp(mode_str, "serial") == 0) {
            c16_t** output_gpu_serial = malloc(nb_rx * sizeof(c16_t*));
            output_gpu_serial[0] = malloc(nb_rx * num_samples * sizeof(c16_t));
            for(int i=1; i<nb_rx; i++) output_gpu_serial[i] = output_gpu_serial[0] + i*num_samples;

            for(int c=0; c<num_channels; c++){
                float path_loss = (float)pow(10, channels[c]->path_loss_dB / 20.0);
                for (int link = 0; link < nb_tx * nb_rx; link++) {
                    for (int l = 0; l < channels[c]->channel_length; l++) {
                        int idx = link * channels[c]->channel_length + l;
                        ((float2*)h_channel_coeffs)[idx].x = (float)channels[c]->ch[link][l].r;
                        ((float2*)h_channel_coeffs)[idx].y = (float)channels[c]->ch[link][l].i;
                    }
                }
                run_channel_pipeline_cuda(
                    sum_outputs ? tx_sig_re[c] : tx_sig_re[0],
                    sum_outputs ? tx_sig_im[c] : tx_sig_im[0],
                    output_gpu_serial,
                    nb_tx, nb_rx, channels[c]->channel_length, num_samples, path_loss, h_channel_coeffs, 0.1, 0, 
                    0xFFFF, 0xFFFF, 0, 0, 
                    d_tx_sig, d_rx_sig, d_individual_gpu_outputs[c], d_curand_states,
                    h_tx_sig_pinned, h_output_sig_pinned,
                    d_channel_coeffs_gpu
                );
            }
            free(output_gpu_serial[0]);
            free(output_gpu_serial);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    }
    
    // --- FINAL REPORT ---
    double avg_cpu_us = (total_cpu_ns / num_trials) / 1000.0;
    double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
    double speedup = (avg_cpu_us > 0 && avg_gpu_us > 0) ? (avg_cpu_us / avg_gpu_us) : 0;
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
        free(output_cpu[c]);
        free_manual_channel_desc(channels[c]);
    }
    free(output_cpu); 
    free(channels);

    if (strcmp(mode_str, "batch") == 0) {
        free(h_channel_coeffs_batch);
        free(h_path_loss_batch);
        #if defined(USE_ATS_MEMORY)
                free(d_tx_sig_batch);
        #else
                cudaFree(d_tx_sig_batch);
        #endif
        cudaFree(d_intermediate_sig_batch);
        cudaFree(d_final_output_batch);
        cudaFree(d_channel_coeffs_batch);
        cudaFree(d_path_loss_batch);
    } else { 
        free(h_channel_coeffs);
        cudaFree(d_channel_coeffs_gpu);
        for (int c=0; c<num_channels; c++) {
            cudaFree(d_individual_gpu_outputs[c]);
        }
        free(d_individual_gpu_outputs);
        if (sum_outputs) {
            cudaFree(d_summed_gpu_output);
        }
        #if defined(USE_UNIFIED_MEMORY)
                cudaFree(d_tx_sig);
                cudaFree(d_rx_sig);
                if (strcmp(mode_str, "serial") == 0) {
                    free(h_output_sig_pinned);
                    free(output_gpu[0]);
                    free(output_gpu);
                }
        #elif defined(USE_ATS_MEMORY)
                cudaFree(d_rx_sig);
                free(h_tx_sig_pinned);
                if (strcmp(mode_str, "serial") == 0) {
                    free(h_output_sig_pinned);
                    free(output_gpu[0]);
                    free(output_gpu);
                }
        #else
                cudaFree(d_tx_sig);
                cudaFree(d_rx_sig);
                cudaFreeHost(h_tx_sig_pinned);
                if (strcmp(mode_str, "serial") == 0) {
                    cudaFreeHost(h_output_sig_pinned);
                    free(output_gpu[0]);
                    free(output_gpu);
                }
        #endif
    }

    destroy_curand_states_cuda(d_curand_states);
        
    printf("Benchmark finished.\n");
    return 0;
}

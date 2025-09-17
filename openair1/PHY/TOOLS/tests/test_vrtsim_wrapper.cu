/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <time.h>
#include <getopt.h>
#include "oai_cuda.h"
#include "common/config/config_userapi.h"
#include <cuda_profiler_api.h>

typedef struct complexd {
  double r;
  double i;
} complexd;

configmodule_interface_t* uniqCfg = NULL;

void exit_function(const char* file, const char* function, const int line, const char* s, ...)
{
  fprintf(stderr, "FATAL: %s at %s:%s:%d\n", s, file, function, line);
  exit(EXIT_FAILURE);
}

void generate_random_signal_interleaved(float* sig_interleaved, int num_samples)
{
  for (int j = 0; j < num_samples; j++) {
    sig_interleaved[2 * j] = (float)((rand() % 2000) - 1000);
    sig_interleaved[2 * j + 1] = (float)((rand() % 2000) - 1000);
  }
}

channel_desc_t* create_manual_channel_desc(int nb_tx, int nb_rx, int channel_length)
{
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
      desc->ch[i][l].r = ((double)rand() / (double)RAND_MAX * 0.1);
      desc->ch[i][l].i = ((double)rand() / (double)RAND_MAX * 0.1);
    }
  }
  return desc;
}

void free_manual_channel_desc(channel_desc_t* desc)
{
  if (!desc)
    return;
  int num_links = desc->nb_tx * desc->nb_rx;
  for (int i = 0; i < num_links; i++) {
    if (desc->ch[i])
      free(desc->ch[i]);
  }
  if (desc->ch)
    free(desc->ch);
  free(desc);
}

int main(int argc, char** argv)
{
  int nb_tx = 4;
  int nb_rx = 4;
  int num_samples = 61440;
  int channel_length = 16;
  int num_trials = 50;

  struct option long_options[] = {{"nb-tx", required_argument, 0, 't'},
                                  {"nb-rx", required_argument, 0, 'r'},
                                  {"num-samples", required_argument, 0, 's'},
                                  {"ch-len", required_argument, 0, 'l'},
                                  {"trials", required_argument, 0, 'n'},
                                  {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "t:r:s:l:n:", long_options, NULL)) != -1) {
    switch (opt) {
      case 't':
        nb_tx = atoi(optarg);
        break;
      case 'r':
        nb_rx = atoi(optarg);
        break;
      case 's':
        num_samples = atoi(optarg);
        break;
      case 'l':
        channel_length = atoi(optarg);
        break;
      case 'n':
        num_trials = atoi(optarg);
        break;
      default:
        exit(1);
    }
  }

  printf("\n--- vrtsim_cuda_process Benchmark ---\n");
  printf("+----------------------------------+--------------------------+\n");
  printf("| %-32s | %-24s |\n", "Configuration", "Value");
  printf("+----------------------------------+--------------------------+\n");
  printf("| %-32s | %d x %d                    |\n", "MIMO Config (Tx x Rx)", nb_tx, nb_rx);
  printf("| %-32s | %-24d |\n", "Signal Length (Samples)", num_samples);
  printf("| %-32s | %-24d |\n", "Channel Length (Taps)", channel_length);
  printf("| %-32s | %-24d |\n", "Trials for averaging", num_trials);
  printf("+----------------------------------+--------------------------+\n");

  srand(time(NULL));

  std::vector<c16_t*> h_input_samples(nb_tx);
  for (int i = 0; i < nb_tx; ++i) {
    cudaMallocHost(&h_input_samples[i], num_samples * sizeof(c16_t));
  }
  c16_t* h_final_output;
  cudaMallocHost(&h_final_output, num_samples * nb_rx * sizeof(c16_t));
  channel_desc_t* h_channel_desc = create_manual_channel_desc(nb_tx, nb_rx, channel_length);

  void* gpu_context = nullptr;
  vrtsim_cuda_init(&gpu_context, num_samples, nb_tx, nb_rx, channel_length);

  double total_gpu_ns = 0;
  struct timespec start, end;

  printf("Warming up GPU...\n");
  for (int w = 0; w < 10; ++w) {
    vrtsim_cuda_process(gpu_context,
                        h_input_samples.data(),
                        num_samples,
                        nb_tx,
                        nb_rx,
                        h_channel_desc,
                        1.0f,
                        1.0 / (30720 * 2000),
                        1,
                        1,
                        h_final_output);
  }
  printf("Warm-up complete.\n");

  printf("Running %d trials...\n", num_trials);
  for (int t = 0; t < num_trials; t++) {
    for (int i = 0; i < nb_tx; ++i) {
      float* temp_float_sig = new float[num_samples * 2];
      generate_random_signal_interleaved(temp_float_sig, num_samples);
      for (int j = 0; j < num_samples; ++j) {
        h_input_samples[i][j].r = (int16_t)temp_float_sig[j * 2];
        h_input_samples[i][j].i = (int16_t)temp_float_sig[j * 2 + 1];
      }
      delete[] temp_float_sig;
    }
    // You could also re-randomize the channel here if desired

    cudaProfilerStart();
    clock_gettime(CLOCK_MONOTONIC, &start);

    vrtsim_cuda_process(gpu_context,
                        h_input_samples.data(),
                        num_samples,
                        nb_tx,
                        nb_rx,
                        h_channel_desc,
                        1.0f,
                        1.0 / (30720 * 2000),
                        1,
                        1,
                        h_final_output);

    clock_gettime(CLOCK_MONOTONIC, &end);
    cudaProfilerStop();
    total_gpu_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
  }
  printf("Finished.\n\n");

  vrtsim_cuda_shutdown(gpu_context);

  double avg_gpu_us = (total_gpu_ns / num_trials) / 1000.0;
  double total_samples_processed = (double)nb_rx * num_samples;
  double gpu_throughput_gsps = total_samples_processed / (avg_gpu_us * 1000.0);

  printf("+----------------------------------+--------------------------+\n");
  printf("| %-32s | %-24s |\n", "Performance Metric", "Value");
  printf("+----------------------------------+--------------------------+\n");
  printf("| %-32s | %-24.2f |\n", "Avg GPU Time per Call (us)", avg_gpu_us);
  printf("| %-32s | %-24.3f |\n", "Est. Throughput (GSPS)", gpu_throughput_gsps);
  printf("+----------------------------------+--------------------------+\n");

  for (int i = 0; i < nb_tx; ++i)
    delete[] h_input_samples[i];
  cudaFreeHost(h_final_output);
  free_manual_channel_desc(h_channel_desc);

  return 0;
}

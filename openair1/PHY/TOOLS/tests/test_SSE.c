/**
 * @file test_cpu_versions.c
 * @brief Standalone test to verify correctness and measure performance of the
 * double-precision multipath_channel function, with added debug output.
 * @author Nika Ghaderi & Gemini
 * @date July 25, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h> 


#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "common/utils/utils.h"
#include "common/utils/LOG/log.h"


#include "channel_io.h" 


configmodule_interface_t *uniqCfg = NULL;


void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}


void generate_random_signal_double(double **sig_re, double **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (double)rand() / (double)RAND_MAX;
            sig_im[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

double calculate_checksum(double **sig_re, double **sig_im, int nb_rx, int num_samples) {
    double checksum = 0.0;
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            checksum += sig_re[i][j];
            checksum -= sig_im[i][j];
        }
    }
    return checksum;
}





int main(int argc, char **argv) {
    
    logInit();
    randominit(1); 
    
    
    int nb_tx = 2;
    int nb_rx = 2;
    int num_samples = 30720;
    int num_trials = 100;
    SCM_t channel_model = TDL_A;
    channel_desc_t *chan_desc = NULL;
    const char* channel_filename = "test_channel.bin";

    
    if (argc > 1 && strcmp(argv[1], "--use-channel") == 0) {
        printf("Mode: Loading channel from %s\n", channel_filename);
        chan_desc = load_channel(channel_filename);
        if (!chan_desc) {
            fprintf(stderr, "Error: Could not load channel file.\n");
            exit(1);
        }
    } else {
        printf("Mode: Generating new channel and saving to %s\n", channel_filename);
        chan_desc = new_channel_desc_scm(nb_tx, nb_rx, channel_model, 30.72, 0, 0, 0.03, 0, 0, 0, 0, 0, 0);
        random_channel(chan_desc, 0); 
        if (save_channel(chan_desc, channel_filename) != 0) {
            fprintf(stderr, "Error: Could not save channel file.\n");
            exit(1);
        }
    }

    #ifdef CHANNEL_SSE
        printf("Version: SSE\n");
    #else
        printf("Version: Standard C\n");
    #endif


    

    double **s_re = malloc(nb_tx * sizeof(double *));
    double **s_im = malloc(nb_tx * sizeof(double *));
    double **r_re = malloc(nb_rx * sizeof(double *));
    double **r_im = malloc(nb_rx * sizeof(double *));

    for (int i=0; i<nb_tx; i++) { s_re[i] = malloc(num_samples * sizeof(double)); s_im[i] = malloc(num_samples * sizeof(double)); }
    for (int i=0; i<nb_rx; i++) { r_re[i] = malloc(num_samples * sizeof(double)); r_im[i] = malloc(num_samples * sizeof(double)); }

   generate_random_signal_double(s_re, s_im, nb_tx, num_samples);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int t = 0; t < num_trials; t++) {
        
        multipath_channel(chan_desc, s_re, s_im, r_re, r_im, num_samples, 1, 0);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    double avg_us = (total_ns / num_trials) / 1000.0;

    double checksum = calculate_checksum(r_re, r_im, nb_rx, num_samples);
    printf("--------------------------------------------------\n");
    printf("Average Execution Time: %.2f us\n", avg_us);
    printf("Output Checksum:        %f\n", checksum);
    printf("--------------------------------------------------\n");

    
    printf("First 10 output samples for antenna 0:\n");
    for (int i = 0; i < 10; i++) {
        
        printf("Sample %2d: (%12.8f, %12.8f)\n", i, r_re[0][i], r_im[0][i]);
    }
    printf("--------------------------------------------------\n");
    printf("Test finished.\n");

    
    for (int i=0; i<nb_tx; i++) { free(s_re[i]); free(s_im[i]); }
    for (int i=0; i<nb_rx; i++) { free(r_re[i]); free(r_im[i]); }
    free(s_re); free(s_im); free(r_re); free(r_im);
    free_channel_desc_scm(chan_desc);

    return 0;
}

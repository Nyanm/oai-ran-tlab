#include "channel_io.h"
#include <string.h>

configmodule_interface_t *uniqCfg = NULL;


typedef enum {
    MODE_STD_DOUBLE,
    MODE_SSE_DOUBLE,
    MODE_STD_FLOAT,
    MODE_SSE_FLOAT
} test_mode_t;

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

double calculate_checksum_double(double **sig_re, double **sig_im, int nb_rx, int num_samples) {
    double checksum = 0.0;
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            checksum += sig_re[i][j];
            checksum -= sig_im[i][j];
        }
    }
    return checksum;
}


void generate_random_signal_float(float **sig_re, float **sig_im, int nb_ant, int num_samples) {
    for (int i = 0; i < nb_ant; i++) {
        for (int j = 0; j < num_samples; j++) {
            sig_re[i][j] = (float)rand() / (float)RAND_MAX;
            sig_im[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

double calculate_checksum_float(float **sig_re, float **sig_im, int nb_rx, int num_samples) {
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
    const char* channel_filename = "test_channel.bin";
    test_mode_t mode = MODE_STD_DOUBLE;
    int use_channel_file = 0;

    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--use-channel") == 0) {
            use_channel_file = 1;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "std_double") == 0) mode = MODE_STD_DOUBLE;
            else if (strcmp(argv[i+1], "sse_double") == 0) mode = MODE_SSE_DOUBLE;
            else if (strcmp(argv[i+1], "std_float") == 0) mode = MODE_STD_FLOAT;
            else if (strcmp(argv[i+1], "sse_float") == 0) mode = MODE_SSE_FLOAT;
            i++; 
        }
    }
    
    channel_desc_t *chan_desc = new_channel_desc_scm(nb_tx, nb_rx, channel_model, 30.72, 0, 0, 0.03, 0, 0, 0, 0, 0, 0);

    if (use_channel_file) {
        if (load_channel_taps(chan_desc, channel_filename) != 0) {
            fprintf(stderr, "Error: Could not load channel file.\n"); exit(1);
        }
    } else {
        printf("Mode: Generating new channel taps and saving to %s\n", channel_filename);
        random_channel(chan_desc, 0); 
        if (save_channel_taps(chan_desc, channel_filename) != 0) {
            fprintf(stderr, "Error: Could not save channel file.\n"); exit(1);
        }
    }

    struct timespec start, end;
    double avg_us = 0;
    double checksum = 0;
    const char* mode_str = "";

    
    if (mode == MODE_STD_DOUBLE || mode == MODE_SSE_DOUBLE) {
        #ifndef CHANNEL_SSE
            if (mode == MODE_SSE_DOUBLE) {
                fprintf(stderr, "Error: SSE mode requested but not compiled with CHANNEL_SSE flag.\n"); exit(1);
            }
        #endif
        mode_str = (mode == MODE_STD_DOUBLE) ? "Standard C (double)" : "SSE (double)";
        
        double **s_re = malloc(nb_tx * sizeof(double *));
        double **s_im = malloc(nb_tx * sizeof(double *));
        double **r_re = malloc(nb_rx * sizeof(double *));
        double **r_im = malloc(nb_rx * sizeof(double *));
        for (int i=0; i<nb_tx; i++) { s_re[i] = malloc(num_samples * sizeof(double)); s_im[i] = malloc(num_samples * sizeof(double)); }
        for (int i=0; i<nb_rx; i++) { r_re[i] = malloc(num_samples * sizeof(double)); r_im[i] = malloc(num_samples * sizeof(double)); }
        
        generate_random_signal_double(s_re, s_im, nb_tx, num_samples);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int t = 0; t < num_trials; t++) {
            multipath_channel(chan_desc, s_re, s_im, r_re, r_im, num_samples, 1, 0);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        checksum = calculate_checksum_double(r_re, r_im, nb_rx, num_samples);

        for (int i=0; i<nb_tx; i++) { free(s_re[i]); free(s_im[i]); }
        for (int i=0; i<nb_rx; i++) { free(r_re[i]); free(r_im[i]); }
        free(s_re); free(s_im); free(r_re); free(r_im);

    } else { 
        #ifndef CHANNEL_SSE
            if (mode == MODE_SSE_FLOAT) {
                fprintf(stderr, "Error: SSE float mode requested but not compiled with CHANNEL_SSE flag.\n"); exit(1);
            }
        #endif
        mode_str = (mode == MODE_STD_FLOAT) ? "Standard C (float)" : "SSE (float)";

        float **s_re = malloc(nb_tx * sizeof(float *));
        float **s_im = malloc(nb_tx * sizeof(float *));
        float **r_re = malloc(nb_rx * sizeof(float *));
        float **r_im = malloc(nb_rx * sizeof(float *));
        for (int i=0; i<nb_tx; i++) { s_re[i] = malloc(num_samples * sizeof(float)); s_im[i] = malloc(num_samples * sizeof(float)); }
        for (int i=0; i<nb_rx; i++) { r_re[i] = malloc(num_samples * sizeof(float)); r_im[i] = malloc(num_samples * sizeof(float)); }
        
        generate_random_signal_float(s_re, s_im, nb_tx, num_samples);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int t = 0; t < num_trials; t++) {
            multipath_channel_float(chan_desc, s_re, s_im, r_re, r_im, num_samples, 1, 0);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        checksum = calculate_checksum_float(r_re, r_im, nb_rx, num_samples);

        for (int i=0; i<nb_tx; i++) { free(s_re[i]); free(s_im[i]); }
        for (int i=0; i<nb_rx; i++) { free(r_re[i]); free(r_im[i]); }
        free(s_re); free(s_im); free(r_re); free(r_im);
    }
    
    double total_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    avg_us = (total_ns / num_trials) / 1000.0;
    
    printf("Mode:%s,Time:%.2f,Checksum:%f\n", mode_str, avg_us, checksum);
    
    free_channel_desc_scm(chan_desc);
    return 0;
}
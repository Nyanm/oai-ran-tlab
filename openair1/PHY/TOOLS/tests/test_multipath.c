/**
 * @file test_multipath.c
 * @brief Test harness for the non-SSE multipath_channel function.
 * @author Nika Ghaderi & Gemini
 * @date July 21, 2025
 *
 * This file should be placed in: openairinterface5g/openair1/PHY/TOOLS/tests/
 * It is designed to be built using the project's CMake build system.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- OAI Header Includes ---
#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "common/utils/LOG/log.h" // Needed for the logInit() function

// --- Function Prototypes for functions NOT in a header ---
// This forward declaration is needed because the function is in a separate .c file.
void fill_channel_desc(channel_desc_t *chan_desc,
                       uint8_t nb_tx,
                       uint8_t nb_rx,
                       uint8_t nb_taps,
                       uint8_t channel_length,
                       double *amps,
                       double *delays,
                       struct complexd *R_sqrt,
                       double Td,
                       double sampling_rate,
                       double channel_bandwidth,
                       double ricean_factor,
                       double aoa,
                       double forgetting_factor,
                       double max_Doppler,
                       uint64_t channel_offset,
                       double path_loss_dB,
                       uint8_t random_aoa);

// --- Global Variable Definitions ---
// We must define the global variables that the linked libraries expect to find.
#include "common/config/config_userapi.h"
configmodule_interface_t *uniqCfg = NULL;

// --- Dummy Implementations for High-Level Functions ---
// We still need a dummy for exit_function, as it's a high-level application function.
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    printf("Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

// --- Test Configuration ---
#define NB_TX 1
#define NB_RX 1
#define SIGNAL_LENGTH 1024
#define SAMPLING_RATE 7.68

int main() {
    // Initialize the OAI logging system. This is the crucial step.
    logInit();

    // Seed the random number generator
    randominit(time(NULL));
    tableNor(time(NULL));

    printf("--- Multipath Channel Test Harness ---\n");

    // --- 1. Memory Allocation ---
    printf("1. Allocating memory...\n");
    channel_desc_t *channel_desc = (channel_desc_t *)malloc(sizeof(channel_desc_t));
    memset(channel_desc, 0, sizeof(channel_desc_t));

    // Statically declare arrays of pointers to match the function prototype in sim.h
    double *tx_sig_re[NB_ANTENNAS_TX];
    double *tx_sig_im[NB_ANTENNAS_TX];
    double *rx_sig_re[NB_ANTENNAS_RX];
    double *rx_sig_im[NB_ANTENNAS_RX];

    for (int i = 0; i < NB_TX; i++) {
        tx_sig_re[i] = (double *)malloc(SIGNAL_LENGTH * sizeof(double));
        tx_sig_im[i] = (double *)malloc(SIGNAL_LENGTH * sizeof(double));
        memset(tx_sig_re[i], 0, SIGNAL_LENGTH * sizeof(double));
        memset(tx_sig_im[i], 0, SIGNAL_LENGTH * sizeof(double));
    }

    for (int i = 0; i < NB_RX; i++) {
        rx_sig_re[i] = (double *)malloc(SIGNAL_LENGTH * sizeof(double));
        rx_sig_im[i] = (double *)malloc(SIGNAL_LENGTH * sizeof(double));
        memset(rx_sig_re[i], 0, SIGNAL_LENGTH * sizeof(double));
        memset(rx_sig_im[i], 0, SIGNAL_LENGTH * sizeof(double));
    }

    // --- 2. Create an Input Signal ---
    printf("2. Creating a simple input signal (impulse)...\n");
    tx_sig_re[0][10] = 1.0;
    tx_sig_im[0][10] = 0.0;

    // --- 3. Initialize the Channel Model ---
    printf("3. Initializing a simple channel model (Rayleigh, 3 taps)...\n");
    int nb_taps = 3;
    double amps[] = {0.5, 0.3, 0.2};
    double delays[] = {0.0, 0.2, 0.5};
    double Td = 0.5;
    int channel_length = 100;

    fill_channel_desc(channel_desc,
                      NB_TX, NB_RX,
                      nb_taps, channel_length,
                      amps, delays,
                      NULL, Td, SAMPLING_RATE, 5.0,
                      1.0, 0.0, 1.0, 0.0,
                      0, 0.0, 0);
    
                      channel_desc->normalization_ch_factor = 1.0;

    // --- 4. Generate the Channel Impulse Response ---
    printf("4. Calling random_channel() to generate CIR...\n");
    random_channel(channel_desc, 0);

    // --- 5. Run the Multipath Channel Function ---
    printf("5. Calling multipath_channel() to process the signal...\n");
    multipath_channel(channel_desc,
                      tx_sig_re, tx_sig_im,
                      rx_sig_re, rx_sig_im,
                      SIGNAL_LENGTH, 1, 0);

    // --- 6. Print Results ---
    printf("6. Displaying results...\n");
    printf("   Input Signal (first 20 samples):\n");
    for (int i = 0; i < 20; i++) {
        printf("   tx[%02d]: (%+1.2f, %+1.2f)\n", i, tx_sig_re[0][i], tx_sig_im[0][i]);
    }
    printf("\n   Output Signal (first 20 samples):\n");
    for (int i = 0; i < 20; i++) {
        printf("   rx[%02d]: (%+1.4f, %+1.4f)\n", i, rx_sig_re[0][i], rx_sig_im[0][i]);
    }

    // --- 7. Cleanup ---
    printf("\n7. Freeing memory...\n");
    for (int i = 0; i < NB_TX; i++) { free(tx_sig_re[i]); free(tx_sig_im[i]); }
    for (int i = 0; i < NB_RX; i++) { free(rx_sig_re[i]); free(rx_sig_im[i]); }
    free(channel_desc);

    printf("\n--- Test complete. ---\n");
    return 0;
}

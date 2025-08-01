#ifndef CHANNEL_IO_H
#define CHANNEL_IO_H

#include "SIMULATION/TOOLS/sim.h"

// A struct to hold only the essential dimensions for verification.
typedef struct {
    int nb_tx;
    int nb_rx;
    int channel_length;
} channel_dims_t;

// Saves only the essential dimensions and random taps from a channel descriptor.
int save_channel_taps(const channel_desc_t* desc, const char* filename);

// Loads random taps from a file into a pre-existing, valid channel descriptor.
int load_channel_taps(channel_desc_t* desc, const char* filename);

#endif // CHANNEL_IO_H
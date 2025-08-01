#ifndef CHANNEL_IO_H
#define CHANNEL_IO_H

#include "SIMULATION/TOOLS/sim.h"


int save_channel(const channel_desc_t* desc, const char* filename);


channel_desc_t* load_channel(const char* filename);

#endif 
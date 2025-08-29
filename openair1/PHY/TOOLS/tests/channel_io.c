#include <stdio.h>
#include <stdlib.h>
#include "channel_io.h"

int save_channel_taps(const channel_desc_t* desc, const char* filename)
{
  FILE* f = fopen(filename, "wb");
  if (!f)
    return -1;

  // 1. Write the essential dimensions.
  channel_dims_t dims = {desc->nb_tx, desc->nb_rx, desc->channel_length};
  fwrite(&dims, sizeof(channel_dims_t), 1, f);

  // 2. Write the raw channel coefficient data.
  int num_links = desc->nb_tx * desc->nb_rx;
  for (int i = 0; i < num_links; i++) {
    fwrite(desc->ch[i], sizeof(struct complexd), desc->channel_length, f);
  }
  fclose(f);
  return 0;
}

int load_channel_taps(channel_desc_t* desc, const char* filename)
{
  FILE* f = fopen(filename, "rb");
  if (!f)
    return -1;

  // 1. Read the dimensions for verification.
  channel_dims_t dims;
  if (fread(&dims, sizeof(channel_dims_t), 1, f) != 1) {
    fclose(f);
    return -1;
  }

  // 2. Sanity check to ensure the file matches the expected channel structure.
  if (dims.nb_tx != desc->nb_tx || dims.nb_rx != desc->nb_rx || dims.channel_length != desc->channel_length) {
    fprintf(stderr, "Error: Channel file dimensions do not match current configuration.\n");
    fclose(f);
    return -1;
  }

  // 3. Read the raw coefficients into the pre-allocated channel descriptor.
  int num_links = desc->nb_tx * desc->nb_rx;
  for (int i = 0; i < num_links; i++) {
    if (fread(desc->ch[i], sizeof(struct complexd), desc->channel_length, f) != desc->channel_length) {
      fclose(f);
      return -1; // Read error
    }
  }
  fclose(f);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include "channel_io.h"

int save_channel(const channel_desc_t* desc, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    
    fwrite(desc, sizeof(channel_desc_t), 1, f);

    
    int num_links = desc->nb_tx * desc->nb_rx;
    for (int i = 0; i < num_links; i++) {
        fwrite(desc->ch[i], sizeof(struct complexd), desc->channel_length, f);
    }

    fclose(f);
    return 0;
}

channel_desc_t* load_channel(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    
    channel_desc_t* desc = (channel_desc_t*)malloc(sizeof(channel_desc_t));
    if (fread(desc, sizeof(channel_desc_t), 1, f) != 1) {
        fclose(f);
        free(desc);
        return NULL;
    }

    
    int num_links = desc->nb_tx * desc->nb_rx;
    desc->ch = (struct complexd**)malloc(num_links * sizeof(struct complexd*));
    for (int i = 0; i < num_links; i++) {
        desc->ch[i] = (struct complexd*)malloc(desc->channel_length * sizeof(struct complexd));
        if (fread(desc->ch[i], sizeof(struct complexd), desc->channel_length, f) != desc->channel_length) {
            
            fclose(f);
            
            return NULL;
        }
    }

    fclose(f);
    return desc;
}
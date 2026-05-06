#include "csi_rb_logging.h"
#include <dlfcn.h>
#include <stdio.h>

typedef void (*csi_rb_logging_callback_t)(
    const void *ue,
    const void *proc,
    const void *csi_rs_estimated_channel_freq,
    const void *csirs_config_pdu);

csi_rb_logging_callback_t csi_rb_logging_callback = NULL;
int csi_rb_logging_enabled = 0;

static int load_csi_logging_library(void) {
  void *handle = dlopen("libcsi_logging.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "Warning: CSI logging library not found\n");
    return -1;
  }
  
  csi_rb_logging_callback = dlsym(handle, "csi_rb_logging_callback_impl");
  if (!csi_rb_logging_callback) {
    fprintf(stderr, "Warning: csi_rb_logging_callback_impl not found\n");
    dlclose(handle);
    return -1;
  }
  
  return 0;
}

int csi_rb_logging_init(void) {
  if (load_csi_logging_library() < 0)
    return -1;
  
  csi_rb_logging_enabled = 1;
  fprintf(stderr, "CSI per-RB logging initialized\n");
  return 0;
}

void csi_rb_logging_exit(void) {
  csi_rb_logging_enabled = 0;
}

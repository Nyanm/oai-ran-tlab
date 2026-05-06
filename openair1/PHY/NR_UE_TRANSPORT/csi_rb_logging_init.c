#include "csi_rb_logging.h"
#include <stdio.h>

__attribute__((constructor))
static void csi_logging_constructor(void) {
  fprintf(stderr, "[CSI] Loading CSI per-RB logging module...\n");
  if (csi_rb_logging_init() == 0) {
    fprintf(stderr, "[CSI] CSI logging module loaded successfully\n");
  } else {
    fprintf(stderr, "[CSI] Failed to load CSI logging module\n");
  }
}

__attribute__((destructor))
static void csi_logging_destructor(void) {
  fprintf(stderr, "[CSI] Unloading CSI logging module...\n");
  csi_rb_logging_exit();
}

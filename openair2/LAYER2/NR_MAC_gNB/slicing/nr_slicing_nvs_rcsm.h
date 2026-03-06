#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

#ifndef NR_SLICING_NVS_RCSM_H
#define NR_SLICING_NVS_RCSM_H

#define NVS_SLICING 20
#define MAX_NVS_SLICES 10
#define BETA 0.001f

typedef struct {
  enum nvs_type {
    NVS_RATE,
    NVS_RES
  } type;
  union {
    struct {
      float Mbps_reserved;
      float Mbps_reference;
    };
    struct {
      float pct_reserved;
    };
  };
} nvs_nr_slice_param_t;

nr_pp_impl_param_dl_t nvs_nr_rcsm_dl_init(module_id_t mod_id);

#endif /* NR_SLICING_NVS_RCSM_H */

#ifndef NR_SLICING_COMMON_H
#define NR_SLICING_COMMON_H

#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"

typedef struct nr_slice_s {
  /// Arbitrary ID
  slice_id_t id;
  /// 3GPP slice id
  nssai_t nssai;
  /// Arbitrary label
  char *label;

  union {
    nr_dl_sched_algo_t dl_algo;
    nr_ul_sched_algo_t ul_algo;
  };

  /// A specific algorithm's implementation parameters
  void *algo_data;
  /// Internal data that might be kept alongside a slice's params
  void *int_data;
  // list of users in this slice
  int num_UEs;
  NR_UE_info_t *UE_list[MAX_MOBILES_PER_GNB+1];
} nr_slice_t;

typedef struct nr_slice_info_s {
  uint8_t num;
  nr_slice_t **s;
} nr_slice_info_t;

bool nr_mac_slice_add_lcid(NR_UE_slice_info_t* slice, const nr_lc_config_t *c, const uint32_t id);

int nr_slicing_get_UE_slice_idx(nr_slice_info_t *si, rnti_t rnti);

seq_arr_t nr_slicing_get_UE_slice_idx_list(nr_slice_info_t *si, rnti_t rnti);

int nr_slicing_get_UE_idx(nr_slice_t *si, rnti_t rnti);

int _nr_exists_slice(uint8_t n, nr_slice_t **s, int id);

nr_slice_t *_nr_add_slice(uint8_t *n, nr_slice_t **s);

nr_slice_t *_nr_remove_slice(uint8_t *n, nr_slice_t **s, int idx);

void nr_slicing_rcsm_dl_add_UE(nr_slice_info_t *si, NR_UE_info_t *new_ue);

void nr_slicing_dl_move_UE_from_list(nr_slice_info_t *si, NR_UE_info_t* assoc_ue, int new_idx);

void nr_slicing_dl_remove_UE_from_list(nr_slice_info_t *si, NR_UE_info_t* rm_ue, int idx);

int remove_nr_slice_dl_from_list(nr_slice_info_t *si, uint8_t slice_idx);


#endif // NR_SLICING_COMMON_H
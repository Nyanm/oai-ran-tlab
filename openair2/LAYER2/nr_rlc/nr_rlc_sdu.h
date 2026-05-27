/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef _NR_RLC_SDU_H_
#define _NR_RLC_SDU_H_

#include <stdint.h>

typedef struct nr_rlc_sdu_t {
  int sn;
  int upper_layer_id;
  char *data;
  int size;
  int retx_count;

  int ref_count;      /* incremented each time the SDU is segmented */
  int free_count;     /* incremented each time a segment is freed
                       * when it equals ref_count we can free the SDU
                       * completely
                       */

  /* for statistics, will be set to 0 after SDU (or first part of it) has
   * been serialized for MAC for the first time so that only the first
   * transmission is used for statistics
   */
  uint64_t time_of_arrival;  /* unit microsecond */

  /* RLC time when the SDU was first received .Used for the HOL-delay
   * in MAC. TODO: unify with time_of_arrival (feed E2AP/KPM in us) 
   */
  uint64_t arrival_ms;
} nr_rlc_sdu_t;

typedef struct nr_rlc_sdu_segment_t {
  nr_rlc_sdu_t *sdu;
  int size;
  int so;
  int is_first;
  int is_last;
  struct nr_rlc_sdu_segment_t *next;
} nr_rlc_sdu_segment_t;

nr_rlc_sdu_segment_t *nr_rlc_new_sdu(
    char *buffer, int size,
    int upper_layer_id);
/* return 1 if the SDU has been freed too, 0 if not (more segments to free) */
int nr_rlc_free_sdu_segment(nr_rlc_sdu_segment_t *sdu);
void nr_rlc_sdu_segment_list_append(nr_rlc_sdu_segment_t **list,
                                    nr_rlc_sdu_segment_t **end,
                                    nr_rlc_sdu_segment_t *sdu);
void nr_rlc_free_sdu_segment_list(nr_rlc_sdu_segment_t *l);

#endif /* _NR_RLC_SDU_H_ */

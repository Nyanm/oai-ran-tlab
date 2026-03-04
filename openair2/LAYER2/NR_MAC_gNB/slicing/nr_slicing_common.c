#include "nr_slicing_common.h"
#include "../mac_proto.h"
#include "assert.h"
#include "common/utils/alg/find.h"

int nr_slicing_get_UE_slice_idx(nr_slice_info_t *si, rnti_t rnti)
{
  for (int s_len = 0; s_len < si->num; s_len++) {
    for (int i = 0; i < MAX_MOBILES_PER_GNB; i++) {
      if (si->s[s_len]->UE_list[i] != NULL) {
        if (si->s[s_len]->UE_list[i]->rnti == rnti) {
          return s_len;
        }
      }
    }
  }
  LOG_E(NR_MAC, "cannot find slice idx for UE rnti 0x%04x\n", rnti);
  return -99;
}

int nr_slicing_get_UE_idx(nr_slice_t *si, rnti_t rnti)
{
  for (int i = 0; i < MAX_MOBILES_PER_GNB; i++) {
    if (si->UE_list[i] != NULL) {
      LOG_D(NR_MAC, "%s(): si->UE_list[%d]->rnti %x map to rnti %x\n", __func__, i, si->UE_list[i]->rnti, rnti);
      if (si->UE_list[i]->rnti == rnti)
        return i;
    }
  }
  LOG_E(NR_MAC, "cannot find ue idx for UE rnti 0x%04x\n", rnti);
  return -99;
}

seq_arr_t nr_slicing_get_UE_slice_idx_list(nr_slice_info_t *si, rnti_t rnti)
{
  assert(si != NULL);

  seq_arr_t list;
  seq_arr_init(&list, sizeof(int));

  for (int s_idx = 0; s_idx < si->num; s_idx++) {
    nr_slice_t *s = si->s[s_idx];
    LOG_D(NR_MAC, "id %d, num UEs %d\n", s->id, s->num_UEs);
    UE_iterator(s->UE_list, UE) {
      if (UE->rnti == rnti) {
        LOG_D(NR_MAC, "%s(): UE rnti %4x is associated with slice idx %d id %d\n", __func__, rnti, s_idx, s->id);
        seq_arr_push_back(&list, &s_idx, sizeof(int));
      }
    }
  }

  return list;
}
/*
// return the list of the slices that UE is associated with
void print_UE_slice_idx_list(nr_slice_info_t *si, rnti_t rnti)
{

  for (int s_len = 0; s_len < si->num; s_len++) {
    for (int i = 0; i < MAX_MOBILES_PER_GNB; i++) {
      if (si->s[s_len]->UE_list[i] != NULL) {
        if (si->s[s_len]->UE_list[i]->rnti == rnti) {
          LOG_D(NR_MAC, "%s(): UE %d is associated with slice idx %d and id %d \n", __func__, rnti, s_len, si->s[s_len]->id);
        }
      }
    }
  }
}

// print the list of the slices that UE is associated with
void print_NR_list(NR_list_t *list)
{
  if (list->head != -1) {
    int current = list->head;
    while (current != -1) {
      LOG_D(NR_MAC, "%s(): UE %d is associated with slice %d \n", __func__, current, list->next[current]);
      current = list->next[current];
    }
  } else {
    LOG_E(NR_MAC, "The list is empty.\n");
  }
}
*/

int _nr_exists_slice(uint8_t n, nr_slice_t **s, int id)
{
  for (int i = 0; i < n; ++i) {
    LOG_D(NR_MAC, "%s(): n %d, s[%d]->id %d, id %d\n", __func__, n ,i, s[i]->id, id);
    if (s[i]->id == id)
      return i;
  }
  return -1;
}

nr_slice_t *_nr_add_slice(uint8_t *n, nr_slice_t **s)
{
  s[*n] = calloc(1, sizeof(nr_slice_t));
  if (!s[*n])
    return NULL;
  *n += 1;
  return s[*n - 1];
}

nr_slice_t *_nr_remove_slice(uint8_t *n, nr_slice_t **s, int idx)
{
  if (idx >= *n)
    return NULL;

  nr_slice_t *sr = s[idx];

  for (int i = idx + 1; i < *n; ++i)
    s[i - 1] = s[i];
  *n -= 1;
  s[*n] = NULL;

  if (sr->label)
    free(sr->label);

  return sr;
}

static bool nssai_matches(nssai_t a_nssai, uint8_t b_sst, const uint32_t *b_sd)
{
  AssertFatal(b_sd == NULL || *b_sd <= 0xffffff, "illegal SD %d\n", *b_sd);
  if (b_sd == NULL) {
    return a_nssai.sst == b_sst && a_nssai.sd == 0;
  } else {
    return a_nssai.sst == b_sst && a_nssai.sd == *b_sd;
  }
}

static bool eq_lcid_config(const void *vval, const void *vit)
{
  const nr_lc_config_t *val = (const nr_lc_config_t *)vval;
  const nr_lc_config_t *it = (const nr_lc_config_t *)vit;
  return it->lcid == val->lcid;
}

static int cmp_lc_config(const void *va, const void *vb)
{
  const nr_lc_config_t *a = (const nr_lc_config_t *)va;
  const nr_lc_config_t *b = (const nr_lc_config_t *)vb;

  if (a->priority < b->priority)
    return -1;
  if (a->priority == b->priority)
    return 0;
  return 1;
}

// Add LCID in sliceInfoDl
bool nr_mac_slice_add_lcid(NR_UE_slice_info_t* slice, const nr_lc_config_t *c, const uint32_t id)
{
  elm_arr_t elm = find_if(&slice->lc_config, (void *) c, eq_lcid_config);
  if (elm.found) {
    LOG_D(NR_MAC, "cannot add LCID %d in slice: already present, updating configuration\n", c->lcid);
    nr_lc_config_t *exist = (nr_lc_config_t *)elm.it;
    *exist = *c;
  } else {
    LOG_D(NR_MAC, "Add LCID %d in slice id %d\n", c->lcid, id);
    seq_arr_push_back(&slice->lc_config, (void*) c, sizeof(*c));
  }
  void *base = seq_arr_front(&slice->lc_config);
  size_t nmemb = seq_arr_size(&slice->lc_config);
  size_t size = sizeof(*c);
  qsort(base, nmemb, size, cmp_lc_config);
  return true;
}

// Use RC SM to add UE to DL slices
void nr_slicing_rcsm_dl_add_UE(nr_slice_info_t *si, NR_UE_info_t *new_ue)
{
  AssertFatal(si->num > 0 && si->s != NULL, "no slices exists, cannot add UEs\n");
  AssertFatal(new_ue != NULL, "UE is NULL\n");

  NR_UE_sched_ctrl_t *sched_ctrl = &new_ue->UE_sched_ctrl;

  reset_nr_list(&new_ue->dl_id);

  for (int i = 0; i < si->num; ++i) {
    nr_slice_t* s = si->s[i];
    for (size_t l = 0; l < seq_arr_size(&sched_ctrl->lc_config); ++l) {
      const nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, l);
      long lcid = c->lcid;
      if (nssai_matches(c->nssai, s->nssai.sst, &s->nssai.sd)) {

        nr_mac_slice_add_lcid(&new_ue->UE_sched_ctrl.sliceInfoDl[i], c, s->id);
        LOG_D(NR_MAC, "Add lcid %ld to DL slice idx %d\n", lcid, i);

        if (!check_nr_list(&new_ue->dl_id, s->id)) {
          add_nr_list(&new_ue->dl_id, s->id);
          LOG_D(NR_MAC, "Add DL slice id %d to new UE rnti %x\n", s->id, new_ue->rnti);
        }

        // make sure new_ue is not in the s->UE_list before add
        bool add_UE = true;
        UE_iterator(s->UE_list, UE) {
          if (UE->rnti == new_ue->rnti) {
            LOG_D(NR_MAC, "UE rnti 0x%04x exists in DL slice idx %d, sst %d, sd %06x\n", new_ue->rnti, i, s->nssai.sst, s->nssai.sd);
            add_UE = false;
            break;
          }
        }
        if (add_UE) {
          bool success = add_UE_to_list(MAX_MOBILES_PER_GNB, s->UE_list, new_ue);
          s->num_UEs += 1;
          AssertFatal(success, "Try to add UE but the s->UE_list is full\n");
          LOG_A(NR_MAC, "Matched DL slice, Add UE rnti 0x%04x to DL slice idx %d, sst %d, sd %06x\n",
                new_ue->rnti, i, s->nssai.sst, s->nssai.sd);
        } else {
          LOG_D(NR_MAC, "Cannot add new UE rnti 0x%04x to DL slice idx %d, num_UEs %d\n",
                new_ue->rnti, i, s->num_UEs);
        }
      } else {
        LOG_D(NR_MAC, "Cannot find matched DL slice (lcid %ld <sst %d sd %d>, DL slice idx %d <sst %d sd %d>), do nothing for UE rnti 0x%04x\n",
              lcid, c->nssai.sst, c->nssai.sd, i, s->nssai.sst, s->nssai.sd, new_ue->rnti);
      }
    }
  }
}

// Remove UE from DL slices list
void nr_slicing_dl_remove_UE_from_list(nr_slice_info_t *si, NR_UE_info_t* rm_ue, int idx)
{
  AssertFatal(si->num > 0 && si->s != NULL, "no slices exists, cannot remove UEs\n");
  AssertFatal(rm_ue != NULL, "UE is NULL\n");

  nr_slice_t* s = si->s[idx];

  // this is for deassociation, in other case that UE disconnects, r->dl_id will be destroyed at delete_nr_ue_data()
  if (check_nr_list(&rm_ue->dl_id, s->id)) {
    remove_nr_list(&rm_ue->dl_id, s->id);
    LOG_I(NR_MAC, "Remove dl id %d from UE rnti 0x%04x\n", s->id, rm_ue->rnti);
  }

  // make sure rm_ue is in the si->s[idx]->UE_list before remove
  bool remove_UE = false;
  UE_iterator(s->UE_list, ue) {
    if (ue->rnti == rm_ue->rnti) {
      LOG_D(NR_MAC, "UE rnti 0x%04x exists in DL slice id %d\n", rm_ue->rnti, s->id);
      remove_UE = true;
      break;
    }
  }
  if (remove_UE) {
    NR_UE_info_t *r = remove_UE_from_list(NR_NB_RA_PROC_MAX, s->UE_list, rm_ue->rnti);
    DevAssert(r == rm_ue);
    si->s[idx]->num_UEs -= 1;
    LOG_A(NR_MAC, "Remove UE rnti 0x%04x from DL slice id %d, num UEs %d\n", rm_ue->rnti, s->id, s->num_UEs);
  }
  //print_UE_slice_idx_list(si, rm_ue->rnti);
}



#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>

#include "assertions.h"
#include "common/utils/LOG/log.h"
#include "common/utils/nr/nr_common.h"

#include "NR_MAC_COMMON/nr_mac.h"
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"

#include "nr_slicing_common.h"
#include "nr_slicing_nvs_rcsm.h"

#include "executables/softmodem-common.h"

extern RAN_CONTEXT_t RC;

#define RET_FAIL(ret, x...) do { LOG_E(MAC, x); return ret; } while (0)

typedef struct {
  float exp; // exponential weight. mov. avg for weight calc
  int   rb;  // number of RBs this slice has been scheduled in last round
  float eff; // effective rate for rate slices
  float beta_eff; // averaging coeff so we average over roughly one second
  int   active;   // activity state for rate slices
} _nvs_int_t;

static int _nvs_nr_admission_control(const nr_slice_info_t *si,
                              const nvs_nr_slice_param_t *p,
                              int idx)
{
  if (p->type != NVS_RATE && p->type != NVS_RES)
    RET_FAIL(-1, "%s(): invalid slice type %d\n", __func__, p->type);
  if (p->type == NVS_RATE && p->Mbps_reserved > p->Mbps_reference)
    RET_FAIL(-1,
             "%s(): a rate slice cannot reserve more than the reference rate\n",
             __func__);
  if (p->type == NVS_RES && p->pct_reserved > 1.0f)
    RET_FAIL(-1, "%s(): cannot reserve more than 1.0\n", __func__);
  float sum_req = 0.0f;
  for (int i = 0; i < si->num; ++i) {
    LOG_D(NR_MAC, "i %d, idx %d\n", i, idx);
    const nvs_nr_slice_param_t *sp = i == idx ? p : si->s[i]->algo_data;
    if (sp->type == NVS_RATE) {
      sum_req += sp->Mbps_reserved / sp->Mbps_reference;
    } else {
      LOG_D(NR_MAC, "slice idx %d, si->num %d, pct_reserved %.2f, sum_req %.2f\n", i, si->num, sp->pct_reserved, sum_req);
      sum_req += sp->pct_reserved;
    }
    LOG_D(NR_MAC, "slice idx %d, sum_req %.2f\n", i, sum_req);
  }
  if (idx < 0) { /* not an existing slice */
    if (p->type == NVS_RATE)
      sum_req += p->Mbps_reserved / p->Mbps_reference;
    else
      sum_req += p->pct_reserved;
  }
  LOG_D(NR_MAC, "slice idx %u, pct_reserved %.2f, sum_req %.2f\n", idx, p->pct_reserved, sum_req);
  if (sum_req > 1.0)
    RET_FAIL(-3,
             "%s(): admission control failed: sum of resources is %f > 1.0\n",
             __func__, sum_req);
  return 0;
}

static int addmod_nvs_nr_slice_dl(nr_slice_info_t *si,
                           int id,
                           nssai_t nssai,
                           char *label,
                           void *algo,
                           void *slice_params_dl)
{
  nvs_nr_slice_param_t *dl = slice_params_dl;
  int index = _nr_exists_slice(si->num, si->s, id);
  if (index < 0 && si->num >= MAX_NVS_SLICES)
    RET_FAIL(-2, "%s(): cannot handle more than %d slices\n", __func__, MAX_NVS_SLICES);

  if (index < 0 && !dl)
    RET_FAIL(-100, "%s(): no parameters for new slice %d, aborting\n", __func__, id);

  if (dl) {
    int rc = _nvs_nr_admission_control(si, dl, index);
    if (rc < 0)
      return rc;
  }

  nr_slice_t *s = NULL;
  if (index >= 0) {
    s = si->s[index];
    if (label) {
      if (s->label) free(s->label);
      s->label = label;
    }
    if (algo && s->dl_algo.run != ((nr_dl_sched_algo_t*)algo)->run) {
      s->dl_algo.unset(&s->dl_algo.data);
      s->dl_algo = *(nr_dl_sched_algo_t*) algo;
      if (!s->dl_algo.data)
        s->dl_algo.data = s->dl_algo.setup();
    }
    if (dl) {
      free(s->algo_data);
      s->algo_data = dl;
    } else { /* we have no parameters: we are done */
      return index;
    }
    s->nssai = nssai;
  } else {
    if (!algo)
      RET_FAIL(-14, "%s(): no scheduler algorithm provided\n", __func__);

    s = _nr_add_slice(&si->num, si->s);
    if (!s)
      RET_FAIL(-4, "%s(): cannot allocate memory for slice\n", __func__);
    s->int_data = malloc(sizeof(_nvs_int_t));
    if (!s->int_data)
      RET_FAIL(-5, "%s(): cannot allocate memory for slice internal data\n", __func__);

    s->id = id;
    s->label = label;
    s->dl_algo = *(nr_dl_sched_algo_t*) algo;
    if (!s->dl_algo.data)
      s->dl_algo.data = s->dl_algo.setup();
    s->algo_data = dl;
    s->nssai = nssai;
  }

  _nvs_int_t *nvs_p = s->int_data;
  /* reset all slice-internal parameters */
  nvs_p->rb = 0;
  nvs_p->active = 0;
  if (dl->type == NVS_RATE) {
    nvs_p->exp = dl->Mbps_reserved / dl->Mbps_reference;
    nvs_p->eff = dl->Mbps_reference;
  } else {
    nvs_p->exp = dl->pct_reserved;
    nvs_p->eff = 0; // not used
  }
  // scale beta so we (roughly) average the eff rate over 1s
  nvs_p->beta_eff = BETA / nvs_p->exp;

  return index < 0 ? si->num - 1 : index;
}

void nvs_nr_dl(gNB_MAC_INST *mac, post_process_pdsch_t *pp_pdsch)
{
  NR_UEs_t *UE_info = &mac->UE_info;
  if (UE_info->connected_ue_list[0] == NULL)
    return;

  /* check if we are supposed to schedule something */
  NR_ServingCellConfigCommon_t *scc = mac->common_channels[0].ServingCellConfigCommon;
  int bw = scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->carrierBandwidth;
  int num_beams = mac->beam_info.beam_allocation ? mac->beam_info.beams_per_period : 1;
  AssertFatal(num_beams == 1, "we do not support multiple beam in PR slicing algorithm\n");
  int remain_n_rb_sched[num_beams];
  int original_n_rb_sched[num_beams];
  for (int i = 0; i < num_beams; i++) {
    remain_n_rb_sched[i] = bw;
    original_n_rb_sched[i] = bw;
  }

  int average_agg_level = 4; // TODO find a better estimation
  int max_sched_ues = bw / (average_agg_level * NR_NB_REG_PER_CCE);

  // FAPI cannot handle more than MAX_DCI_CORESET DCIs
  max_sched_ues = min(max_sched_ues, MAX_DCI_CORESET);

  /* Schedule slices */
  nr_slice_info_t *si = mac->pre_processor_dl.slices;
  int bytes_last_round[MAX_NVS_SLICES] = {0};
  for (int s_idx = 0; s_idx < si->num; ++s_idx) {
    UE_iterator (si->s[s_idx]->UE_list, UE) {
      const NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
      bytes_last_round[s_idx] += UE->mac_stats.dl.current_bytes;

      /* if UE has data or retransmission, mark respective slice as active */
      //const int retx_pid = sched_ctrl->retrans_dl_harq.head;
      //const int retx_slice = sched_ctrl->harq_slice_map[retx_pid];
      const bool active = sched_ctrl->sliceInfoDl[s_idx].num_total_bytes > 0; //|| retx_slice == s_idx;
      ((_nvs_int_t *)si->s[s_idx]->int_data)->active |= active;
    }
  }

  float maxw = 0.0f;
  int maxidx = -1;
  for (int i = 0; i < si->num; ++i) {
    nr_slice_t *s = si->s[i];
    nvs_nr_slice_param_t *p = s->algo_data;
    _nvs_int_t *ip = s->int_data;

    float w = 0.0f;
    if (p->type == NVS_RATE) {
      float inst = 0.0f;
      if (ip->rb > 0) { /* it was scheduled last round */
        /* inst rate: B in last round * 8(bit) / 1000000 (Mbps) * 1000 (1ms) */
        inst = (float)bytes_last_round[i] * 8 / 1000;

        /* Calculate the exponential moving average for throughput */
        ip->eff = (1.0f - ip->beta_eff) * ip->eff + ip->beta_eff * inst;

        /* Calculate the exponential moving average for current with scheduled slice */
        ip->exp += BETA * inst;
      }

      const float rsv = p->Mbps_reserved * min(1.0f, ip->eff / p->Mbps_reference);

      /* Calculate the weight for the slice */
      w = rsv / ip->exp;

      /* Calculate the exponential moving average for next for the slice */
      ip->exp = (1 - BETA) * ip->exp;

      /* Check exp and eff overflow */
      if (ip->rb == 0) {
        if (ip->exp == (1 - BETA) * ip->exp || ip->eff == (1.0f - ip->beta_eff) * ip->eff) {
          /* Reload parameters */
          ip->exp = p->Mbps_reserved / p->Mbps_reference;
          ip->eff = p->Mbps_reference;
        }
      }
    } else { /* NVS CAPACITY */
      /* Calculate the exponential moving average for current with scheduled slice */
      if (ip->rb > 0) {
        ip->exp += BETA;
      }

      /* Calculate the weight for the slice */
      w = p->pct_reserved / (ip->exp);

      /* Calculate the exponential moving average for next for the slice */
      ip->exp = (1.0f - BETA) * ip->exp;
    }
    // LOG_I(NR_MAC, "i %d slice %d type %d ip->exp %f w %f\n", i, s->id, p->type, ip->exp, w);
    ip->rb = 0;

    /* Find the max weight */
    if (w > maxw) {
      maxw = w;
      maxidx = i;
    }
  }

  if (maxidx < 0)
    return;

  /* Statistic */
  static int counter[MAX_NVS_SLICES] = {0};
  static int times = 0;
  counter[maxidx]++;
  times++;
  if (pp_pdsch->frame % 100 == 0 && pp_pdsch->slot == 0) {
    for (int i = 0; i < si->num; ++i) {
      LOG_D(NR_MAC,
            "[NVS Statistic DL] Slice Id %d,  counter %d, probabilities %f\n",
            si->s[i]->id,
            counter[i],
            (float)counter[i] / times);
      counter[i] = 0;
    }
    times = 0;
  }

  // TODO: now is one beam associates to multiple slices
  ((_nvs_int_t *)si->s[maxidx]->int_data)->rb = remain_n_rb_sched[0];

  UE_iterator (si->s[maxidx]->UE_list, UE) {
    UE->UE_sched_ctrl.sched_dl_idx = maxidx;
    LOG_D(NR_MAC, "UE %4x sched_dl_idx %d\n", UE->rnti, UE->UE_sched_ctrl.sched_dl_idx);
    LOG_D(NR_MAC, "%4d.%2d schedule UE %4x at slice idx %d ID %d \n", pp_pdsch->frame, pp_pdsch->slot, UE->rnti, maxidx, si->s[maxidx]->id);
  }

  /* proportional fair scheduling algorithm */
  si->s[maxidx]->dl_algo.run(mac,
                             pp_pdsch,
                             si->s[maxidx]->UE_list,
                             max_sched_ues,
                             num_beams,
                             remain_n_rb_sched,
                             si->s[maxidx]->dl_algo.data);
  for (int i = 0; i < num_beams; i++) {
    if (original_n_rb_sched[i] - remain_n_rb_sched[i] > 0)
      LOG_D(NR_MAC, "%4d.%2d schedule %d RBs at slice idx %d ID %d \n", pp_pdsch->frame, pp_pdsch->slot, original_n_rb_sched[i] - remain_n_rb_sched[i], maxidx, si->s[maxidx]->id);

    if (remain_n_rb_sched[i] == original_n_rb_sched[i]) // if no RBs have been used mark as inactive
      ((_nvs_int_t *)si->s[maxidx]->int_data)->active = 0;
  }
}

void nvs_nr_destroy(nr_slice_info_t **si)
{
  const int n_dl = (*si)->num;
  (*si)->num = 0;
  for (int i = 0; i < n_dl; ++i) {
    nr_slice_t *s = (*si)->s[i];
    if (s->label)
      free(s->label);
    free(s->algo_data);
    free(s->int_data);
    free(s);
  }
  free((*si)->s);
}

nr_pp_impl_param_dl_t nvs_nr_rcsm_dl_init(module_id_t mod_id)
{
  nr_slice_info_t *si = calloc(1, sizeof(nr_slice_info_t));
  DevAssert(si);

  si->num = 0;
  si->s = calloc(MAX_NVS_SLICES, sizeof(*si->s));
  DevAssert(si->s);

  /* insert default slice, all resources */
  nvs_nr_slice_param_t *dlp = malloc(sizeof(nvs_nr_slice_param_t));
  DevAssert(dlp);
  dlp->type = NVS_RES;
  dlp->pct_reserved = 1.0f;

  nr_dl_sched_algo_t *algo = &RC.nrmac[mod_id]->pre_processor_dl.dl_algo;
  algo->data = NULL;

  nssai_t nssai = {.sst = 0, .sd = 0};

  const int rc = addmod_nvs_nr_slice_dl(si, 0, nssai, strdup("default"), algo, dlp);
  DevAssert(0 == rc);

  LOG_A(NR_MAC, "%s(): add default DL slice id 0, label default, sst %d, sd %d\n",__func__, nssai.sst, nssai.sd);

  nr_pp_impl_param_dl_t nvs;
  nvs.algorithm = NVS_SLICING;
  nvs.get_UE_slice_idx = nr_slicing_get_UE_slice_idx;
  nvs.get_UE_slice_idx_list = nr_slicing_get_UE_slice_idx_list;
  nvs.get_UE_idx = nr_slicing_get_UE_idx;
  nvs.add_UE = nr_slicing_rcsm_dl_add_UE;
  nvs.move_UE = NULL; // move UE function is not defined by the RC ctrl
  nvs.remove_UE = nr_slicing_dl_remove_UE_from_list;
  nvs.addmod_slice = addmod_nvs_nr_slice_dl;
  nvs.remove_slice = NULL; // remove function is not defined by the RC ctrl
  nvs.dl = nvs_nr_dl;
  // current DL algo becomes default scheduler
  nvs.dl_algo = *algo;
  nvs.destroy = nvs_nr_destroy;
  nvs.slices = si;

  return nvs;
}

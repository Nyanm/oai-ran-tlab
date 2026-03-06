#include "ran_func_rc_ctrl_slice.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/rc_sm_id.h"
#include "common/utils/ds/byte_array.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "openair2/LAYER2/NR_MAC_gNB/slicing/nr_slicing_common.h"
#include "openair2/LAYER2/NR_MAC_gNB/slicing/nr_slicing_nvs_rcsm.h"
#include "openair2/RRC/NR/rrc_gNB_UE_context.h"
#include "openair2/COMMON/f1ap_messages_types.h"
#include "openair2/F1AP/f1ap_ids.h"

typedef struct {
  uint16_t mcc;
  uint16_t mnc;
} mccmnc_t;

// 38.413 clause 9.3.3.5
static
mccmnc_t convert_plmn_to_mccmnc(uint8_t* plmn)
{
  assert(plmn != NULL);

  mccmnc_t res = {0};

  uint8_t mcc1 = plmn[0] & 0x0F; // digit 1
  uint8_t mcc2 = (plmn[0] >> 4) & 0x0F; // digit 2
  uint8_t mcc3 = plmn[1] & 0x0F; // digit 3

  res.mcc = mcc1*100 + mcc2*10 + mcc3;

  uint8_t mnc1 = (plmn[1] >> 4 ) & 0x0F; // digit 4
  uint8_t mnc2 = plmn[2] & 0x0F; // digit 5
  uint8_t mnc3 = (plmn[2] >> 4)  & 0x0F; // digit 6

  if (mnc1 != 0x0f){
    res.mnc = mnc1*100 + mnc2*10 + mnc3;
  } else {
    res.mnc = mnc2*10 + mnc3;
  }

  return res;
}

static
int add_mod_dl_slice(int mod_id,
                        slice_algorithm_e current_algo,
                        int8_t id,
                        nssai_t nssai,
                        char* label,
                        void *params)
{
  nr_pp_impl_param_dl_t *dl = &RC.nrmac[mod_id]->pre_processor_dl;
  char *slice_algo = NULL;
  if (current_algo == NVS_SLICING) {
    nvs_nr_slice_param_t* nvs_params = (nvs_nr_slice_param_t *)params;
    if (!nvs_params) return -1;
    slice_algo = strdup("NVS_SLICING");
    if (nvs_params->type == NVS_RATE) {
      LOG_A(NR_MAC, "add/mod DL slice sched algo %s (type: NVS_RATE), Mbps_reserved %.2f, Mbps_reference %.2f, ue sched algo %s\n",
            slice_algo, nvs_params->Mbps_reserved, nvs_params->Mbps_reference, dl->dl_algo.name);
    } else if (nvs_params->type == NVS_RES) {
      LOG_A(NR_MAC, "add/mod DL slice sched algo %s (type: NVS_RES), pct_reserved %.2f, ue sched algo %s\n",
            slice_algo, nvs_params->pct_reserved, dl->dl_algo.name);
    } else {
      assert(0 != 0 && "Unknow nvs_params->type");
    }

  } else {
    assert(0 != 0 && "Unknow current_algo, only support PR algo for DL scheduler");
  }

  void *algo = &dl->dl_algo;
  char *l = NULL;
  if (label)
    l = strdup(label);
  LOG_A(NR_MAC, "add/mod DL slice id %d, label %s\n", id, l);
  return dl->addmod_slice(dl->slices, id, nssai, l, algo, (void*) params);
}

static void set_new_dl_slice_algo(int mod_id, int algo)
{
  gNB_MAC_INST *nrmac = RC.nrmac[mod_id];
  assert(nrmac);
  NR_SCHED_LOCK(&nrmac->sched_lock);

  nr_pp_impl_param_dl_t dl = nrmac->pre_processor_dl;
  switch (algo) {
    case NVS_SLICING:
      nrmac->pre_processor_dl = nvs_nr_rcsm_dl_init(mod_id);
      break;
    default:
      nrmac->pre_processor_dl.algorithm = 0;
      nrmac->pre_processor_dl = nr_init_dlsch_preprocessor(0); // assume CC_id = 0
      nrmac->pre_processor_dl.slices = NULL;
      break;
  }
  if (dl.slices)
    dl.destroy(&dl.slices);
  if (dl.dl_algo.data)
    dl.dl_algo.unset(&dl.dl_algo.data);
  NR_SCHED_UNLOCK(&nrmac->sched_lock);
}

static bool nssai_matches(nssai_t a_nssai, uint8_t b_sst, const uint32_t *b_sd)
{
  AssertFatal(b_sd == NULL || *b_sd <= 0xffffff, "illegal SD %d\n", *b_sd);
  if (b_sd == NULL) {
    return a_nssai.sst == b_sst && a_nssai.sd == 0xffffff;
  } else {
    return a_nssai.sst == b_sst && a_nssai.sd == *b_sd;
  }
}

static ngran_node_t get_e2_node_type(void)
{
  ngran_node_t node_type = 0;

#if defined(NGRAN_GNB_DU) && defined(NGRAN_GNB_CUUP) && defined(NGRAN_GNB_CUCP)
  node_type = RC.nrrrc[0]->node_type;
#elif defined (NGRAN_GNB_CUUP)
  node_type =  ngran_gNB_CUUP;
#endif

  return node_type;
}

static
mccmnc_t get_ue_mcc_mnc(uint16_t rnti)
{
  mccmnc_t dst = {0};
  const ngran_node_t node_type = get_e2_node_type();
  if(node_type == ngran_gNB){
    rrc_gNB_ue_context_t* rrc_ue_context_list = rrc_gNB_get_ue_context_by_rnti_any_du(RC.nrrrc[0], rnti);
    dst.mcc = rrc_ue_context_list->ue_context.ue_guami.plmn.mcc;
    dst.mnc = rrc_ue_context_list->ue_context.ue_guami.plmn.mnc;
  } else {
    gNB_MAC_INST const *mac = RC.nrmac[0];
    f1_config_t const* f1_config = &mac->f1_config;
    f1ap_setup_req_t const* setup_req = f1_config->setup_req;
    // Here we suppose that all the cells share the same MCC and MNC
    // True?
    f1ap_served_cell_info_t const* info = &setup_req->cell[0].info;

    dst.mcc = info->plmn.mcc;
    dst.mnc = info->plmn.mnc;
  }

  return dst;
}

static
int8_t get_slice_idx_nssai(nr_slice_info_t *slice_info, nssai_t src)
{
  assert(slice_info->num > 0 && "no default slice exists, check init slice function");

  for (size_t i = 0; i < slice_info->num; ++i) {
    nr_slice_t* s = slice_info->s[i];
    if (nssai_matches(s->nssai, src.sst, &src.sd)) {
      LOG_D(NR_MAC, "found existing slice idx %zu, sst %d, sd %d\n", i, s->nssai.sst, s->nssai.sd);
      return i;
    }
  }

  LOG_I(NR_MAC, "not found nssai from existing slices, create a new slice with idx %d, sst %d, sd 0x%06x\n", slice_info->num, src.sst, src.sd);
  return slice_info->num;
}

bool add_mod_rc_slice(int mod_id, size_t slices_len, ran_param_list_t* lst)
{
  gNB_MAC_INST *nrmac = RC.nrmac[mod_id];
  assert(nrmac);

  int current_algo = nrmac->pre_processor_dl.algorithm;
  // use PR algorithm by default
  int new_algo = 0;
  new_algo = NVS_SLICING;

  if (current_algo != new_algo) {
    set_new_dl_slice_algo(mod_id, new_algo);
    current_algo = new_algo;
    if (new_algo > 0)
      LOG_D(NR_MAC, "set new algorithm %d\n", current_algo);
    else
      LOG_W(NR_MAC, "reset slicing algorithm as NONE\n");
  }

  for (size_t i = 0; i < slices_len; ++i) {
    lst_ran_param_t* RRM_Policy_Ratio_Group = &lst->lst_ran_param[i];
    //Bug in rc_enc_asn.c:1003, asn didn't define ran_param_id for lst_ran_param_t...
    //assert(RRM_Policy_Ratio_Group->ran_param_id == RRM_Policy_Ratio_Group_8_4_3_6 && "wrong RRM_Policy_Ratio_Group id");
    assert(RRM_Policy_Ratio_Group->ran_param_struct.sz_ran_param_struct == 4 && "wrong RRM_Policy_Ratio_Group->ran_param_struct.sz_ran_param_struct");
    assert(RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct != NULL && "NULL RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct");

    seq_ran_param_t* RRM_Policy = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[0];
    assert(RRM_Policy->ran_param_id == RRM_Policy_8_4_3_6 && "wrong RRM_Policy id");
    assert(RRM_Policy->ran_param_val.type == STRUCTURE_RAN_PARAMETER_VAL_TYPE && "wrong RRM_Policy type");
    assert(RRM_Policy->ran_param_val.strct != NULL && "NULL RRM_Policy->ran_param_val.strct");
    assert(RRM_Policy->ran_param_val.strct->sz_ran_param_struct == 1 && "wrong RRM_Policy->ran_param_val.strct->sz_ran_param_struct");
    assert(RRM_Policy->ran_param_val.strct->ran_param_struct != NULL && "NULL RRM_Policy->ran_param_val.strct->ran_param_struct");

    seq_ran_param_t* RRM_Policy_Member_List = &RRM_Policy->ran_param_val.strct->ran_param_struct[0];
    assert(RRM_Policy_Member_List->ran_param_id == RRM_Policy_Member_List_8_4_3_6 && "wrong RRM_Policy_Member_List id");
    assert(RRM_Policy_Member_List->ran_param_val.type == LIST_RAN_PARAMETER_VAL_TYPE && "wrong RRM_Policy_Member_List type");
    assert(RRM_Policy_Member_List->ran_param_val.lst != NULL && "NULL RRM_Policy_Member_List->ran_param_val.lst");
    assert(RRM_Policy_Member_List->ran_param_val.lst->sz_lst_ran_param == 1 && "wrong RRM_Policy_Member_List->ran_param_val.lst->sz_lst_ran_param");
    assert(RRM_Policy_Member_List->ran_param_val.lst->lst_ran_param != NULL && "NULL RRM_Policy_Member_List->ran_param_val.lst->lst_ran_param");

    lst_ran_param_t* RRM_Policy_Member = &RRM_Policy_Member_List->ran_param_val.lst->lst_ran_param[0];
    //Bug in rc_enc_asn.c:1003, asn didn't define ran_param_id for lst_ran_param_t ...
    //assert(RRM_Policy_Member->ran_param_id == RRM_Policy_Member_8_4_3_6 && "wrong RRM_Policy_Member id");
    assert(RRM_Policy_Member->ran_param_struct.sz_ran_param_struct == 2 && "wrong RRM_Policy_Member->ran_param_struct.sz_ran_param_struct");
    assert(RRM_Policy_Member->ran_param_struct.ran_param_struct != NULL && "NULL RRM_Policy_Member->ran_param_struct.ran_param_struct");

    seq_ran_param_t* PLMN_Identity = &RRM_Policy_Member->ran_param_struct.ran_param_struct[0];
    assert(PLMN_Identity->ran_param_id == PLMN_Identity_8_4_3_6 && "wrong PLMN_Identity id");
    assert(PLMN_Identity->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong PLMN_Identity type");
    assert(PLMN_Identity->ran_param_val.flag_false != NULL && "NULL PLMN_Identity->ran_param_val.flag_false");
    assert(PLMN_Identity->ran_param_val.flag_false->type == OCTET_STRING_RAN_PARAMETER_VALUE && "wrong PLMN_Identity->ran_param_val.flag_false->type");
    ///// GET RC PLMN ////
    mccmnc_t const RC_plmn = convert_plmn_to_mccmnc(PLMN_Identity->ran_param_val.flag_false->octet_str_ran.buf);
    LOG_D(NR_MAC, "RC PLMN: MCC %u, MNC %u\n", RC_plmn.mcc, RC_plmn.mnc);

    seq_ran_param_t* S_NSSAI = &RRM_Policy_Member->ran_param_struct.ran_param_struct[1];
    assert(S_NSSAI->ran_param_id == S_NSSAI_8_4_3_6 && "wrong S_NSSAI id");
    assert(S_NSSAI->ran_param_val.type == STRUCTURE_RAN_PARAMETER_VAL_TYPE && "wrong S_NSSAI type");
    assert(S_NSSAI->ran_param_val.strct != NULL && "NULL S_NSSAI->ran_param_val.strct");
    assert(S_NSSAI->ran_param_val.strct->sz_ran_param_struct == 2 && "wrong S_NSSAI->ran_param_val.strct->sz_ran_param_struct");
    assert(S_NSSAI->ran_param_val.strct->ran_param_struct != NULL && "NULL S_NSSAI->ran_param_val.strct->ran_param_struct");

    seq_ran_param_t* SST = &S_NSSAI->ran_param_val.strct->ran_param_struct[0];
    assert(SST->ran_param_id == SST_8_4_3_6 && "wrong SST id");
    assert(SST->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong SST type");
    assert(SST->ran_param_val.flag_false != NULL && "NULL SST->ran_param_val.flag_false");
    assert(SST->ran_param_val.flag_false->type == OCTET_STRING_RAN_PARAMETER_VALUE && "wrong SST->ran_param_val.flag_false type");
    seq_ran_param_t* SD = &S_NSSAI->ran_param_val.strct->ran_param_struct[1];
    assert(SD->ran_param_id == SD_8_4_3_6 && "wrong SD id");
    assert(SD->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong SD type");
    assert(SD->ran_param_val.flag_false != NULL && "NULL SD->ran_param_val.flag_false");
    assert(SD->ran_param_val.flag_false->type == OCTET_STRING_RAN_PARAMETER_VALUE && "wrong SD->ran_param_val.flag_false type");

    ///// GET RC NSSAI ////
    nssai_t RC_nssai = {0};
    char* rc_sst_str = cp_ba_to_str(SST->ran_param_val.flag_false->octet_str_ran);
    RC_nssai.sst = atoi(rc_sst_str);
    char* rc_sd_str = NULL;
    if (SD->ran_param_val.flag_false->octet_str_ran.len > 0) {
      rc_sd_str = cp_ba_to_str(SD->ran_param_val.flag_false->octet_str_ran);
      RC_nssai.sd = atoi(rc_sd_str);
    } else {
      RC_nssai.sd = 0xffffff;
    }
    LOG_D(NR_MAC, "RC (oct) SST %s, SD %s -> (uint) SST %d, SD %d\n", rc_sst_str, rc_sd_str, RC_nssai.sst, RC_nssai.sd);


    ///// SLICE LABEL NAME /////
    char sst_str[] = "SST";
    char sd_str[] = "SD";
    size_t label_nssai_len = strlen(sst_str) + strlen(rc_sst_str) + strlen(sd_str) + (rc_sd_str != NULL ? strlen(rc_sd_str) : 0) + 1;
    char* label_nssai = malloc(label_nssai_len);
    assert(label_nssai != NULL && "Memory exhausted");
    if (rc_sd_str != NULL)
      sprintf(label_nssai, "%s%s%s%s", sst_str, rc_sst_str, sd_str, rc_sd_str);
    else
      sprintf(label_nssai, "%s%s%s%s", sst_str, rc_sst_str, sd_str, "null");
    free(rc_sst_str);
    free(rc_sd_str);

    ///// SLICE-LEVEL PRB QUOTA /////
    seq_ran_param_t* Min_PRB_Policy_Ratio = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[1];
    assert(Min_PRB_Policy_Ratio->ran_param_id == Min_PRB_Policy_Ratio_8_4_3_6 && "wrong Min_PRB_Policy_Ratio id");
    assert(Min_PRB_Policy_Ratio->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong Min_PRB_Policy_Ratio type");
    assert(Min_PRB_Policy_Ratio->ran_param_val.flag_false != NULL && "NULL Min_PRB_Policy_Ratio->ran_param_val.flag_false");
    assert(Min_PRB_Policy_Ratio->ran_param_val.flag_false->type == INTEGER_RAN_PARAMETER_VALUE && "wrong Min_PRB_Policy_Ratio->ran_param_val.flag_false type");
    int64_t min_prb_ratio = Min_PRB_Policy_Ratio->ran_param_val.flag_false->int_ran;
    LOG_D(NR_MAC, "configure slice %ld, label %s, Min_PRB_Policy_Ratio %ld\n", i, label_nssai, min_prb_ratio);

    seq_ran_param_t* Max_PRB_Policy_Ratio = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[2];
    assert(Max_PRB_Policy_Ratio->ran_param_id == Max_PRB_Policy_Ratio_8_4_3_6 && "wrong Max_PRB_Policy_Ratio id");
    assert(Max_PRB_Policy_Ratio->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong Max_PRB_Policy_Ratio type");
    assert(Max_PRB_Policy_Ratio->ran_param_val.flag_false != NULL && "NULL Max_PRB_Policy_Ratio->ran_param_val.flag_false");
    assert(Max_PRB_Policy_Ratio->ran_param_val.flag_false->type == INTEGER_RAN_PARAMETER_VALUE && "wrong Max_PRB_Policy_Ratio->ran_param_val.flag_false type");
    int64_t max_prb_ratio = Max_PRB_Policy_Ratio->ran_param_val.flag_false->int_ran;
    LOG_D(NR_MAC, "configure slice %ld, label %s, Max_PRB_Policy_Ratio %ld\n", i, label_nssai, max_prb_ratio);

    seq_ran_param_t* Dedicated_PRB_Policy_Ratio = &RRM_Policy_Ratio_Group->ran_param_struct.ran_param_struct[3];
    assert(Dedicated_PRB_Policy_Ratio->ran_param_id == Dedicated_PRB_Policy_Ratio_8_4_3_6 && "wrong Dedicated_PRB_Policy_Ratio id");
    assert(Dedicated_PRB_Policy_Ratio->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE && "wrong Dedicated_PRB_Policy_Ratio type");
    assert(Dedicated_PRB_Policy_Ratio->ran_param_val.flag_false != NULL && "NULL Dedicated_PRB_Policy_Ratio->ran_param_val.flag_false");
    assert(Dedicated_PRB_Policy_Ratio->ran_param_val.flag_false->type == INTEGER_RAN_PARAMETER_VALUE && "wrong Dedicated_PRB_Policy_Ratio->ran_param_val.flag_false type");
    int64_t dedicated_prb_ratio = Dedicated_PRB_Policy_Ratio->ran_param_val.flag_false->int_ran;
    LOG_D(NR_MAC, "configure slice %ld, label %s, Dedicated_PRB_Policy_Ratio %ld\n", i, label_nssai, dedicated_prb_ratio);

    void *params = NULL;
    if (new_algo == NVS_SLICING) {
      params = malloc(sizeof(nvs_nr_slice_param_t));
      ((nvs_nr_slice_param_t *)params)->type = NVS_RES;
      ((nvs_nr_slice_param_t *)params)->pct_reserved = dedicated_prb_ratio/100.0;
      //require to convert max/min/dedicated_prb_ratio to Mbps_reserved/reference for NVS_RATE
      //((nvs_nr_slice_param_t *)params)->type = NVS_RATE;
      //((nvs_nr_slice_param_t *)params)->Mbps_reserved = //todo
      //((nvs_nr_slice_param_t *)params)->Mbps_reference = //todo
    } else {
      LOG_E(NR_MAC, "unknown slice algo\n");
      return false;
    }

    ///// ADD DL SLICE /////
    nr_pp_impl_param_dl_t *dl = &RC.nrmac[mod_id]->pre_processor_dl;
    int8_t s_id = get_slice_idx_nssai(dl->slices, RC_nssai);
    NR_SCHED_LOCK(&nrmac->sched_lock);
    int rc = add_mod_dl_slice(mod_id, current_algo, s_id, RC_nssai, label_nssai, params);
    NR_SCHED_UNLOCK(&nrmac->sched_lock);
    if (rc < 0) {
      LOG_E(NR_MAC, "error code %d while updating DL slices\n", rc);
      return false;
    }

    /// ASSOC DL SLICE ///
    if (nrmac->pre_processor_dl.algorithm <= 0)
      LOG_W(NR_MAC, "current DL/UL slice algo is NONE, no UE can be associated\n");

    if (nrmac->UE_info.connected_ue_list[0] == NULL)
      LOG_W(NR_MAC, "no UE connected, no UE can be associated\n");


    NR_UEs_t *UE_info = &RC.nrmac[mod_id]->UE_info;
    UE_iterator(UE_info->connected_ue_list, UE) {
      rnti_t rnti = UE->rnti;
      NR_UE_sched_ctrl_t *sched_ctrl = &UE->UE_sched_ctrl;
      bool assoc_ue = false;
      for (size_t l = 0; l < seq_arr_size(&sched_ctrl->lc_config); ++l) {
        const nr_lc_config_t *c = seq_arr_at(&sched_ctrl->lc_config, l);
        long lcid = c->lcid;
        LOG_D(NR_MAC, "l %lu, lcid %ld, sst %d, sd %d\n", l, lcid, c->nssai.sst, c->nssai.sd);
        if (nssai_matches(c->nssai, RC_nssai.sst, &RC_nssai.sd)) {

          mccmnc_t const UE_plmn = get_ue_mcc_mnc(rnti);
          nssai_t const UE_nssai = c->nssai;
          LOG_D(NR_MAC, "UE: mcc %u mnc %u, sst %d sd %d, RC: mcc %u mnc %u, sst %d sd %d\n",
                UE_plmn.mcc, UE_plmn.mnc, UE_nssai.sst, UE_nssai.sd, RC_plmn.mcc, RC_plmn.mnc, RC_nssai.sst, RC_nssai.sd);

          if (UE_plmn.mcc == RC_plmn.mcc && UE_plmn.mnc == RC_plmn.mnc && UE_nssai.sst == RC_nssai.sst && UE_nssai.sd == RC_nssai.sd) {
            // for one UE has two slices
            NR_SCHED_LOCK(&nrmac->sched_lock);
            dl->add_UE(dl->slices, UE);
            NR_SCHED_UNLOCK(&nrmac->sched_lock);
            assoc_ue = true;
            // for one UE has one slice
            // assoc_ue = assoc_ue_to_slices(dl, s_id, UE);
          } else {
            LOG_E(NR_MAC, "Failed adding UE (PLMN: mcc %u mnc %u, NSSAI: sst %d sd %d) to slice (PLMN: mcc %u mnc %u, NSSAI: sst %d sd %d)\n",
                  UE_plmn.mcc, UE_plmn.mnc, UE_nssai.sst, UE_nssai.sd, RC_plmn.mcc, RC_plmn.mnc, RC_nssai.sst, RC_nssai.sd);
          }
        }
      }
      if (!assoc_ue) {
        LOG_D(NR_MAC, "Failed matching UE rnti %x with current slice (sst %d, sd %d), might lost user plane data\n", rnti, RC_nssai.sst, RC_nssai.sd);
      }
    }

  }

  LOG_D(NR_MAC, "All slices add/mod successfully!\n");
  return true;
}



/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "common/utils/var_array.h"
#include "executables/nr-uesoftmodem.h"
#include "executables/softmodem-common.h"
#include "LAYER2/NR_MAC_UE/mac_defs.h"
#include "LAYER2/NR_MAC_UE/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/mac_rrc_dl_handler.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_gNB/nr_radio_config.h"
#include "NR_BCCH-BCH-Message.h"
#include "NR_BWP-Downlink.h"
#include "NR_CellGroupConfig.h"
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_COMMON/nr_mac_common.h"
#include "NR_PHY_INTERFACE/NR_IF_Module.h"
#include "NR_ReconfigurationWithSync.h"
#include "NR_ServingCellConfig.h"
#include "NR_SetupRelease.h"
#include "NR_UE_PHY_INTERFACE/NR_IF_Module.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/MODULATION/modulation_common.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_RU.h"
#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/defs_nr_common.h"
#include "PHY/impl_defs_nr.h"
#include "PHY/phy_vars_nr_ue.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR_UE/defs.h"
#include "SCHED_NR_UE/fapi_nr_ue_l1.h"
#include "T.h"
#include "asn_internal.h"
#include "assertions.h"
#include "common/config/config_load_configmodule.h"
#include "common/ngran_types.h"
#include "common/ran_context.h"
#include "common/utils/T/T.h"
#include "common/utils/nr/nr_common.h"
#include "common/utils/var_array.h"
#include "common_lib.h"
#include "e1ap_messages_types.h"
#include "fapi_nr_ue_interface.h"
#include "nfapi_interface.h"
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"
#include "nr_ue_phy_meas.h"
#include "oai_asn1.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "thread-pool.h"
#include "time_meas.h"
#include "utils.h"
#define inMicroS(a) (((double)(a))/(get_cpu_freq_GHz()*1000.0))
#include "SIMULATION/LTE_PHY/common_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];
double cpuf;
char *uecap_file;

//uint8_t nfapi_mode = 0;
uint64_t downlink_frequency[MAX_NUM_CCs][4];
THREAD_STRUCT thread_struct;
nfapi_ue_release_request_body_t release_rntis;
//Fixme: Uniq dirty DU instance, by global var, datamodel need better management
instance_t DUuniqInstance=0;
instance_t CUuniqInstance=0;

// needed for some functions
openair0_config_t openair0_cfg[MAX_CARDS];
configmodule_interface_t *uniqCfg = NULL;

int dummy_nr_ue_ul_indication(nr_uplink_indication_t *ul_info) { return(0);  }

void e1_bearer_context_setup(const e1ap_bearer_setup_req_t *req) { abort(); }
void e1_bearer_context_modif(const e1ap_bearer_mod_req_t *req) { abort(); }
void e1_bearer_release_cmd(const e1ap_bearer_release_cmd_t *cmd) { abort(); }

int8_t nr_rrc_RA_succeeded(const module_id_t mod_id, const uint8_t gNB_index) {
  return 0;
}



void processSlotTX(void *arg) {}

// needed for some functions
openair0_config_t openair0_cfg[MAX_CARDS];
void update_ptrs_config(NR_CellGroupConfig_t *secondaryCellGroup, uint16_t *rbSize, uint8_t *mcsIndex,int8_t *ptrs_arg);
void update_dmrs_config(NR_CellGroupConfig_t *scg, int8_t* dmrs_arg);
extern void fix_scd(NR_ServingCellConfig_t *scd);// forward declaration



/* specific dlsim DL preprocessor: uses rbStart/rbSize/mcs/nrOfLayers from command line of dlsim */
int g_mcsIndex = -1, g_mcsTableIdx = 0, g_rbStart = -1, g_rbSize = -1, g_nrOfLayers = 1, g_pmi = 0;
void nr_dlsim_preprocessor(gNB_MAC_INST *nr_mac, post_process_pdsch_t *pp_pdsch)
{
  NR_UE_info_t *UE_info = nr_mac->UE_info.connected_ue_list[0];
  AssertFatal(nr_mac->UE_info.connected_ue_list[1] == NULL, "Only single UE allowed in dlsim\n");
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl;
  NR_UE_DL_BWP_t *current_BWP = &UE_info->current_DL_BWP;
  NR_ServingCellConfigCommon_t *scc = nr_mac->common_channels[0].ServingCellConfigCommon;

  int nr_of_candidates = 0;
  if (g_mcsIndex < 4) {
    find_aggregation_candidates(&sched_ctrl->aggregation_level, &nr_of_candidates, sched_ctrl->search_space, 8);
  }
  if (nr_of_candidates == 0) {
    find_aggregation_candidates(&sched_ctrl->aggregation_level, &nr_of_candidates, sched_ctrl->search_space, 4);
  }
  uint32_t Y = get_Y(sched_ctrl->search_space, pp_pdsch->slot, UE_info->rnti);
  int CCEIndex = find_pdcch_candidate(nr_mac,
                                      /* CC_id = */ 0,
                                      sched_ctrl->aggregation_level,
                                      nr_of_candidates,
                                      0,
                                      &sched_ctrl->sched_pdcch,
                                      sched_ctrl->coreset,
                                      Y);
  AssertFatal(CCEIndex >= 0,
              "%4d.%2d could not find CCE for DL DCI UE %d/RNTI %04x\n",
              pp_pdsch->frame,
              pp_pdsch->slot,
              0,
              UE_info->rnti);
  sched_ctrl->cce_index = CCEIndex;

  NR_sched_pdsch_t sched_pdsch = {
      .rbStart = g_rbStart,
      .rbSize = g_rbSize,
      .bwp_info = get_pdsch_bwp_start_size(nr_mac, UE_info),
      .mcs = g_mcsIndex,
      .nrOfLayers = g_nrOfLayers,
      .pm_index = g_pmi,
  };
  /* the following might override the table that is mandated by RRC
   * configuration */
  current_BWP->mcsTableIdx = g_mcsTableIdx;
  sched_pdsch.time_domain_allocation = get_dl_tda(nr_mac, pp_pdsch->slot);
  AssertFatal(sched_pdsch.time_domain_allocation >= 0,"Unable to find PDSCH time domain allocation in list\n");

  sched_pdsch.tda_info = get_dl_tda_info(current_BWP,
                                          sched_ctrl->search_space->searchSpaceType->present,
                                          sched_pdsch.time_domain_allocation,
                                          NR_MIB__dmrs_TypeA_Position_pos2,
                                          1,
                                          TYPE_C_RNTI_,
                                          sched_ctrl->coreset->controlResourceSetId,
                                          false);

  sched_pdsch.dmrs_parms = get_dl_dmrs_params(scc,
                                               current_BWP,
                                               &sched_pdsch.tda_info,
                                               sched_pdsch.nrOfLayers);

  sched_pdsch.Qm = nr_get_Qm_dl(sched_pdsch.mcs, current_BWP->mcsTableIdx);
  sched_pdsch.R = nr_get_code_rate_dl(sched_pdsch.mcs, current_BWP->mcsTableIdx);
  sched_pdsch.tb_size = nr_compute_tbs(sched_pdsch.Qm,
                                        sched_pdsch.R,
                                        sched_pdsch.rbSize,
                                        sched_pdsch.tda_info.nrOfSymbols,
                                        sched_pdsch.dmrs_parms.N_PRB_DMRS * sched_pdsch.dmrs_parms.N_DMRS_SLOT,
                                        0 /* N_PRB_oh, 0 for initialBWP */,
                                        0 /* tb_scaling */,
                                        sched_pdsch.nrOfLayers) >> 3;

  /* the simulator assumes the HARQ PID is equal to the slot number */
  sched_pdsch.dl_harq_pid = pp_pdsch->slot;

  /* The scheduler uses lists to track whether a HARQ process is
   * free/busy/awaiting retransmission, and updates the HARQ process states.
   * However, in the simulation, we never get ack or nack for any HARQ process,
   * thus the list and HARQ states don't match what the scheduler expects.
   * Therefore, below lines just "repair" everything so that the scheduler
   * won't remark that there is no HARQ feedback */
  sched_ctrl->feedback_dl_harq.head = -1; // always overwrite feedback HARQ process
  if (sched_ctrl->harq_processes[pp_pdsch->slot].round == 0) // depending on round set in simulation ...
    add_front_nr_list(&sched_ctrl->available_dl_harq, pp_pdsch->slot); // ... make PID available
  else
    add_front_nr_list(&sched_ctrl->retrans_dl_harq, pp_pdsch->slot);   // ... make PID retransmission
  sched_ctrl->harq_processes[pp_pdsch->slot].is_waiting = false;
  AssertFatal(sched_pdsch.rbStart >= 0, "invalid rbStart %d\n", sched_pdsch.rbStart);
  AssertFatal(sched_pdsch.rbSize > 0, "invalid rbSize %d\n", sched_pdsch.rbSize);
  AssertFatal(sched_pdsch.mcs >= 0, "invalid mcs %d\n", sched_pdsch.mcs);
  AssertFatal(current_BWP->mcsTableIdx >= 0 && current_BWP->mcsTableIdx <= 2, "invalid mcsTableIdx %d\n", current_BWP->mcsTableIdx);

  post_process_dlsch(nr_mac, pp_pdsch, UE_info, &sched_pdsch);
}

nrUE_params_t nrUE_params;

nrUE_params_t *get_nrUE_params(void) {
  return &nrUE_params;
}

void validate_input_pmi(nfapi_nr_config_request_scf_t *gNB_config,
                        nr_pdsch_AntennaPorts_t pdsch_AntennaPorts,
                        int nrOfLayers,
                        int pmi)
{
  if (pmi == 0)
    return;

  nfapi_nr_pm_pdu_t *pmi_pdu = &gNB_config->pmi_list.pmi_pdu[pmi - 1]; // pmi 0 is identity matrix
  AssertFatal(pmi == pmi_pdu->pm_idx, "PMI %d doesn't match to the one in precoding matrix %d\n", pmi, pmi_pdu->pm_idx);
  AssertFatal(nrOfLayers == pmi_pdu->numLayers, "Number of layers %d doesn't match to the one in precoding matrix %d for PMI %d\n",
              nrOfLayers, pmi_pdu->numLayers, pmi);
  int num_antenna_ports = pdsch_AntennaPorts.N1 * pdsch_AntennaPorts.N2 * pdsch_AntennaPorts.XP;
  AssertFatal(num_antenna_ports == pmi_pdu->num_ant_ports, "Configured antenna ports %d does not match precoding matrix AP size %d for PMI %d\n",
              num_antenna_ports, pmi_pdu->num_ant_ports, pmi);
}


float **s_interleaved, **r_re, **r_im;
uint8_t n_tx=1,n_rx=1;
channel_desc_t *gNB2UE;
NR_Sched_Rsp_t *Sched_INFO;
c16_t **txdata;

int oai_lib_init() {

      stop = false;
  __attribute__((unused)) struct sigaction oldaction;
  sigaction(SIGINT, &sigint_action, &oldaction);

  FILE *csv_file = NULL;
  char *filename_csv = NULL;
  setbuf(stdout, NULL);
  int c;
  int i,aa;//,l;
  double SNR, snr0 = -2.0, snr1 = 2.0;
  uint8_t snr1set=0;
  double effRate;
  //float psnr;
  double eff_tp_check = 0.7;
  uint32_t TBS = 0;
  //double iqim = 0.0;
  //unsigned char pbch_pdu[6];
  //  int sync_pos, sync_pos_slot;
  //  FILE *rx_frame_file;
  FILE *output_fd = NULL;
  //uint8_t write_output_file=0;
  //int result;
  //int freq_offset;
  //  int subframe_offset;
  //  char fname[40], vname[40];
  int trial, n_trials = 1, n_false_positive = 0;
  //int n_errors2, n_alamouti;
  uint8_t round;
  uint8_t num_rounds = 4;
  char gNBthreads[128]="n";

  //uint32_t nsymb,tx_lev,tx_lev1 = 0,tx_lev2 = 0;
  //uint8_t extended_prefix_flag=0;
  //int8_t interf1=-21,interf2=-21;

  FILE *input_fd=NULL,*pbch_file_fd=NULL;
  //char input_val_str[50],input_val_str2[50];

  //uint8_t frame_mod4,num_pdcch_symbols = 0;

  SCM_t channel_model = AWGN; // AWGN Rayleigh1 Rayleigh1_anticorr;
  double DS_TDL = .03;
  int delay = 0;

  //double pbch_sinr;
  //int pbch_tx_ant;
  int N_RB_DL=106,mu=1;

  //unsigned char frame_type = 0;

  int frame=1,slot=1;
  int frame_length_complex_samples;
  //int frame_length_complex_samples_no_prefix;
  NR_DL_FRAME_PARMS *frame_parms;
  UE_nr_rxtx_proc_t UE_proc;
  channel_desc_t *gNB2UE;
  gNB_MAC_INST *gNB_mac;
  NR_UE_MAC_INST_t *UE_mac;
  int cyclic_prefix_type = NFAPI_CP_NORMAL;
  int loglvl=OAILOG_INFO;

  //float target_error_rate = 0.01;
  cpuf = get_cpu_freq_GHz();
  int8_t enable_ptrs = 0;
  int8_t modify_dmrs = 0;

  int8_t dmrs_arg[3] = {-1,-1,-1};// Invalid values
  /* L_PTRS = ptrs_arg[0], K_PTRS = ptrs_arg[1] */
  int8_t ptrs_arg[2] = {-1,-1};// Invalid values

  uint16_t ptrsRePerSymb = 0;
  uint16_t pdu_bit_map = 0x0;
  uint16_t dlPtrsSymPos = 0;
  uint16_t ptrsSymbPerSlot = 0;
  uint16_t rbSize = 106;
  uint8_t  mcsIndex = 9;
  uint8_t  dlsch_threads = 0;
  int chest_type[2] = {0};
  uint8_t  max_ldpc_iterations = 5;

  int print_perf = 0;

  int use_cuda = 0;

  void *h_tx_sig_pinned = NULL;

#ifdef ENABLE_CUDA
  void *d_tx_sig = NULL, *d_intermediate_sig = NULL, *d_final_output = NULL;
  void *d_curand_states = NULL;
  void *h_final_output_pinned = NULL;
  void *d_channel_coeffs_gpu = NULL;
#endif
    
  logInit();
  set_glog(loglvl);
  /* initialize the sin table */
  InitSinLUT();
#ifdef ENABLE_CUDA
  init_cuda_chsim_buffers(use_cuda,
                          n_tx,
                          n_rx,
                          &d_tx_sig,
                          &d_intermediate_sig,
                          &d_final_output,
                          &d_curand_states,
                          &h_tx_sig_pinned,
                          &h_final_output_pinned,
                          &d_channel_coeffs_gpu);
#endif

#if !defined(ENABLE_CUDA) || !use_cuda
  printf("Pre-allocating padded host memory for the CPU channel pipeline...\n");
  int num_samples_alloc = 153600;
  const int max_padding_alloc = 256 - 1;
  size_t padded_tx_alloc_bytes = n_tx * (num_samples_alloc + max_padding_alloc) * 2 * sizeof(float);
  h_tx_sig_pinned = malloc(padded_tx_alloc_bytes);
  if (h_tx_sig_pinned == NULL) {
    printf("Error: Failed to allocate host buffer for CPU path\n");
    exit(-1);
  }
#endif

  get_softmodem_params()->phy_test = 1;
  get_softmodem_params()->do_ra = 0;
  IS_SOFTMODEM_DLSIM = true;
  RC.gNB = (PHY_VARS_gNB**) malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = (PHY_VARS_gNB*) malloc(sizeof(PHY_VARS_gNB ));
  memset(RC.gNB[0],0,sizeof(PHY_VARS_gNB));

  gNB = RC.gNB[0];
  gNB->ofdm_offset_divisor = UINT_MAX;
  gNB->phase_comp = true; // we need to perform phase compensation, otherwise everything will fail
  frame_parms = &gNB->frame_parms; //to be initialized I suppose (maybe not necessary for PBCH)
  frame_parms->nb_antennas_tx = n_tx;
  frame_parms->nb_antennas_rx = n_rx;
  frame_parms->N_RB_DL = N_RB_DL;
  frame_parms->N_RB_UL = N_RB_DL;

  AssertFatal((gNB->if_inst = NR_IF_Module_init(0)) != NULL, "Cannot register interface");
  gNB->if_inst->NR_PHY_config_req = nr_phy_config_request;

  NR_ServingCellConfigCommon_t *scc = calloc(1,sizeof(*scc));;
  prepare_scc(scc);
  uint64_t ssb_bitmap = 1; // Enable only first SSB with index ssb_indx=0
  fill_scc_sim(scc, &ssb_bitmap, N_RB_DL, N_RB_DL, mu, mu);
  fix_scc(scc, ssb_bitmap);

  frame_structure_t frame_structure = {0};
  frame_type_t frame_type = TDD;
  config_frame_structure(mu,
                         scc->tdd_UL_DL_ConfigurationCommon,
                         get_tdd_period_idx(scc->tdd_UL_DL_ConfigurationCommon),
                         frame_type,
                         &frame_structure);
  AssertFatal(is_dl_slot(slot, &frame_structure), "The slot selected is not DL. Can't run DLSIM\n");

  // TODO do a UECAP for phy-sim
  nr_pdsch_AntennaPorts_t pdsch_AntennaPorts = {0};
  pdsch_AntennaPorts.N1 = n_tx > 1 ? n_tx >> 1 : 1;
  pdsch_AntennaPorts.N2 = 1;
  pdsch_AntennaPorts.XP = n_tx > 1 ? 2 : 1;
  const nr_mac_config_t conf = {.pdsch_AntennaPorts = pdsch_AntennaPorts,
                                .pusch_AntennaPorts = n_tx,
                                .minRXTXTIME = 6,
                                .do_CSIRS = 0,
                                .do_SRS = 0,
                                .num_dlharq = 16,
                                .num_ulharq = 16,
                                .maxMIMO_layers = g_nrOfLayers,
                                .force_256qam_off = false,
                                .timer_config.sr_ProhibitTimer = 0,
                                .timer_config.sr_TransMax = 64,
                                .timer_config.sr_ProhibitTimer_v1700 = 0,
                                .timer_config.t300 = 400,
                                .timer_config.t301 = 400,
                                .timer_config.t310 = 2000,
                                .timer_config.n310 = 10,
                                .timer_config.t311 = 3000,
                                .timer_config.n311 = 1,
                                .timer_config.t319 = 400,
                                .num_agg_level_candidates = {0, 0, 1, 1, 0}};
  const nr_rlc_configuration_t rlc_config = {
    .srb = {
      .t_poll_retransmit = 45,
      .t_reassembly = 35,
      .t_status_prohibit = 0,
      .poll_pdu = -1,
      .poll_byte = -1,
      .max_retx_threshold = 8,
      .sn_field_length = 12,
    },
    .drb_am = {
      .t_poll_retransmit = 45,
      .t_reassembly = 15,
      .t_status_prohibit = 15,
      .poll_pdu = 64,
      .poll_byte = 1024 * 500,
      .max_retx_threshold = 32,
      .sn_field_length = 18,
    },
    .drb_um = {
      .t_reassembly = 15,
      .sn_field_length = 12,
    }
  };

  RC.nb_nr_macrlc_inst = 1;
  RC.nb_nr_mac_CC = (int*)malloc(RC.nb_nr_macrlc_inst*sizeof(int));
  for (i = 0; i < RC.nb_nr_macrlc_inst; i++)
    RC.nb_nr_mac_CC[i] = 1;
  mac_top_init_gNB(ngran_gNB, scc, &conf, &rlc_config);
  gNB_mac = RC.nrmac[0];
  nr_mac_config_scc(RC.nrmac[0], scc, &conf);

  gNB_mac->dl_bler.harq_round_max = num_rounds;

  gNB->ap_N1 = pdsch_AntennaPorts.N1;
  gNB->ap_N2 = pdsch_AntennaPorts.N2;
  gNB->ap_XP = pdsch_AntennaPorts.XP;

  validate_input_pmi(&gNB_mac->config[0], pdsch_AntennaPorts, g_nrOfLayers, g_pmi);

  NR_UE_NR_Capability_t *UE_Capability_nr = CALLOC(1,sizeof(NR_UE_NR_Capability_t));
  prepare_sim_uecap(UE_Capability_nr, scc, mu, N_RB_DL, g_mcsTableIdx, 0);
  rnti_t rnti = 0x1234;
  int uid = 0;
  int ssb_index = 0;
  NR_CellGroupConfig_t *secondaryCellGroup = get_default_secondaryCellGroup(scc, UE_Capability_nr, 0, 1, &conf, uid, ssb_index);
  secondaryCellGroup->spCellConfig->reconfigurationWithSync = get_reconfiguration_with_sync(rnti, uid, scc, frame);

  /* -U option modify DMRS */
  if(modify_dmrs) {
    update_dmrs_config(secondaryCellGroup, dmrs_arg);
  }
  /* -T option enable PTRS */
  if(enable_ptrs) {
    update_ptrs_config(secondaryCellGroup, &rbSize, &mcsIndex, ptrs_arg);
  }


  //xer_fprint(stdout, &asn_DEF_NR_CellGroupConfig, (const void*)secondaryCellGroup);

  // UE dedicated configuration
  nr_mac_add_test_ue(RC.nrmac[0], rnti, secondaryCellGroup);
  // reset preprocessor to the one of DLSIM after it has been set during
  // nr_mac_config_scc()
  gNB_mac->pre_processor_dl = nr_dlsim_preprocessor;
  phy_init_nr_gNB(gNB);
  N_RB_DL = gNB->frame_parms.N_RB_DL;
  NR_UE_info_t *UE_info = RC.nrmac[0]->UE_info.connected_ue_list[0];

  configure_UE_BWP(RC.nrmac[0], scc, UE_info, false, NR_SearchSpace__searchSpaceType_PR_ue_Specific, -1, -1);

  // stub to configure frame_parms
  //  nr_phy_config_request_sim(gNB,N_RB_DL,N_RB_DL,mu,Nid_cell,SSB_positions);
  // call MAC to configure common parameters

  /* nr_mac_add_test_ue() has created one user, so set the scheduling
   * parameters from command line in global variables that will be picked up by
   * scheduling preprocessor */
  if (g_mcsIndex < 0) g_mcsIndex = 9;
  if (g_rbStart < 0) g_rbStart=0;
  if (g_rbSize < 0) g_rbSize = N_RB_DL - g_rbStart;

  double fs,txbw,rxbw;
  get_samplerate_and_bw(mu,
                        N_RB_DL,
                        frame_parms->threequarter_fs,
                        &fs,
                        &txbw,
                        &rxbw);

  gNB2UE = new_channel_desc_scm(n_tx,
                                n_rx,
                                channel_model,
                                fs/1e6,//sampling frequency in MHz
                                0,
                                txbw,
                                DS_TDL,
                                0.0,
                                CORR_LEVEL_LOW,
                                0,
                                delay,
                                0,
                                0);

#ifdef ENABLE_CUDA
  float *h_channel_coeffs = NULL;
  if (use_cuda) {
    int num_links = n_tx * n_rx;
    h_channel_coeffs = (float *)malloc(num_links * gNB2UE->channel_length * sizeof(float) * 2);
  }
#endif
  if (gNB2UE==NULL) {
    printf("Problem generating channel model. Exiting.\n");
    exit(-1);
  }

  frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  //frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  int slot_offset = get_samples_slot_timestamp(frame_parms, slot);
  int slot_length = slot_offset - get_samples_slot_timestamp(frame_parms, slot - 1);

  s_interleaved = malloc(n_tx * sizeof(float *));
  r_re = malloc(n_rx * sizeof(float *));
  r_im = malloc(n_rx * sizeof(float *));
  txdata = malloc(n_tx * sizeof(int *));

  for (i = 0; i < n_tx; i++) {
    s_interleaved[i] = malloc(slot_length * 2 * sizeof(float));
    printf("Allocating %d samples for txdata\n", frame_length_complex_samples);
    txdata[i] = calloc(1, frame_length_complex_samples * sizeof(int));
  }

  for (i = 0; i < n_rx; i++) {
    r_re[i] = calloc(1, slot_length * sizeof(float));
    r_im[i] = calloc(1, slot_length * sizeof(float));
  }




  //configure UE
  UE = malloc(sizeof(PHY_VARS_NR_UE));
  memset((void*)UE,0,sizeof(PHY_VARS_NR_UE));
  PHY_vars_UE_g = malloc(sizeof(PHY_VARS_NR_UE**));
  PHY_vars_UE_g[0] = malloc(sizeof(PHY_VARS_NR_UE*));
  PHY_vars_UE_g[0][0] = UE;
  memcpy(&UE->frame_parms,frame_parms,sizeof(NR_DL_FRAME_PARMS));
  UE->frame_parms.nb_antennas_rx = n_rx;
  UE->frame_parms.nb_antenna_ports_gNB = n_tx;
  UE->nrLDPC_coding_interface = gNB->nrLDPC_coding_interface;
  UE->max_ldpc_iterations = max_ldpc_iterations;
  init_nr_ue_phy_cpu_stats(&UE->phy_cpu_stats);
  UE->is_synchronized = 1;

  if (init_nr_ue_signal(UE, 1) != 0)
  {
    printf("Error at UE NR initialisation\n");
    exit(-1);
  }

  init_nr_ue_transport(UE);

  UE_mac = nr_l2_init_ue(0, mu);
  ue_init_config_request(UE_mac, get_slots_per_frame_from_scs(mu));

  UE->if_inst = nr_ue_if_module_init(0);
  UE->if_inst->scheduled_response = nr_ue_scheduled_response;
  UE->if_inst->phy_config_request = nr_ue_phy_config_request;
  UE->if_inst->dl_indication = nr_ue_dl_indication;
  UE->if_inst->ul_indication = dummy_nr_ue_ul_indication;
  UE->chest_freq = chest_type[0];
  UE->chest_time = chest_type[1];

  UE_mac->if_module = nr_ue_if_module_init(0);


  initFloatingCoresTpool(dlsch_threads, &nrUE_params.Tpool, false, "UE-tpool");

  // generate signal
  AssertFatal(input_fd==NULL,"Not ready for input signal file\n");

  // clone CellGroup to have a separate copy at UE
  NR_CellGroupConfig_t *UE_CellGroup = clone_CellGroupConfig(secondaryCellGroup);

  //Configure UE
  NR_BCCH_BCH_Message_t *mib = get_new_MIB_NR(scc);
  nr_rrc_mac_config_req_mib(0, 0, mib->message.choice.mib, false);
  nr_rrc_mac_config_req_cg(0, 0, 0, 0, UE_CellGroup, UE_Capability_nr);

  asn1cFreeStruc(asn_DEF_NR_CellGroupConfig, UE_CellGroup);

  UE_mac->state = UE_CONNECTED;
  UE_mac->ra.ra_state = nrRA_SUCCEEDED;

  nr_phy_data_t phy_data = {0};
  fapi_nr_dl_config_request_t dl_config = {.sfn = frame, .slot = slot};
  nr_scheduled_response_t scheduled_response = {.dl_config = &dl_config, .phy_data = &phy_data, .mac = UE_mac};

  nr_ue_phy_config_request(&UE_mac->phy_config);
  //NR_COMMON_channels_t *cc = RC.nrmac[0]->common_channels;
  int ret = 1;
  initNamedTpool(gNBthreads, &gNB->threadPool, true, "gNB-tpool");
  initNotifiedFIFO(&gNB->L1_tx_out);

  // Buffers to store internal memory of slot process
  int rx_size = (((14 * UE->frame_parms.N_RB_DL * 12 * sizeof(int32_t)) + 15) >> 4) << 4;
  UE->phy_sim_rxdataF = calloc(sizeof(int32_t *) * UE->frame_parms.nb_antennas_rx * g_nrOfLayers,
                               UE->frame_parms.samples_per_slot_wCP * sizeof(int32_t));
  UE->phy_sim_pdsch_llr = calloc(1, (8 * (3 * 8 * 8448)) * sizeof(int16_t)); // Max length
  UE->phy_sim_pdsch_rxdataF_ext = calloc(sizeof(int32_t *) * UE->frame_parms.nb_antennas_rx * g_nrOfLayers, rx_size);
  UE->phy_sim_pdsch_rxdataF_comp = calloc(sizeof(int32_t *) * UE->frame_parms.nb_antennas_rx * g_nrOfLayers, rx_size);
  UE->phy_sim_pdsch_dl_ch_estimates = calloc(sizeof(int32_t *) * UE->frame_parms.nb_antennas_rx * g_nrOfLayers, rx_size);
  UE->phy_sim_pdsch_dl_ch_estimates_ext = calloc(sizeof(int32_t *) * UE->frame_parms.nb_antennas_rx * g_nrOfLayers, rx_size);
  int a_segments = MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER*NR_MAX_NB_LAYERS;  //number of segments to be allocated
  if (g_rbSize != 273) {
    a_segments = a_segments*g_rbSize;
    a_segments = (a_segments/273)+1;
  }
  uint32_t dlsch_bytes = a_segments*1056;  // allocated bytes per segment
  UE->phy_sim_dlsch_b = calloc(1, dlsch_bytes);

  // csv file
  if (filename_csv != NULL) {
    csv_file = fopen(filename_csv, "a");
    if (csv_file == NULL) {
      printf("Can't open file \"%s\", errno %d\n", filename_csv, errno);
      free(s_interleaved);
      free(r_re);
      free(r_im);
      free(txdata);
      return 1;
    }
    // adding name of parameters into file
    fprintf(csv_file,"SNR,false_positive,");
    for (int r = 0; r < num_rounds; r++)
      fprintf(csv_file,"n_errors_%d,errors_scrambling_%d,channel_bler_%d,channel_ber_%d,",r,r,r,r);
    fprintf(csv_file,"avg_round,eff_rate,eff_throughput,TBS\n");
  }
  //---------------

  Sched_INFO = memalign(32, sizeof(*Sched_INFO));
  if (Sched_INFO == NULL) {
    LOG_E(PHY, "out of memory\n");
    exit(1);
  }

    return 0;
}

void oai_lib_shutdown(void) {

      free(Sched_INFO);

  free_channel_desc_scm(gNB2UE);

  for (int i = 0; i < n_tx; i++) {
    free(s_interleaved[i]);
    free(txdata[i]);
  }
  for (int i = 0; i < n_rx; i++) {
    free(r_re[i]);
    free(r_im[i]);
  }

#ifdef ENABLE_CUDA
  free_cuda_chsim_buffers(use_cuda,
                          &d_tx_sig,
                          &d_intermediate_sig,
                          &d_final_output,
                          &d_curand_states,
                          &h_tx_sig_pinned,
                          &h_final_output_pinned,
                          &h_channel_coeffs,
                          &d_channel_coeffs_gpu);
#endif

  free(s_interleaved);
  free(r_re);
  free(r_im);
  free(txdata);
  free(UE->phy_sim_rxdataF);
  free(UE->phy_sim_pdsch_llr);
  free(UE->phy_sim_pdsch_rxdataF_ext);
  free(UE->phy_sim_pdsch_rxdataF_comp);
  free(UE->phy_sim_pdsch_dl_ch_estimates);
  free(UE->phy_sim_pdsch_dl_ch_estimates_ext);
  free(UE->phy_sim_dlsch_b);

  free_nrLDPC_coding_interface(&gNB->nrLDPC_coding_interface);

}


int oai_lib_nr_polar_decoder(int16_t *x, 
                             uint64_t *out, 
                             uint8_t ones_flag, 
                             int8_t messageType, 
                             uint16_t messageLength, 
                             uint8_t aggregation_level) {
    
   polar_decoder_int16(x, out, ones_flag, messageType, messageLength,aggregation_level);

   return 0;
}

int oai_lib_nr_polar_encoder(uint64_t *A,
                            void **out,
                            int32_t crcmask,
                            uint8_t ones_flag,
                            int8_t messageType,
                            uint16_t messageLength,
                            uint8_t aggregation_level) {

// A is the input which is at most 64 bits
// out is the rate-matched output, encoded as a bit-packed 32-bit element array. The expected length in bits depends on the messageType:
// messageType = 0 => PBCH => output length (bits) = 864 bits (27 32-bit words)
// messageType = 1 => DCI => output length (bits) = 108 * aggregation_level
// messageType = 2 => PUCCH => output length (bits) = 16 * aggregation_level
// messageType = 3 => PSBCH =? output length (bits) = 1792 bits (56 32-bit words))
// crc is automatically computed based on the messageLength and messageType and crcmask is applied to the resultant CRC if required. Set to 0 if not required
// ones_flag is used to append 24 1's at the beginning of a DCI message prior to computing the CRC parity
// aggregation_level is 1,2,4,8,16
    int encodedLength=0;
    switch(messageType) {
        case 0: //PBCH
        default:
            encodedLength = 864;
            break;
        case 1: //DCI
            encodedLength = (108*aggregation_level);
            break;
        case 2: //UCI/PUCCH
            encodedLength = (16*aggregation_level);
            break;
        case 3: //PSBCH
            encodedLength = 1792;
    }
    *out = malloc(1+(encodedLength>>3));        
    polar_encoder_fast(A,*out,crcmask,ones_flag,messageType,messageLength,aggregation_level);
    return(encodedLength);
}

const char *oai_lib_last_error(void) {
  

}

#ifdef __cplusplus
}
#endif
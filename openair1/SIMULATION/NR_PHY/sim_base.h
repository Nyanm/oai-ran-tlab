/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

// gNB MAC
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
// UE MAC
#include "openair2/LAYER2/NR_MAC_UE/mac_defs.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
// gNB PHY
#include "openair1/PHY/defs_gNB.h"
#include "e1ap_messages_types.h"
#include "common/config/config_load_configmodule.h"
// UE PHY
#include "openair1/PHY/defs_nr_UE.h"
#include "openair1/PHY/phy_vars_nr_ue.h"

#include "nr_unitary_defs.h"

#define BASE_OPTSTR_LEN 100

// Necessary globals
RAN_CONTEXT_t RC;
configmodule_interface_t *uniqCfg;
char *uecap_file;
THREAD_STRUCT thread_struct;
int64_t uplink_frequency_offset[MAX_NUM_CCs][4];
uint64_t downlink_frequency[MAX_NUM_CCs][4];

void e1_bearer_context_setup(const e1ap_bearer_setup_req_t *req)
{
  abort();
}
void e1_bearer_context_modif(const e1ap_bearer_mod_req_t *req)
{
  abort();
}
void e1_bearer_release_cmd(const e1ap_bearer_release_cmd_t *cmd)
{
  abort();
}

typedef struct PhySim PhySim;

struct PhySim {
  RAN_CONTEXT_t *rc;
  gNB_MAC_INST *gnb_mac;
  NR_UE_MAC_INST_t *ue_mac;
  PHY_VARS_gNB gnb_phy;
  PHY_VARS_NR_UE ue_phy;
  int frame;
  int slot;
  int num_trials;
  int mu;
  int N_RB_BW;
  int num_rb;
  float snr_min, snr_max, snr_step;
  int n_tx, n_rx;
  NR_ServingCellConfigCommon_t *scc;
  int loglvl;
  char optstr[BASE_OPTSTR_LEN];
  int verbose;
  long n_bits_total;
  long n_errors_total;
};

static void physim_print_usage(void)
{
  printf(
      "Physim base options:\n"
      "\t-n <int>   Number of trials                       [default: 0]\n"
      "\t-m <int>   Numerology                             [default: 1]\n"
      "\t-s <float> Start SNR                              [default: -2.0]\n"
      "\t-y <int>   Number of transmit antennas            [default: 1]\n"
      "\t-z <int>   Number of receive antennas             [default: 1]\n"
      "\t-R <int>   Number of PRBs configured for the cell [default: 106]\n"
      "\t-S <float> Stop SNR                               [default: 2.0]\n");
}

static void physim_parse_common_args(PhySim *self, int argc, char **argv)
{
  memset(self, 0, sizeof(*self));

  // Defaults
  self->frame = 0;
  self->slot = 0;
  self->num_trials = 1;
  self->N_RB_BW = 106;
  self->num_rb = 106;
  self->snr_min = -2.0;
  self->snr_max = -1.0;
  self->snr_step = 1.0;
  self->mu = 1;
  self->loglvl = OAILOG_WARNING;

  const char opts[] = "n:m:s:y:z:L:R:S:h";
  DevAssert(sizeof(opts) < BASE_OPTSTR_LEN);
  strncpy(self->optstr, opts, sizeof(self->optstr));

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_CSIRSSIM] Error, configuration module init failed\n");
  }

  opterr = 0; // Supress error from sim specific arguments which are parsed later
  optind = 1;
  int c = 0;

  while ((c = getopt(argc, argv, self->optstr)) != -1) {
    switch (c) {
      case 'n':
        self->num_trials = atoi(optarg);
        break;

      case 'm':
        self->mu = atoi(optarg);
        break;

      case 'y':
        self->n_tx = atoi(optarg);
        break;

      case 'z':
        self->n_rx = atoi(optarg);
        break;

      case 'L':
        self->loglvl = atoi(optarg);
        break;

      case 'R':
        self->N_RB_BW = atoi(optarg);
        break;

      case 's':
        self->snr_min = atof(optarg);
        break;

      case 'S':
        self->snr_max = atof(optarg);
        break;
    }
  }
  opterr = 1;
  optind = 1;
}

static void physim_init(PhySim *self)
{
  logInit();
  set_glog(self->loglvl);
  /* initialize the sin table */
  InitSinLUT();

  // gNB MAC
  RC.nrmac = malloc(sizeof(*RC.nrmac));
  RC.nrmac[0] = malloc(sizeof(**RC.nrmac));
  self->gnb_mac = RC.nrmac[0];
  self->scc = calloc(1, sizeof(*self->scc));
  prepare_scc(self->scc);
  uint64_t ssb_bitmap = 1; // Enable only first SSB with index ssb_indx=0
  fill_scc_sim(self->scc, &ssb_bitmap, self->N_RB_BW, self->N_RB_BW, self->mu, self->mu);
  fix_scc(self->scc, ssb_bitmap);
  self->gnb_mac->common_channels[0].ServingCellConfigCommon = self->scc;
  self->gnb_mac->common_channels[0].mib = get_new_MIB_NR(self->scc);

  // UE MAC
  self->ue_mac = nr_l2_init_ue(0, self->mu);
}

static void physim_cleanup(PhySim *self)
{
  ASN_STRUCT_FREE(asn_DEF_NR_BCCH_BCH_Message, self->gnb_mac->common_channels[0].mib);
  ASN_STRUCT_FREE(asn_DEF_NR_ServingCellConfigCommon, self->scc);
  free_and_zero(RC.nrmac[0]);
  free_and_zero(RC.nrmac);
}
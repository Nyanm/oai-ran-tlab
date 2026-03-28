/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_UNITARY_DEFS__H__
#define __NR_UNITARY_DEFS__H__

#include "NR_ServingCellConfigCommon.h"
#include "NR_ServingCellConfig.h"

// Define signal handler to attempt graceful termination
extern bool stop;

void sigint_handler(int arg);

const struct sigaction sigint_action = {.sa_handler = sigint_handler,
                                        // restore handler to default upon entry to the signal handler
                                        .sa_flags = SA_RESETHAND};

int oai_exit = 0;

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  const char *msg = s == NULL ? "no comment" : s;
  printf("Exiting at: %s:%d %s(), %s\n", file, line, function, msg);
  exit(-1);
}

signed char quantize(double D, double x, unsigned char B)
{
  double qxd;
  short maxlev;
  qxd = floor(x / D);
  maxlev = 1 << (B - 1); //(char)(pow(2,B-1));

  if (qxd <= -maxlev)
    qxd = -maxlev;
  else if (qxd >= maxlev)
    qxd = maxlev - 1;

  return ((char)qxd);
}

int oai_nfapi_rach_ind(nfapi_rach_indication_t *rach_ind)
{
  return (0);
}
// NR_IF_Module_t *NR_IF_Module_init(int Mod_id){return(NULL);}
int oai_nfapi_ul_config_req(nfapi_ul_config_request_t *ul_config_req)
{
  return (0);
}

void stop_nr_nfapi_vnf()
{
  abort();
}

void stop_nr_nfapi_pnf()
{
  abort();
}

void fill_scc_sim(NR_ServingCellConfigCommon_t *scc, uint64_t *ssb_bitmap, int N_RB_DL, int N_RB_UL, int mu_dl, int mu_ul);
void fix_scc(NR_ServingCellConfigCommon_t *scc, uint64_t ssbmap);
void prepare_scc(NR_ServingCellConfigCommon_t *scc);
void prepare_msgA_scc(NR_ServingCellConfigCommon_t *scc);
void prepare_scd(NR_ServingCellConfig_t *scd);
uint32_t ngap_generate_gNB_id(void)
{
  return 0;
}

#endif

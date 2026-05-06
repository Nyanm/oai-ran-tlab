#ifndef CSI_RB_LOGGING_H
#define CSI_RB_LOGGING_H

#include "PHY_VARS_NR_UE.h"
#include "nfapi_nr_interface_scf.h"
#include "platform_types.h"

typedef void (*csi_rb_logging_callback_t)(
    const PHY_VARS_NR_UE *ue,
    const UE_nr_rxtx_proc_t *proc,
    const c16_t csi_rs_estimated_channel_freq[][],
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu);

extern csi_rb_logging_callback_t csi_rb_logging_callback;
extern int csi_rb_logging_enabled;

#endif

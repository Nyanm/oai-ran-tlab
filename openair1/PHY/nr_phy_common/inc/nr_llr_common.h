/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_LLR_COMMON__H__
#define __NR_LLR_COMMON__H__

#include "PHY/impl_defs_top.h"
#include "PHY/TOOLS/tools_defs.h"

void nr_qpsk_llr_2layers(c16_t *stream0_in,
                         c16_t *stream1_in,
                         int16_t *stream0_out,
                         c16_t *rho01,
                         uint32_t length);

void nr_16qam_llr_2layers(c16_t *stream0_in,
                          c16_t *stream1_in,
                          c16_t *ch_mag,
                          c16_t *ch_mag_i,
                          int16_t *stream0_out,
                          c16_t *rho01,
                          uint32_t length);

void nr_64qam_llr_2layers(c16_t *stream0_in,
                          c16_t *stream1_in,
                          c16_t *ch_mag,
                          c16_t *ch_mag_i,
                          int16_t *stream0_out,
                          c16_t *rho01,
                          uint32_t length);

/** \brief This function computes the log-likelihood ratios for QPSK, 16, 64, and 256 QAM
    @param rxdataF_comp Compensated channel output
    @param ch_mag  channel magnitude multiplied by the 1st amplitude threshold
    @param ch_magb channel magnitude multiplied by the 2nd amplitude threshold
    @param ch_magc channel magnitude multiplied by the 3rd amplitude threshold (256QAM)
    @param ulsch_llr llr output
    @param nb_re number of resource elements
    @param mod_order modulation order
*/
void nr_compute_llr(const c16_t *rxdataF_comp,
                    const c16_t *ch_mag,
                    const c16_t *ch_magb,
                    const c16_t *ch_magc,
                    int16_t *ulsch_llr,
                    uint32_t nb_re,
                    uint8_t mod_order);

void nr_compute_ML_llr(uint32_t rxdataF_ext_offset,
                       c16_t *rxdataF_comp0,
                       c16_t *rxdataF_comp1,
                       c16_t *ch_mag0,
                       c16_t *ch_mag1,
                       int16_t *llr_layers0,
                       int16_t *llr_layers1,
                       c16_t *rho0,
                       c16_t *rho1,
                       uint32_t nb_re,
                       uint8_t mod_order);

#endif /* __NR_LLR_COMMON__H__ */

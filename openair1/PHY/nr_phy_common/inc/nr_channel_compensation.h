/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __NR_CHANNEL_COMPENSATION__H__
#define __NR_CHANNEL_COMPENSATION__H__

#include "PHY/impl_defs_top.h"
#include "PHY/TOOLS/tools_defs.h"

void nr_channel_compensation(uint32_t buffer_length,
                             int nb_rx_ant,
                             c16_t rxFext[][buffer_length],
                             c16_t chFext[][nb_rx_ant][buffer_length],
                             c16_t rxF_ch_maga[][buffer_length],
                             c16_t rxF_ch_magb[][buffer_length],
                             c16_t rxF_ch_magc[][buffer_length],
                             c16_t **rxComp,
                             int nb_layers,
                             c16_t rho[][nb_layers][buffer_length],
                             int numLoopCnt,
                             int mod_order,
                             uint32_t bufOffset,
                             uint32_t output_shift);

uint8_t nr_mmse_2layers(const c16_t **rxdataF_comp,
                        uint32_t buffer_length,
                        int nb_rx_ant,
                        c16_t ul_ch_mag[][buffer_length],
                        c16_t ul_ch_magb[][buffer_length],
                        c16_t ul_ch_magc[][buffer_length],
                        c16_t ul_ch_estimates_ext[][nb_rx_ant][buffer_length],
                        unsigned short nb_rb,
                        unsigned char mod_order,
                        int shift,
                        uint32_t bufOffset,
                        unsigned char symbol,
                        int length,
                        uint32_t noise_var);

#endif /* __NR_CHANNEL_COMPENSATION__H__ */

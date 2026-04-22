/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Top-level defines and structure definitions
 */
#ifndef __DEFS_NR_DMA_H__
#define __DEFS_NR_DMA_H__

#include <stdint.h>

#define DMA_SYNC_MAGIC 0xDA7A0A10

typedef struct __attribute__((packed)) {
    uint32_t magic;

    // --- 1. config->ssb_table ---
    uint32_t config__ssb_table__ssb_mask_list_0__ssb_mask__value;
    uint32_t config__ssb_table__ssb_mask_list_1__ssb_mask__value;
    uint32_t config__ssb_table__case_v3__value;

    // --- 2. config->ssb_config ---
    uint8_t  config__ssb_config__scs_common__value;

    // --- 3. config->analog_beamforming_ve ---
    uint16_t config__analog_beamforming_ve__num_beams_period_vendor_ext__tl__tag;
    uint16_t config__analog_beamforming_ve__num_beams_period_vendor_ext__tl__length;
    uint8_t  config__analog_beamforming_ve__num_beams_period_vendor_ext__value;

    uint16_t config__analog_beamforming_ve__analog_bf_vendor_ext__tl__tag;
    uint16_t config__analog_beamforming_ve__analog_bf_vendor_ext__tl__length;
    uint8_t  config__analog_beamforming_ve__analog_bf_vendor_ext__value;

    // --- 4. config->prach_config ---
    uint16_t config__prach_config__prach_ConfigurationIndex__tl__tag;
    uint8_t  config__prach_config__prach_ConfigurationIndex__value;
    uint8_t  config__prach_config__num_prach_fd_occasions__value;
    uint16_t config__prach_config__num_prach_fd_occasions_list_0__k1__value;
    uint8_t  config__prach_config__prach_sequence_length__value;

    // --- 5. config->tdd_table ---
    uint16_t config__tdd_table__tdd_period__tl__tag;
    uint8_t  config__tdd_table__tdd_period__value;
    
    // 【Big Boss】: 160 slots * 14 symbols = 2240 bytes
    // 用于复原那个噩梦般的多级指针嵌套值
    uint8_t  config__tdd_table__max_tdd_periodicity_list__max_num_of_symbol_per_slot_list__slot_config__value[160 * 14];

    // --- 6. config->cell_config ---
    uint8_t  config__cell_config__frame_duplex_type__tl__tag;
    uint8_t  config__cell_config__frame_duplex_type__value;

    // --- 7. config->carrier_config ---
    // 对 dl_k0 和 ul_k0 进行完整映射
    uint16_t config__carrier_config__dl_k0__value[5];
    uint16_t config__carrier_config__ul_k0__value[5];

    uint16_t config__carrier_config__dl_grid_size__value;
    uint16_t config__carrier_config__ul_grid_size__value;
    uint16_t config__carrier_config__num_rx_ant__value;
    uint16_t config__carrier_config__num_tx_ant__value;

    uint32_t config__carrier_config__dl_frequency__value;
    uint32_t config__carrier_config__dl_bandwidth__value;

    uint32_t config__carrier_config__ul_frequency__value;
    uint32_t config__carrier_config__ul_bandwidth__value;

    uint8_t  ofdm_offset_divisor;

    // --- 9. DMA 内存锚点 ---//目前还没分配
    uint64_t dma_remote_addr;
    uint32_t dma_rkey;

} dma_sync_config_t;

#endif
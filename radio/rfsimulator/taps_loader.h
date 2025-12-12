#ifndef RADIO_RFSIMULATOR_TAPS_LOADER_H
#define RADIO_RFSIMULATOR_TAPS_LOADER_H

#include "openair1/SIMULATION/TOOLS/sim.h"

/**
 * @brief Loads external CIR snapshots from a binary float32 dataset and selects top-N taps per snapshot.
 *
 * The binary file must contain, for each snapshot:
 *   [ Re_0, Re_1, ..., Re_{len-1}, Im_0, Im_1, ..., Im_{len-1} ]
 * where len = number of taps per snapshot, all values are float32.
 * Snapshots are stored consecutively.
 *
 * For each snapshot:
 *   - Computes tap energy (E = Re^2 + Im^2)
 *   - Selects top `desc->max_loaded_taps` by energy
 *   - Restores tap order for causality
 *   - Stores to desc->external_cir_data (calloc'd)
 *
 * On success, sets:
 *   - desc->external_cir_count
 *   - desc->external_cir_len (= max_loaded_taps)
 *   - desc->external_cir_idx (reset)
 *   - desc->external_cir_data (allocated)
 *   - desc->external_cir_file (strdup)
 *
 * @param[in]  bin_path   Path to binary file (float32, real/imag interleaved per snapshot)
 * @param[out] desc       Pointer to channel descriptor to populate
 * @return     0 on success, negative on error (file error, bad length, alloc failure)
 */
int load_external_taps_binary(const char *bin_path, channel_desc_t *desc);

#endif // RADIO_RFSIMULATOR_TAPS_LOADER_H

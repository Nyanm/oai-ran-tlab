#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <common/utils/LOG/log.h>
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "radio/rfsimulator/taps_loader.h"

#define TAPSLOADER_LOG_MAXSNAP 5
// To debug tap selection, enable this macro to print selected taps per snapshot
// #define TAPSLOADER_LOG_TAPS

/**
 * @brief Loads external CIR snapshots from a binary float32 dataset and selects top-N taps per snapshot.
 * See taps_loader.h for binary format and parameter description.
 *
 * @param[in]  bin_path   Path to binary file (float32, real/imag interleaved per snapshot)
 * @param[out] desc       Pointer to channel descriptor to populate
 * @return     0 on success, negative error code on failure
 */
static int cmp_by_energy(const void *a, const void *b, void *energies_v)
{
  const double *E = energies_v;
  int ia = *(const int *)a, ib = *(const int *)b;
  double eA = E[ia], eB = E[ib];
  return (eA < eB) - (eA > eB);
}

static int cmp_int_asc(const void *a, const void *b)
{
  return (*(int *)a - *(int *)b);
}

int load_external_taps_binary(const char *bin_path, channel_desc_t *desc)
{
  if (!desc) {
    LOG_E(HW, "Error: desc pointer is NULL\n");
    return -9;
  }

  // TEMP: For debugging, override bin_path with known absolute path.
  // TODO: Remove this hardcode after filesystem/path issues are resolved.
  // Original bin_path: %s
  bin_path = "../../../cir_datasets/urban_macro/SionnaRT_CIR.bin";

  FILE *f = fopen(bin_path, "rb");
  if (!f) {
    LOG_E(HW, "Error -1: Cannot open CIR file: %s (errno=%d, %s)\n", bin_path, errno, strerror(errno));
    return -1;
  }

  if (fseek(f, 0, SEEK_END) != 0 || ftell(f) < 0) {
    LOG_E(HW, "Error -2: Failed to determine file size\n");
    fclose(f);
    return -2;
  }

  long bytes = ftell(f);
  rewind(f);

  uint8_t external_cir_len = desc->external_cir_len;
  uint32_t max_loaded_taps = desc->max_loaded_taps;
  if (external_cir_len == 0) {
    LOG_E(HW, "Error -3: external_cir_len is zero\n");
    fclose(f);
    return -3;
  }

  if (max_loaded_taps == 0 || max_loaded_taps > external_cir_len)
    max_loaded_taps = external_cir_len;

  long total_float = bytes / sizeof(float);
  if (total_float % (2 * external_cir_len)) {
    LOG_E(HW, "Error -4: File size is not a multiple of 2 * external_cir_len (external_cir_len = %u)\n", external_cir_len);
    fclose(f);
    return -4;
  }

  uint32_t count = total_float / (2 * external_cir_len);
  LOG_I(HW, "CIR binary '%s' contains %u snapshots of %u taps\n", bin_path, count, external_cir_len);

  float *real_buf = calloc_or_fail(count * external_cir_len, sizeof(float));
  float *imag_buf = calloc_or_fail(count * external_cir_len, sizeof(float));

  if (fread(real_buf, sizeof(float), count * external_cir_len, f) != count * external_cir_len
      || fread(imag_buf, sizeof(float), count * external_cir_len, f) != count * external_cir_len) {
    LOG_E(HW, "Error -6: Failed to read CIR data\n");
    free(real_buf);
    free(imag_buf);
    fclose(f);
    return -6;
  }

  fclose(f);

  /* calloc_or_fail will ASSERT_FATAL on failure */
  struct complexd *tap_data = calloc_or_fail(count * max_loaded_taps, sizeof(*tap_data));

  /* use VLAs on the stack for sorting — no frees needed */
  double tap_energies[external_cir_len];
  int tap_indices[external_cir_len];
  int top_indices[max_loaded_taps];

  for (uint32_t s = 0; s < count; s++) {
    for (uint8_t t = 0; t < external_cir_len; t++) {
      double r = real_buf[s * external_cir_len + t];
      double i = imag_buf[s * external_cir_len + t];
      tap_energies[t] = r * r + i * i;
      tap_indices[t] = t;
    }
    /* sort by energy ↓, pick top N, then restore time order ↑ */
    qsort_r(tap_indices, external_cir_len, sizeof(int), cmp_by_energy, tap_energies);
    memcpy(top_indices, tap_indices, sizeof(int) * max_loaded_taps);
    qsort(top_indices, max_loaded_taps, sizeof(int), cmp_int_asc);

    for (uint8_t m = 0; m < max_loaded_taps; m++) {
      int t = top_indices[m];
      tap_data[s * max_loaded_taps + m].r = real_buf[s * external_cir_len + t];
      tap_data[s * max_loaded_taps + m].i = imag_buf[s * external_cir_len + t];
    }

    /* --------------- TAP LOGGING ---------------
     * By default, all CIR tap logs are disabled for performance and clarity.
     * Developers can enable logging for debugging by uncommenting the macro above.
     * Example log will output the selected taps and energies for each snapshot.
     */
#ifdef TAPSLOADER_LOG_TAPS
    if (s < TAPSLOADER_LOG_MAXSNAP) {
      LOG_I(HW, "\n================= [TAPS] Snapshot #%u: Selected Top-%u CIR Taps =================", s, max_loaded_taps);
      for (uint8_t m = 0; m < max_loaded_taps; m++) {
        int t = top_idx[m];
        double r = real_buf[s * external_cir_len + t];
        double i = imag_buf[s * external_cir_len + t];
        double e = r * r + i * i;
        LOG_I(HW, "[TAPS]   Tap[%3d] = ( %+1.6f + %+1.6fj ) | Energy = %.3e", t, r, i, e);
      }
      LOG_I(HW, "[TAPS]   --> Stored taps (Causal Order):");
      for (int m = 0; m < max_loaded_taps; m++)
        LOG_I(HW,
              "[TAPS]       [%2d] ( %+1.6f + %+1.6fj )",
              m,
              tap_data[s * max_loaded_taps + m].r,
              tap_data[s * max_loaded_taps + m].i);
      LOG_I(HW, "=================================================================================\n");
    }
#endif
  }

  free(real_buf);
  free(imag_buf);

  desc->external_cir_count = count;
  desc->external_cir_idx = 0;
  desc->external_cir_data = tap_data;

  if (desc->external_cir_file)
    free(desc->external_cir_file);
  desc->external_cir_file = strdup(bin_path);

  LOG_I(HW, "Successfully loaded %u snapshots, %u taps each (top %u retained)\n", count, external_cir_len, max_loaded_taps);
  return 0;
}

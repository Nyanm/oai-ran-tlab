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
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>

#include "common/utils/LOG/log.h"
#include "common/utils/assertions.h"
#include "SIMULATION/TOOLS/sim.h"

#include "cirdb_provider.h"
#include "cirdb_yaml.h"

#define NUM_TAPS_BUFFERS 4
#define MAX_L_PUBLISH    8

typedef struct {
  void *taps_blob;          /* contiguous complexf for all links */
  channel_desc_t *ch;       /* ch_ps points into taps_blob */
} cirdb_buffer_t;

typedef struct {
  /* I/O */
  FILE *fh;

  /* Selected entry meta */
  int model_id;
  int n_tx;
  int n_rx;
  int L_full;
  int S;
  double fs_hz;
  double snapshot_dt_s;
  float ds_ns;
  float speed_mps;
  uint64_t sel_offset;
  uint64_t sel_nbytes;

  /* Publication control */
  uint32_t L_out;
  uint32_t snap_idx;

  /* Shape in publication */
  int num_tx;
  int num_rx;

  /* Double buffering for readers */
  cirdb_buffer_t bufs[NUM_TAPS_BUFFERS];
  int cur;

  /* Temporary read buffer for one snapshot of full L_full taps */
  void  *snapshot_tmp;
  size_t snapshot_tmp_bytes;

  /* Published pointer location owned by caller */
  channel_desc_t **channel_desc_out;

  /* Path resolution helper */
  char path_override[PATH_MAX];

  /* Bookkeeping of last step computed from elapsed time */
  int64_t last_step_applied;  /* -1 until first update */
} cirdb_g;

static cirdb_g G;

/* returns 1 if file exists and is readable */
static int file_readable(const char *p) { return p && p[0] && access(p, R_OK) == 0; }

static int get_exe_dir(char out[PATH_MAX]) {
  char exe[PATH_MAX] = {0};
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n <= 0) return 0;
  exe[n] = 0;
  char tmp[PATH_MAX];
  if (snprintf(tmp, sizeof(tmp), "%s", exe) >= (int)sizeof(tmp)) return 0;
  char *dir = dirname(tmp);
  if (!dir || !dir[0]) return 0;
  if (snprintf(out, PATH_MAX, "%s", dir) >= (int)PATH_MAX) return 0;
  return 1;
}

static int path_join(char out[PATH_MAX], const char *dir, const char *leaf) {
  if (!dir || !leaf) return 0;
  size_t ld = strnlen(dir, PATH_MAX);
  size_t ll = strnlen(leaf, PATH_MAX);
  if (ld + 1 + ll + 1 > PATH_MAX) return 0;
  int n = snprintf(out, PATH_MAX, "%s/%s", dir, leaf);
  return n > 0 && n < (int)PATH_MAX;
}

/* Optional public override helper kept for backward compatibility */
void cirdb_set_path_override(const char *path) {
  if (!path) { G.path_override[0] = '\0'; return; }
  size_t n = strnlen(path, PATH_MAX - 1);
  memcpy(G.path_override, path, n);
  G.path_override[n] = '\0';
}

/* Resolve cir_db.bin path */
static const char *resolve_db_path(char out[PATH_MAX]) {
  if (file_readable(G.path_override)) {
    snprintf(out, PATH_MAX, "%s", G.path_override);
    return out;
  }
  char exe_dir[PATH_MAX];
  if (get_exe_dir(exe_dir)) {
    if (path_join(out, exe_dir, "cir_db.bin") && file_readable(out)) return out;
    if (path_join(out, exe_dir, "radio/vrtsim/cir_db.bin") && file_readable(out)) return out;
  }
  snprintf(out, PATH_MAX, "%s", "cir_db.bin");
  if (file_readable(out)) return out;
  snprintf(out, PATH_MAX, "%s", "radio/vrtsim/cir_db.bin");
  if (file_readable(out)) return out;
  snprintf(out, PATH_MAX, "%s", "cir_db.bin");
  return out;
}

static inline void fread_exact(void *dst, size_t sz, FILE *fh) {
  size_t r = fread(dst, 1, sz, fh);
  if (r != sz) {
    LOG_E(HW, "CIRDB short read: expected %zu got %zu errno=%d\n", sz, r, errno);
    abort();
  }
}

static inline size_t cf_bytes(size_t n) { return n * sizeof(struct complexf); }

/* Point channel_desc ch_ps into a compacted taps buffer */
static void point_channel_desc(channel_desc_t *ch,
                               struct complexf *base,
                               int num_tx, int num_rx, int L_out) {
  for (int aarx = 0; aarx < num_rx; aarx++) {
    for (int aatx = 0; aatx < num_tx; aatx++) {
      int link = aarx + num_rx * aatx;
      ch->ch_ps[link] = &base[link * L_out];
    }
  }
  ch->channel_length = L_out;
  ch->path_loss_dB = 0;
  ch->nb_tx = num_tx;
  ch->nb_rx = num_rx;
}

/* Copy first L_out taps per link from a full L_full snapshot */
static void compact_L_out(struct complexf *dst,
                          const struct complexf *src_full,
                          int num_tx, int num_rx,
                          int L_full, int L_out) {
  for (int aatx = 0; aatx < num_tx; aatx++) {
    for (int aarx = 0; aarx < num_rx; aarx++) {
      int link = aarx + num_rx * aatx;
      const struct complexf *src = &src_full[(size_t)link * (size_t)L_full];
      struct complexf *dst_link = &dst[(size_t)link * (size_t)L_out];
      memcpy(dst_link, src, cf_bytes(L_out));
    }
  }
}

/* Load snapshot index s into the next publication buffer and flip */
static void load_snapshot_and_publish(uint32_t s)
{
  uint64_t stride = (uint64_t)G.n_tx * (uint64_t)G.n_rx *
                    (uint64_t)G.L_full * sizeof(struct complexf);
  uint64_t off = G.sel_offset + (uint64_t)s * stride;

  if (fseeko(G.fh, (off_t)off, SEEK_SET) != 0) {
    LOG_E(HW, "CIRDB fseoko failed errno=%d\n", errno);
    abort();
  }
  fread_exact(G.snapshot_tmp, (size_t)stride, G.fh);

  int next = (G.cur + 1) % NUM_TAPS_BUFFERS;
  cirdb_buffer_t *buf = &G.bufs[next];

  compact_L_out((struct complexf *)buf->taps_blob,
                (const struct complexf *)G.snapshot_tmp,
                G.n_tx, G.n_rx, G.L_full, (int)G.L_out);

  point_channel_desc(buf->ch,
                     (struct complexf *)buf->taps_blob,
                     G.n_tx, G.n_rx, (int)G.L_out);

  *G.channel_desc_out = buf->ch;
  G.cur = next;
}

void cirdb_connect(int id,
                   int num_tx_antennas,
                   int num_rx_antennas,
                   const cirdb_select_opts_t *sel,
                   channel_desc_t **channel_desc_out)
{
  (void)id;
  memset(&G, 0, sizeof(G));
  G.channel_desc_out  = channel_desc_out;
  G.num_tx            = num_tx_antennas;
  G.num_rx            = num_rx_antennas;
  G.cur               = 0;
  G.last_step_applied = -1;

  /* Resolve binary path with priority: sel->bin_path, optional override, exe-dir fallbacks, CWD fallbacks */
  char binpath[PATH_MAX];
  const char *use_bin = NULL;
  if (sel && sel->bin_path && sel->bin_path[0]) {
    use_bin = sel->bin_path;
  } else {
    use_bin = resolve_db_path(binpath);
  }
  G.fh = fopen(use_bin, "rb");
  AssertFatal(G.fh != NULL, "open %s failed, errno=%d", use_bin, errno);

  /* Resolve YAML sidecar path with priority: sel->yaml_path, then derive from bin */
  char sidecar[PATH_MAX];
  if (sel && sel->yaml_path && sel->yaml_path[0]) {
    snprintf(sidecar, PATH_MAX, "%s", sel->yaml_path);
  } else {
    size_t n = strnlen(use_bin, PATH_MAX - 6);
    snprintf(sidecar, PATH_MAX, "%s", use_bin);
    if (n >= 4 && strcmp(&use_bin[n-4], ".bin") == 0) {
      sidecar[n] = '\0';
      snprintf(sidecar + n, PATH_MAX - n, "%s", ".yaml");
    } else {
      snprintf(sidecar, PATH_MAX, "%s.yaml", use_bin);
    }
  }

  /* Selection request */
  int   want_model_id = (sel && sel->want_model_id  > 0) ? sel->want_model_id  : -1;
  float want_ds       = (sel && sel->want_ds_ns     > 0) ? sel->want_ds_ns     : -1.0f;
  float want_speed    = (sel && sel->want_speed_mps > 0) ? sel->want_speed_mps : -1.0f;

  cirdb_entry_meta_t m = (cirdb_entry_meta_t){0};
  cirdb_select_req_t req = {
    .want_model_id    = want_model_id,
    .want_tx          = G.num_tx,
    .want_rx          = G.num_rx,
    .want_ds_ns       = want_ds,
    .want_speed_mps   = want_speed,
    .allow_shape_swap = 1,
    .w_ds             = 1.0f,
    .w_speed          = 0.2f,
    .yaml_path        = sidecar
  };

  int ok = cirdb_yaml_select(&req, &m);

  if (ok <= 0) {
    cirdb_select_req_t req_relaxed = req;
    req_relaxed.want_tx = 0;
    req_relaxed.want_rx = 0;
    ok = cirdb_yaml_select(&req_relaxed, &m);
  }

  AssertFatal(ok > 0, "No suitable CIR entry found in YAML %s", sidecar);

  G.model_id      = m.model_id;
  G.n_tx          = m.n_tx;
  G.n_rx          = m.n_rx;
  G.L_full        = m.L;
  G.S             = m.S;
  G.fs_hz         = m.fs_hz;
  G.snapshot_dt_s = m.snapshot_dt_s;
  G.ds_ns         = m.ds_ns;
  G.speed_mps     = m.speed_mps;
  G.sel_offset    = m.offset_bytes;
  G.sel_nbytes    = m.nbytes;
  G.L_out         = (m.L <= MAX_L_PUBLISH ? (uint32_t)m.L : MAX_L_PUBLISH);
  G.snap_idx      = 0;

  G.snapshot_tmp_bytes = (size_t)G.n_tx * (size_t)G.n_rx *
                         (size_t)G.L_full * sizeof(struct complexf);
  G.snapshot_tmp = calloc(1, G.snapshot_tmp_bytes);
  AssertFatal(G.snapshot_tmp != NULL, "Alloc snapshot_tmp failed");

  for (int i = 0; i < NUM_TAPS_BUFFERS; i++) {
    size_t blob_cf = (size_t)G.n_tx * (size_t)G.n_rx * (size_t)G.L_out;
    G.bufs[i].taps_blob = calloc(1, blob_cf * sizeof(struct complexf));
    AssertFatal(G.bufs[i].taps_blob != NULL, "Alloc taps_blob failed");
    G.bufs[i].ch = (channel_desc_t *)calloc(1, sizeof(channel_desc_t));
    G.bufs[i].ch->ch_ps = (struct complexf **)calloc(G.n_rx * G.n_tx, sizeof(struct complexf *));
    G.bufs[i].ch->nb_tx = G.n_tx;
    G.bufs[i].ch->nb_rx = G.n_rx;
  }

  if (G.S > 0) {
    load_snapshot_and_publish(0);
  }

  LOG_I(HW, "CIRDB: model=%d DS=%.3fns shape=%ux%u L=%u/%u S=%u fs=%.0f dt=%.6fs speed=%.3fm/s",
        G.model_id, G.ds_ns, G.n_tx, G.n_rx, G.L_out, G.L_full,
        G.S, G.fs_hz, G.snapshot_dt_s, G.speed_mps);
}

void cirdb_update(uint64_t ns_since_start)
{
  if (!G.channel_desc_out || !*G.channel_desc_out) return;
  if (G.S <= 0) return;

  double dt_s = (G.snapshot_dt_s > 0.0 ? G.snapshot_dt_s : 0.5);
  if (dt_s <= 0.0) dt_s = 0.5;

  double steps_f = (ns_since_start * 1e-9) / dt_s;
  int64_t step = (int64_t)(steps_f >= 0.0 ? steps_f : 0.0);

  if (step != G.last_step_applied) {
    uint32_t s = (uint32_t)(step % G.S);
    load_snapshot_and_publish(s);
    G.snap_idx = s;
    G.last_step_applied = step;
  }
}

void cirdb_stop(void)
{
  for (int i = 0; i < NUM_TAPS_BUFFERS; i++) {
    free(G.bufs[i].taps_blob);
    if (G.bufs[i].ch) {
      free(G.bufs[i].ch->ch_ps);
      free(G.bufs[i].ch);
    }
  }
  free(G.snapshot_tmp);
  if (G.fh) fclose(G.fh);
  memset(&G, 0, sizeof(G));
  G.last_step_applied = -1;
}

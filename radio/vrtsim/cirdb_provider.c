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


 
/*
 * In-process CIR DB provider for VRTSIM
 * Reads cir_db.bin and periodically updates channel_desc->ch_ps
 * Layout in file is TX-major, RX-minor, then L complex taps, interleaved float32.
 */

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>

#include "common/utils/LOG/log.h"
#include "common/utils/assertions.h"
#include "SIMULATION/TOOLS/sim.h"
#include "cirdb_provider.h"

// Package header "<4sII"
typedef struct __attribute__((packed)) {
  char     magic[4];   // "CIRP"
  uint32_t version;    // 1
  uint32_t entry_count;
} cir_pkg_hdr_t;

// Directory entry "<Q H H H I I d d f f f f f f I Q Q"
typedef struct __attribute__((packed)) {
  uint64_t key;
  uint16_t model_id;         // TDL-A..E as 0..4
  uint16_t n_rx;
  uint16_t n_tx;
  uint32_t L;                // taps per link in file
  uint32_t S;                // snapshots
  double   fs_hz;
  double   fc_hz;
  float    ds_ns;
  float    speed_mps;
  float    rho_rx;
  float    rho_tx;
  float    snapshot_dt_s;
  float    k1_db_or_nan;
  uint32_t pair_order;       // 0 = TX-major then RX-minor
  uint64_t offset;           // start of snapshots
  uint64_t nbytes;           // total bytes for entry
} cir_dir_ent_t;

#define NUM_TAPS_BUFFERS 4
#define MAX_L_PUBLISH    8

typedef struct {
  void *taps_blob;            // contiguous complexf for all links
  channel_desc_t *ch;         // ch_ps points into taps_blob
} cirdb_buffer_t;

typedef struct {
  pthread_t thread;
  bool should_run;

  FILE *fh;
  cir_dir_ent_t *dir;
  size_t dir_count;

  cir_dir_ent_t sel;
  uint32_t L_out;
  uint32_t snap_idx;
  bool animate;
  float interval_s;

  int num_tx;
  int num_rx;

  cirdb_buffer_t bufs[NUM_TAPS_BUFFERS];
  int cur;

  void *snapshot_tmp;
  size_t snapshot_tmp_bytes;

  channel_desc_t **channel_desc_out;
} cirdb_g;

static cirdb_g G;

// Step 1 defaults
static const int   k_model_id    = 0;     // 0=TDL-A, 1=TDL-B, 2=TDL-C, 3=TDL-D, 4=TDL-E
static const float k_ds_ns       = 30.0f;
static const float k_speed_mps   = 1.5f;
static const bool  k_animate     = true;
static const float k_interval_s  = 0.5f;

/* ---------- path override handling and path helpers ---------- */

static char g_path_override[PATH_MAX] = {0};

void cirdb_set_path_override(const char *path) {
  if (!path) {
    g_path_override[0] = '\0';
    return;
  }
  size_t n = strnlen(path, PATH_MAX - 1);
  memcpy(g_path_override, path, n);
  g_path_override[n] = '\0';
}

// returns 1 if file exists and is readable
static int file_readable(const char *p) {
  return p && p[0] && access(p, R_OK) == 0;
}

static int get_exe_dir(char out[PATH_MAX]) {
  char exe[PATH_MAX] = {0};
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n <= 0) return 0;
  exe[n] = 0;
  char tmp[PATH_MAX];
  // dirname may modify its input
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

// return chosen path into out, even if not readable, so fopen can report
static const char *resolve_db_path(char out[PATH_MAX]) {
  // 1) explicit override via setter
  if (file_readable(g_path_override)) {
    snprintf(out, PATH_MAX, "%s", g_path_override);
    return out;
  }

  // 2) try exe-dir relatives
  char exe_dir[PATH_MAX];
  if (get_exe_dir(exe_dir)) {
    if (path_join(out, exe_dir, "cir_db.bin") && file_readable(out)) return out;
    if (path_join(out, exe_dir, "radio/vrtsim/cir_db.bin") && file_readable(out)) return out;
  }

  // 3) try CWD relatives
  snprintf(out, PATH_MAX, "%s", "cir_db.bin");
  if (file_readable(out)) return out;
  snprintf(out, PATH_MAX, "%s", "radio/vrtsim/cir_db.bin");
  if (file_readable(out)) return out;

  // 4) last resort, return default name and let fopen fail with a clear error
  snprintf(out, PATH_MAX, "%s", "cir_db.bin");
  return out;
}

/* ---------- allocation and IO helpers ---------- */

static inline void *xcalloc(size_t n, size_t sz) {
  void *p = calloc(n, sz);
  AssertFatal(p != NULL, "calloc failed for %zu x %zu", n, sz);
  return p;
}

static inline void fread_exact(void *dst, size_t sz, FILE *fh) {
  size_t r = fread(dst, 1, sz, fh);
  if (r != sz) {
    LOG_E(HW, "CIRDB short read: expected %zu got %zu errno=%d\n", sz, r, errno);
    abort();
  }
}

static inline size_t cf_bytes(size_t n) { return n * sizeof(struct complexf); }

/* ---------- channel_desc pointing and compaction ---------- */

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

/* ---------- selection ---------- */

// robust selector: exact match on model_id and shape, nearest on ds_ns and speed_mps
// also tries swapping tx/rx in case caller and DB differ on orientation
static bool select_entry_best(const cir_dir_ent_t *dir, size_t n,
                              int want_model, int want_tx, int want_rx,
                              float want_ds_ns, float want_speed,
                              cir_dir_ent_t *out) {
  const float W_DS = 1.0f;        // weight for delay spread
  const float W_SP = 0.2f;        // weight for speed
  bool found = false;
  double best_cost = 1e300;

  for (size_t i = 0; i < n; i++) {
    const cir_dir_ent_t *e = &dir[i];
    if (e->pair_order != 0) continue;
    if (e->model_id != (uint16_t)want_model) continue;

    bool shape_ok = (e->n_tx == (uint16_t)want_tx && e->n_rx == (uint16_t)want_rx);
    bool shape_ok_swap = (e->n_tx == (uint16_t)want_rx && e->n_rx == (uint16_t)want_tx);

    if (!shape_ok && !shape_ok_swap) continue;

    double cds = fabs((double)e->ds_ns - (double)want_ds_ns);
    double csp = fabs((double)e->speed_mps - (double)want_speed);
    double cost = W_DS * cds + W_SP * csp;

    if (!found || cost < best_cost) {
      *out = *e;
      best_cost = cost;
      found = true;
    }
  }
  return found;
}

/* ---------- worker thread ---------- */

static void *worker(void *arg) {
  (void)arg;
  while (G.should_run) {
    int next = (G.cur + 1) % NUM_TAPS_BUFFERS;
    cirdb_buffer_t *buf = &G.bufs[next];

    uint64_t stride_bytes = (uint64_t)G.sel.n_tx * (uint64_t)G.sel.n_rx *
                            (uint64_t)G.sel.L * sizeof(struct complexf);
    uint64_t off = G.sel.offset + (uint64_t)G.snap_idx * stride_bytes;

    if (fseeko(G.fh, (off_t)off, SEEK_SET) != 0) {
      LOG_E(HW, "CIRDB fseeko failed errno=%d\n", errno);
      break;
    }
    fread_exact(G.snapshot_tmp, (size_t)stride_bytes, G.fh);

    compact_L_out((struct complexf *)buf->taps_blob,
                  (const struct complexf *)G.snapshot_tmp,
                  G.num_tx, G.num_rx, (int)G.sel.L, (int)G.L_out);

    point_channel_desc(buf->ch,
                       (struct complexf *)buf->taps_blob,
                       G.num_tx, G.num_rx, (int)G.L_out);

    *G.channel_desc_out = buf->ch;
    G.cur = next;

    if (G.animate) G.snap_idx = (G.snap_idx + 1) % G.sel.S;

    useconds_t us = (useconds_t)(G.interval_s * 1e6f);
    if (us == 0) us = 1;
    usleep(us);
  }
  return NULL;
}

/* ---------- public API ---------- */

void cirdb_connect(int id,
                   int num_tx_antennas,
                   int num_rx_antennas,
                   channel_desc_t **channel_desc_out) {
  (void)id;
  memset(&G, 0, sizeof(G));
  G.should_run        = true;
  G.num_tx            = num_tx_antennas;
  G.num_rx            = num_rx_antennas;
  G.animate           = k_animate;
  G.interval_s        = k_interval_s;
  G.channel_desc_out  = channel_desc_out;

  char dbpath[PATH_MAX];
  const char *use_path = resolve_db_path(dbpath);
  G.fh = fopen(use_path, "rb");
  if (!G.fh) {
    LOG_E(HW, "open %s failed, errno=%d", use_path, errno);
    abort();
  }

  cir_pkg_hdr_t hdr;
  fread_exact(&hdr, sizeof(hdr), G.fh);
  if (memcmp(hdr.magic, "CIRP", 4) != 0 || hdr.version != 1) {
    LOG_E(HW, "Bad CIR DB header");
    abort();
  }
  G.dir_count = hdr.entry_count;
  G.dir = (cir_dir_ent_t *)xcalloc(G.dir_count, sizeof(cir_dir_ent_t));
  fread_exact(G.dir, G.dir_count * sizeof(cir_dir_ent_t), G.fh);

  // robust selection
  bool ok = select_entry_best(G.dir, G.dir_count,
                              k_model_id, G.num_tx, G.num_rx,
                              k_ds_ns, k_speed_mps, &G.sel);
  if (!ok) {
    LOG_E(HW, "Entry not found after nearest search: model=%d want %dx%d DS=%.3fns speed=%.3fm/s",
          k_model_id, G.num_tx, G.num_rx, k_ds_ns, k_speed_mps);
    // try with swapped shape as last resort, in case PHY requested reversed
    ok = select_entry_best(G.dir, G.dir_count,
                           k_model_id, G.num_rx, G.num_tx,
                           k_ds_ns, k_speed_mps, &G.sel);
    if (ok) {
      LOG_W(HW, "Using DB entry with swapped shape %ux%u instead of requested %ux%u",
            G.sel.n_tx, G.sel.n_rx, G.num_tx, G.num_rx);
      // use DB shape for publication
      G.num_tx = G.sel.n_tx;
      G.num_rx = G.sel.n_rx;
    }
  }
  AssertFatal(ok, "No suitable CIR entry found in DB");

  G.L_out = G.sel.L < MAX_L_PUBLISH ? G.sel.L : MAX_L_PUBLISH;

  G.snapshot_tmp_bytes = (size_t)G.sel.n_tx * (size_t)G.sel.n_rx *
                         (size_t)G.sel.L * sizeof(struct complexf);
  G.snapshot_tmp = xcalloc(1, G.snapshot_tmp_bytes);

  for (int i = 0; i < NUM_TAPS_BUFFERS; i++) {
    size_t blob_cf = (size_t)G.num_tx * (size_t)G.num_rx * (size_t)G.L_out;
    G.bufs[i].taps_blob = xcalloc(1, blob_cf * sizeof(struct complexf));
    G.bufs[i].ch = (channel_desc_t *)xcalloc(1, sizeof(channel_desc_t));
    G.bufs[i].ch->ch_ps = (struct complexf **)xcalloc(G.num_rx * G.num_tx, sizeof(struct complexf *));
    G.bufs[i].ch->nb_tx = G.num_tx;
    G.bufs[i].ch->nb_rx = G.num_rx;
  }
  G.cur = 0;

  LOG_I(HW, "CIRDB ready: model=%u DS=%.3fns shape=%ux%u L=%u/%u S=%u fs=%.0f dt=%.6fs speed=%.3fm/s",
        G.sel.model_id, G.sel.ds_ns, G.num_tx, G.num_rx,
        G.L_out, G.sel.L, G.sel.S, G.sel.fs_hz, G.sel.snapshot_dt_s, G.sel.speed_mps);

  int ret = pthread_create(&G.thread, NULL, worker, NULL);
  if (ret != 0) {
    LOG_E(HW, "pthread_create failed errno=%d", errno);
    abort();
  }
}

void cirdb_stop(void) {
  if (!G.should_run) return;
  G.should_run = false;
  pthread_join(G.thread, NULL);

  for (int i = 0; i < NUM_TAPS_BUFFERS; i++) {
    free(G.bufs[i].taps_blob);
    if (G.bufs[i].ch) {
      free(G.bufs[i].ch->ch_ps);
      free(G.bufs[i].ch);
    }
  }
  free(G.snapshot_tmp);
  free(G.dir);
  if (G.fh) fclose(G.fh);
}

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

#include "cirdb_yaml.h"
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cmath>
#include <string>

static inline double dabs(double x){ return x < 0 ? -x : x; }

int cirdb_yaml_scan(const char *yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    auto ents = root["entries"];
    if (!ents || !ents.IsSequence()) return -1;
    int n = static_cast<int>(ents.size());
    std::fprintf(stderr, "[cirdb_yaml] %d entries in %s\n", n, yaml_path);
    return n;
  } catch (const YAML::Exception &e) {
    std::fprintf(stderr, "[cirdb_yaml] parse error: %s\n", e.what());
    return -2;
  }
}

int cirdb_yaml_select(const cirdb_select_req_t *req, cirdb_entry_meta_t *out) {
  if (!req || !out || !req->yaml_path) return -3;
  try {
    YAML::Node root = YAML::LoadFile(req->yaml_path);
    auto ents = root["entries"];
    if (!ents || !ents.IsSequence()) return -4;

    bool found = false;
    double best_cost = 1e300;
    cirdb_entry_meta_t best{};

    for (std::size_t i = 0; i < ents.size(); ++i) {
        const YAML::Node e = ents[i];
        int pair_order = e["pair_order"].as<int>(0);

      int model_id = e["model_id"].as<int>(-1);
      if (req->want_model_id >= 0 && model_id != req->want_model_id) continue;

      int n_tx = e["n_tx"].as<int>(0);
      int n_rx = e["n_rx"].as<int>(0);

      bool shape_ok = (n_tx == req->want_tx && n_rx == req->want_rx);
      bool shape_swap_ok = req->allow_shape_swap ? (n_tx == req->want_rx && n_rx == req->want_tx) : false;
      if (!shape_ok && !shape_swap_ok) continue;

      float ds_ns = e["ds_ns"].as<float>(0.0f);
      float sp    = e["speed_mps"].as<float>(0.0f);

      double cds = dabs((double)ds_ns - (double)req->want_ds_ns);
      double csp = dabs((double)sp - (double)req->want_speed_mps);
      double cost = (req->w_ds > 0 ? req->w_ds : 1.0) * cds
                  + (req->w_speed > 0 ? req->w_speed : 0.2) * csp;

      if (!found || cost < best_cost) {
        best_cost = cost;
        found = true;

        best.model_id = model_id;
        best.n_tx     = n_tx;
        best.n_rx     = n_rx;
        best.L        = e["L"].as<int>(0);
        best.S        = e["S"].as<int>(0);
        best.fs_hz    = e["fs_hz"].as<double>(0.0);
        best.snapshot_dt_s = e["snapshot_dt_s"].as<double>(0.0);
        best.ds_ns    = ds_ns;
        best.speed_mps= sp;
        best.pair_order = pair_order;
        best.offset_bytes = e["offset_bytes"].as<uint64_t>(0);
        best.nbytes   = e["nbytes"].as<uint64_t>(0);
      }
    }

    if (!found) return 0;
    *out = best;
    return 1;
  } catch (const YAML::Exception &e) {
    std::fprintf(stderr, "[cirdb_yaml] parse error: %s\n", e.what());
    return -2;
  }
}

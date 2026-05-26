// SPDX-License-Identifier: LicenseRef-CSSL-1.0
//
// ldpc_hexagon_imp.c — Hexagon DSP LDPC decoder, HVX-accelerated min-sum.
//
// Compiled with hexagon-clang -O3 -mhvx -mhvx-length=128B and loaded into
// the cDSP protected domain via FastRPC.
//
// Algorithm:
//   Standard min-sum belief propagation:
//     CN step: for each check-node edge, output = sign(∏ others) × min(|others|)
//     BN step: posterior = channel_LLR + Σ(CN messages);
//              extrinsic = posterior − self CN message
//   Convergence: XOR of sign bits across all connected BNs at each CN = 0
//
// cnProc_group() vectorises the CN step with 128B HVX intrinsics, processing
// Z samples (one per lifting symbol) in parallel.  Full 128-byte HVX vectors
// are used for floor(Z/128) chunks; the remaining tail bytes are handled by
// a scalar fallback.  For Z=384 (= 3×128) there is no tail — pure HVX path.

// Suppress OAI framework headers (time_meas.h etc.) inside nrLDPC_types.h
#define CODEGEN 1

// nrLDPC_mPass.h unconditionally includes common/utils/utils.h, which pulls in
// <malloc.h> — not available in the Hexagon DSP toolchain.  Block it by setting
// its include guard before nrLDPC_mPass.h is seen, and provide the one macro
// from utils.h that mPass.h actually uses.
#ifndef _UTILS_H
#define _UTILS_H
#define sizeofArray(a) ((int)(sizeof(a) / sizeof((a)[0])))
#endif

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef LDPC_SIM_STANDALONE
// Standalone Hexagon simulator build — no FastRPC / HAP framework.
#  include <stdio.h>
#  define FARF(level, fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#else
#  include "HAP_farf.h"
#  include "HAP_power.h"
#endif

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define HVX_VLEN 128   // HVX vector width in bytes (-mhvx-length=128B)

// Portable LDPC decoder headers — no SIMD, no OAI framework dependencies
#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_lut.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"

#ifndef LDPC_SIM_STANDALONE
// qaic-generated dispatch header (not present in standalone sim build)
#include "ldpc_hexagon.h"
#endif

// Wire parameter struct — must match ldpc_hexagon_arm.c
typedef struct __attribute__((packed)) {
    uint8_t  BG;
    uint8_t  R;
    uint8_t  numMaxIter;
    uint8_t  diag;      // 0=memcpy passthrough, 1=scatter/gather roundtrip, 2+=normal BP
    uint16_t Z;
    uint16_t pad1;
} ldpc_hex_params_t;

// =============================================================================
// Saturating arithmetic helpers
// =============================================================================

static inline int8_t sat8_add(int8_t a, int8_t b) {
    int16_t r = (int16_t)a + (int16_t)b;
    return r >  127 ?  127 : (r < -128 ? -128 : (int8_t)r);
}

static inline int8_t sat8_sub(int8_t a, int8_t b) {
    int16_t r = (int16_t)a - (int16_t)b;
    return r >  127 ?  127 : (r < -128 ? -128 : (int8_t)r);
}

// Absolute value for int8; handles -128 safely (clamps to 127)
static inline uint8_t abs8(int8_t v) {
    int16_t w = (int16_t)v;
    int16_t a = w < 0 ? -w : w;
    return (uint8_t)(a > 127 ? 127 : a);
}

// =============================================================================
// BN group sizes per CN group (fixed by base-graph structure)
// =============================================================================

static const int8_t numBN_BG1[NR_LDPC_NUM_CN_GROUPS_BG1] = {3, 4, 5, 6, 7, 8, 9, 10, 19};
static const int8_t numBN_BG2[NR_LDPC_NUM_CN_GROUPS_BG2] = {3, 4, 5, 6, 8, 10};

// =============================================================================
// CN Processing — HVX-vectorised min-sum
// =============================================================================
// For every (CN i, lifting symbol z) in a group:
//   Pass 1 over all BN positions k: track first/second minimum |v|, XOR signs.
//   Pass 2: write extrinsic message for each position j:
//     mag_j  = (|v_j| == min1) ? min2 : min1   (excludes j's contribution)
//     sign_j = sgn_all XOR v_j                   (removes j's sign from product)
//     result = sign_j < 0 ? -mag_j : mag_j
//
// The Z lifting symbols are processed in parallel with 128B HVX vectors.
// floor(Z/128) full vectors are handled by HVX; the residual tail (Z % 128)
// falls back to scalar.  For Z = 384 = 3×128 there is no tail.
//
// Min-sum tie handling: the "mag = (|v_j|==min1)?min2:min1" rule approximates
// the correct "exclude-self" minimum when only one BN achieves min1.  For ties
// (multiple BNs with equal minimum) both positions return min2 rather than min1;
// this is a standard approximation used in all vectorised implementations and
// has negligible effect on convergence.

static void cnProc_group(
    const int8_t *cnProcBuf, int8_t *cnProcBufRes,
    uint32_t startAddr, int numBN, int numCN, uint16_t Z)
{
    int32_t bitOff = (int32_t)numCN * NR_LDPC_ZMAX;
    int32_t full   = (Z / HVX_VLEN) * HVX_VLEN;  // bytes covered by full HVX vectors
    int32_t tail   = Z - full;

    // Constants: initialise once, reuse across all (i, v) iterations.
    HVX_Vector vzero   = Q6_V_vzero();
    HVX_Vector vmax_ub = Q6_Vb_vsplat_R(0x7f);  // 127 in every byte lane

    for (int i = 0; i < numCN; i++) {
        int32_t row = i * Z;  // byte offset of this CN's Z samples within each k-layer

        // ── HVX path: one 128-byte vector per iteration ──────────────────────
        for (int32_t voff = 0; voff < full; voff += HVX_VLEN) {

            // Pass 1: accumulate min1, min2, sgn_all across all numBN messages.
            HVX_Vector vmin1    = vmax_ub;
            HVX_Vector vmin2    = vmax_ub;
            HVX_Vector vsgn_all = vzero;

            for (int k = 0; k < numBN; k++) {
                HVX_Vector vk = *(HVX_UVector *)(cnProcBuf + startAddr
                                                 + (int32_t)k * bitOff + row + voff);
                HVX_Vector vabs = Q6_Vb_vabs_Vb(vk);       // saturates: |-128| → 127
                vsgn_all = Q6_V_vxor_VV(vsgn_all, vk);

                // Update min1 and min2:
                //   where vabs < vmin1: demote old vmin1 to min2 candidate, update min1
                //   where vabs >= vmin1: vabs itself is min2 candidate
                HVX_VectorPred new_min  = Q6_Q_vcmp_gt_VubVub(vmin1, vabs);
                HVX_Vector     old_min1 = vmin1;
                vmin1 = Q6_Vub_vmin_VubVub(vmin1, vabs);
                HVX_Vector min2_cand = Q6_V_vmux_QVV(new_min, old_min1, vabs);
                vmin2 = Q6_Vub_vmin_VubVub(vmin2, min2_cand);
            }

            // Pass 2: compute and store extrinsic for each BN position.
            for (int k = 0; k < numBN; k++) {
                const int8_t *in  = cnProcBuf    + startAddr + (int32_t)k * bitOff + row + voff;
                int8_t       *out = cnProcBufRes + startAddr + (int32_t)k * bitOff + row + voff;
                HVX_Vector vk   = *(HVX_UVector *)in;
                HVX_Vector vabs = Q6_Vb_vabs_Vb(vk);

                // mag: use vmin2 where this position held the overall minimum,
                //      vmin1 everywhere else.
                HVX_VectorPred is_min1 = Q6_Q_vcmp_eq_VbVb(vabs, vmin1);
                HVX_Vector vmag = Q6_V_vmux_QVV(is_min1, vmin2, vmin1);

                // sign_j = sgn_all XOR vk  (removes vk's sign from the product)
                HVX_Vector vsign_j = Q6_V_vxor_VV(vsgn_all, vk);

                // result = (sign_j < 0) ? -vmag : +vmag
                HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VbVb(vzero, vsign_j);
                HVX_Vector vneg       = Q6_Vb_vsub_VbVb(vzero, vmag);
                *(HVX_UVector *)out   = Q6_V_vmux_QVV(is_neg, vneg, vmag);
            }
        }

        // ── Scalar fallback: residual tail bytes (Z % 128 != 0) ─────────────
        for (int32_t z = full; z < Z; z++) {
            int32_t samp = row + z;
            int8_t  sgn_all = 0;
            uint8_t min1 = 127, min2 = 127;
            int     min1_k = 0;

            for (int k = 0; k < numBN; k++) {
                int8_t  v  = cnProcBuf[startAddr + (uint32_t)k * bitOff + samp];
                sgn_all   ^= v;
                uint8_t av = abs8(v);
                if (av < min1) { min2 = min1; min1 = av; min1_k = k; }
                else if (av < min2) { min2 = av; }
            }
            for (int j = 0; j < numBN; j++) {
                int8_t  v_j   = cnProcBuf[startAddr + (uint32_t)j * bitOff + samp];
                int8_t  sgn_j = sgn_all ^ v_j;
                uint8_t mag   = (j == min1_k) ? min2 : min1;
                cnProcBufRes[startAddr + (uint32_t)j * bitOff + samp] =
                    (sgn_j & 0x80) ? -(int8_t)mag : (int8_t)mag;
            }
        }
    }
}

static void cnProc(t_nrLDPC_lut *p_lut,
                   const int8_t *cnProcBuf, int8_t *cnProcBufRes,
                   uint16_t Z, uint8_t BG)
{
    const int8_t *tbl  = (BG == 1) ? numBN_BG1 : numBN_BG2;
    int           ngrp = (BG == 1) ? NR_LDPC_NUM_CN_GROUPS_BG1
                                   : NR_LDPC_NUM_CN_GROUPS_BG2;
    for (int g = 0; g < ngrp; g++) {
        int numCN = (int)p_lut->numCnInCnGroups[g];
        if (numCN == 0) continue;
        cnProc_group(cnProcBuf, cnProcBufRes,
                     p_lut->startAddrCnGroups[g],
                     (int)tbl[g], numCN, Z);
    }
}

// =============================================================================
// Parity Check — cnProcPc (scalar)
// =============================================================================
// After bn2cnProcBuf, cnProcBuf holds BN→CN extrinsic messages.
// XOR of sign bits across all BN positions at each CN = 0 ↔ parity satisfied.
// Returns 0 if all CNs pass, 1 otherwise.

static int32_t scalar_cnProcPc(const int8_t *cnProcBuf, t_nrLDPC_lut *p_lut,
                                uint16_t Z, uint8_t BG)
{
    const int8_t *tbl  = (BG == 1) ? numBN_BG1 : numBN_BG2;
    int           ngrp = (BG == 1) ? NR_LDPC_NUM_CN_GROUPS_BG1
                                   : NR_LDPC_NUM_CN_GROUPS_BG2;
    for (int g = 0; g < ngrp; g++) {
        int numCN = (int)p_lut->numCnInCnGroups[g];
        if (numCN == 0) continue;
        int      numBN = (int)tbl[g];
        uint32_t bitOff = (uint32_t)numCN * NR_LDPC_ZMAX;
        uint32_t sa     = p_lut->startAddrCnGroups[g];

        for (int i = 0; i < numCN; i++) {
            for (int z = 0; z < (int)Z; z++) {
                int samp = i * (int)Z + z;
                int8_t sgn = 0;
                for (int k = 0; k < numBN; k++)
                    sgn ^= cnProcBuf[sa + (uint32_t)k * bitOff + samp];
                if (sgn & 0x80) return 1;
            }
        }
    }
    return 0;
}

// =============================================================================
// HVX helpers — saturating int8 arithmetic via int16 widening
// =============================================================================
// HVX has no single-instruction signed-byte saturating add/sub.
// Strategy: widen to int16 (Q6_Wh_vsxt_Vb), add/sub in 16-bit, clamp to
// [-128,127] with vmin/vmax, pack low bytes back (Q6_Vb_vpacke_VhVh).
//
// vpacke(Vu.h, Vv.h): output[0..63]  = low bytes of Vv halfwords,
//                      output[64..127] = low bytes of Vu halfwords.
// After clamping, the low byte of each int16 is the correct int8 value.

static inline HVX_Vector hvx_sat8_add(HVX_Vector va, HVX_Vector vb,
                                       HVX_Vector v127, HVX_Vector vn128)
{
    HVX_VectorPair wa = Q6_Wh_vsxt_Vb(va), wb = Q6_Wh_vsxt_Vb(vb);
    HVX_Vector lo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wa), Q6_V_lo_W(wb));
    HVX_Vector hi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wa), Q6_V_hi_W(wb));
    lo = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(lo, v127), vn128);
    hi = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(hi, v127), vn128);
    return Q6_Vb_vpacke_VhVh(hi, lo);
}

static inline HVX_Vector hvx_sat8_sub(HVX_Vector va, HVX_Vector vb,
                                       HVX_Vector v127, HVX_Vector vn128)
{
    HVX_VectorPair wa = Q6_Wh_vsxt_Vb(va), wb = Q6_Wh_vsxt_Vb(vb);
    HVX_Vector lo = Q6_Vh_vsub_VhVh(Q6_V_lo_W(wa), Q6_V_lo_W(wb));
    HVX_Vector hi = Q6_Vh_vsub_VhVh(Q6_V_hi_W(wa), Q6_V_hi_W(wb));
    lo = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(lo, v127), vn128);
    hi = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(hi, v127), vn128);
    return Q6_Vb_vpacke_VhVh(hi, lo);
}

// 2D L2 prefetch: `height` rows of `width` bytes, rows spaced `stride` bytes apart.
// Descriptor format (Hexagon ISA): [stride:16 | width:16 | height:16] in bits [47:0].
// Useful for strided access patterns that the hardware prefetcher cannot track.
static inline void hvx_l2fetch_2d(const void *base,
                                   uint32_t stride, uint32_t width, uint32_t height)
{
    uint64_t desc = ((uint64_t)(stride & 0xFFFFu) << 32)
                  | ((uint64_t)(width  & 0xFFFFu) << 16)
                  |  (uint64_t)(height & 0xFFFFu);
    Q6_l2fetch_AP((void *)(uintptr_t)base, desc);
}

// =============================================================================
// BN Processing — parity-check variant (HVX bnProcPc)
// =============================================================================
// For each BN b in group g (cnidx+1 connected CNs):
//   llrRes[b]         = sat8( llrProcBuf[b] + Σ_k bnProcBuf[k, b] )
//   bnProcBufRes[0,b] = llrProcBuf[b]   (stored for bnProc subtraction)
//
// Buffer layout in bnProcBuf/bnProcBufRes:
//   CN layer k: byte range [startBn + k * numBn * ZMAX  ..  + numBn * ZMAX - 1]
//   BN b within layer k: bytes [k*cnOff + b]  (b in 0..numBn*Z-1)
//
// HVX path: outer loop steps 128 bytes at a time; inner k-loop accumulates
// int16 sums in two register-pair halves, then saturates and packs to int8.
// All accesses within a group are sequential — no cache-miss-latency issue.

static void hvx_bnProcPc(t_nrLDPC_lut *p_lut,
                          const int8_t *bnProcBuf, int8_t *bnProcBufRes,
                          const int8_t *llrProcBuf, int8_t *llrRes,
                          uint16_t Z)
{
    const uint8_t  *numBn = p_lut->numBnInBnGroups;
    const uint32_t *sBn   = p_lut->startAddrBnGroups;
    const uint16_t *sLlr  = p_lut->startAddrBnGroupsLlr;

    HVX_Vector v127  = Q6_Vh_vsplat_R(127);
    HVX_Vector vn128 = Q6_Vh_vsplat_R(-128);

    // Group 0: BNs connected to exactly 1 CN
    {
        uint32_t nb    = (uint32_t)numBn[0];
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[0];
        uint32_t sl    = sLlr[0];
        memcpy(&bnProcBufRes[sa], &llrProcBuf[sl], total);
        uint32_t full = (total / HVX_VLEN) * HVX_VLEN;
        for (uint32_t b = 0; b < full; b += HVX_VLEN)
            *(HVX_UVector *)(llrRes + sl + b) = hvx_sat8_add(
                *(HVX_UVector *)(llrProcBuf + sl + b),
                *(HVX_UVector *)(bnProcBuf  + sa + b), v127, vn128);
        for (uint32_t b = full; b < total; b++)
            llrRes[sl + b] = sat8_add(llrProcBuf[sl + b], bnProcBuf[sa + b]);
    }

    // Groups 1+: accumulate cnidx+1 CN messages per BN
    uint8_t idxBnGroup = 0;
    for (int cnidx = 1; cnidx < NR_LDPC_NUM_BN_GROUPS_BG1_R13; cnidx++) {
        if (numBn[cnidx] == 0) continue;
        idxBnGroup++;

        uint32_t nb    = (uint32_t)numBn[cnidx];
        uint32_t cnOff = nb * NR_LDPC_ZMAX;
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[idxBnGroup];
        uint32_t sl    = sLlr[idxBnGroup];
        uint32_t full  = (total / HVX_VLEN) * HVX_VLEN;

        for (uint32_t b = 0; b < full; b += HVX_VLEN) {
            // Prefetch next b-chunk across all k-layers: stride=cnOff, width=128, height=cnidx+1.
            // Covers the strided bnProcBuf accesses the hardware prefetcher cannot detect.
            if (b + HVX_VLEN < full)
                hvx_l2fetch_2d(bnProcBuf + sa + b + HVX_VLEN,
                               cnOff, HVX_VLEN, (uint32_t)cnidx + 1);

            // Init accumulator with channel LLR (widened to int16)
            HVX_VectorPair wsum = Q6_Wh_vsxt_Vb(*(HVX_UVector *)(llrProcBuf + sl + b));
            HVX_Vector sum_lo = Q6_V_lo_W(wsum);
            HVX_Vector sum_hi = Q6_V_hi_W(wsum);

            for (int k = 0; k <= cnidx; k++) {
                HVX_VectorPair wbn = Q6_Wh_vsxt_Vb(
                    *(HVX_UVector *)(bnProcBuf + sa + (uint32_t)k * cnOff + b));
                sum_lo = Q6_Vh_vadd_VhVh(sum_lo, Q6_V_lo_W(wbn));
                sum_hi = Q6_Vh_vadd_VhVh(sum_hi, Q6_V_hi_W(wbn));
            }

            sum_lo = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(sum_lo, v127), vn128);
            sum_hi = Q6_Vh_vmax_VhVh(Q6_Vh_vmin_VhVh(sum_hi, v127), vn128);
            *(HVX_UVector *)(llrRes + sl + b) = Q6_Vb_vpacke_VhVh(sum_hi, sum_lo);
        }

        for (uint32_t b = full; b < total; b++) {
            int16_t s = (int16_t)llrProcBuf[sl + b];
            for (int k = 0; k <= cnidx; k++)
                s += (int16_t)bnProcBuf[sa + (uint32_t)k * cnOff + b];
            llrRes[sl + b] = s > 127 ? 127 : s < -128 ? -128 : (int8_t)s;
        }
    }
}

// =============================================================================
// BN Processing — extrinsic message (HVX bnProc)
// =============================================================================
// For each BN-to-CN edge (group g, CN layer k, BN b):
//   bnProcBufRes[k, b] = sat8( llrRes[b] - bnProcBuf[k, b] )
// Group 0 was already handled by bnProcPc (1-CN BNs: extrinsic = channel LLR).
//
// Inner loop over b is sequential in all buffers — ideal for HVX.

static void hvx_bnProc(t_nrLDPC_lut *p_lut,
                        const int8_t *bnProcBuf, int8_t *bnProcBufRes,
                        const int8_t *llrRes, uint16_t Z)
{
    const uint8_t  *numBn = p_lut->numBnInBnGroups;
    const uint32_t *sBn   = p_lut->startAddrBnGroups;
    const uint16_t *sLlr  = p_lut->startAddrBnGroupsLlr;

    HVX_Vector v127  = Q6_Vh_vsplat_R(127);
    HVX_Vector vn128 = Q6_Vh_vsplat_R(-128);

    uint8_t idxBnGroup = 0;
    for (int cnidx = 1; cnidx < NR_LDPC_NUM_BN_GROUPS_BG1_R13; cnidx++) {
        if (numBn[cnidx] == 0) continue;
        idxBnGroup++;

        uint32_t nb    = (uint32_t)numBn[cnidx];
        uint32_t cnOff = nb * NR_LDPC_ZMAX;
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[idxBnGroup];
        uint32_t sl    = sLlr[idxBnGroup];
        uint32_t full  = (total / HVX_VLEN) * HVX_VLEN;

        for (int k = 0; k <= cnidx; k++) {
            uint32_t off = (uint32_t)k * cnOff;
            // Prefetch first 128 bytes of the next k-layer to hide inter-layer stride latency.
            if (k < cnidx) {
                const int8_t *nxt = bnProcBuf + sa + (uint32_t)(k + 1) * cnOff;
                Q6_dcfetch_A((void *)(uintptr_t)nxt);
                Q6_dcfetch_A((void *)(uintptr_t)(nxt + 32));
                Q6_dcfetch_A((void *)(uintptr_t)(nxt + 64));
                Q6_dcfetch_A((void *)(uintptr_t)(nxt + 96));
            }
            for (uint32_t b = 0; b < full; b += HVX_VLEN)
                *(HVX_UVector *)(bnProcBufRes + sa + off + b) = hvx_sat8_sub(
                    *(HVX_UVector *)(llrRes    + sl      + b),
                    *(HVX_UVector *)(bnProcBuf + sa + off + b), v127, vn128);
            for (uint32_t b = full; b < total; b++)
                bnProcBufRes[sa + off + b] =
                    sat8_sub(llrRes[sl + b], bnProcBuf[sa + off + b]);
        }
    }
}

// =============================================================================
// DSP session context — forward declaration (full definition before open/close)
// =============================================================================
typedef struct {
    int8_t *cnProcBuf;
    int8_t *cnProcBufRes;
    int8_t *bnProcBuf;
    int8_t *bnProcBufRes;
    int8_t *llrRes;
    int8_t *llrProcBuf;
} ldpc_dsp_ctx_t;

// =============================================================================
// Main decoder core — uses pre-allocated buffers from session context
// =============================================================================
// Returns number of BP iterations completed.

static int32_t ldpc_scalar_core(
    const int8_t *p_llr, uint8_t *llr_out, uint32_t numLLR,
    t_nrLDPC_lut *p_lut, uint8_t BG, uint16_t Z, uint8_t numMaxIter,
    ldpc_dsp_ctx_t *ctx)
{
    int8_t *cnProcBuf    = ctx->cnProcBuf;
    int8_t *cnProcBufRes = ctx->cnProcBufRes;
    int8_t *bnProcBuf    = ctx->bnProcBuf;
    int8_t *bnProcBufRes = ctx->bnProcBufRes;
    int8_t *llrRes       = ctx->llrRes;
    int8_t *llrProcBuf   = ctx->llrProcBuf;
    int32_t numIter = 0;

    memset(cnProcBuf,    0, NR_LDPC_SIZE_CN_PROC_BUF);
    memset(cnProcBufRes, 0, NR_LDPC_SIZE_CN_PROC_BUF);
    memset(bnProcBuf,    0, NR_LDPC_SIZE_BN_PROC_BUF);
    memset(bnProcBufRes, 0, NR_LDPC_SIZE_BN_PROC_BUF);

    nrLDPC_llr2llrProcBuf(p_lut, (int8_t *)p_llr, llrProcBuf, Z, BG);
    if (BG == 1)
        nrLDPC_llr2CnProcBuf_BG1(p_lut, (int8_t *)p_llr, cnProcBuf, Z);
    else
        nrLDPC_llr2CnProcBuf_BG2(p_lut, (int8_t *)p_llr, cnProcBuf, Z);

    // First iteration without parity check
    cnProc(p_lut, cnProcBuf, cnProcBufRes, Z, BG);

    if (BG == 1)
        nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
    else
        nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);

    hvx_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
    hvx_bnProc  (p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);

    if (BG == 1)
        nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
    else
        nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);

    // BP iteration loop with parity check
    int32_t pcRes = 1;
    while (numIter < (int32_t)numMaxIter && pcRes != 0) {
        cnProc(p_lut, cnProcBuf, cnProcBufRes, Z, BG);

        if (BG == 1)
            nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
        else
            nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);

        hvx_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
        hvx_bnProc  (p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);

        if (BG == 1)
            nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
        else
            nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);

        pcRes = scalar_cnProcPc(cnProcBuf, p_lut, Z, BG);
        numIter++;
    }

    nrLDPC_llrRes2llrOut(p_lut, (int8_t *)llr_out, llrRes, Z, BG);
    return numIter;
}

// =============================================================================
// FastRPC session lifecycle and RPC entry point
// Not compiled in standalone simulator builds.
// =============================================================================
#ifndef LDPC_SIM_STANDALONE

int ldpc_hexagon_open(const char *uri, remote_handle64 *handle)
{
    ldpc_dsp_ctx_t *ctx = calloc(1, sizeof(ldpc_dsp_ctx_t));
    if (!ctx) return -1;

    ctx->cnProcBuf    = malloc(NR_LDPC_SIZE_CN_PROC_BUF);
    ctx->cnProcBufRes = malloc(NR_LDPC_SIZE_CN_PROC_BUF);
    ctx->bnProcBuf    = malloc(NR_LDPC_SIZE_BN_PROC_BUF);
    ctx->bnProcBufRes = malloc(NR_LDPC_SIZE_BN_PROC_BUF);
    ctx->llrRes       = malloc(NR_LDPC_MAX_NUM_LLR);
    ctx->llrProcBuf   = malloc(NR_LDPC_MAX_NUM_LLR);

    if (!ctx->cnProcBuf || !ctx->cnProcBufRes || !ctx->bnProcBuf ||
        !ctx->bnProcBufRes || !ctx->llrRes || !ctx->llrProcBuf) {
        FARF(ERROR, "ldpc_hexagon: working buffer allocation failed");
        free(ctx->cnProcBuf);    free(ctx->cnProcBufRes);
        free(ctx->bnProcBuf);    free(ctx->bnProcBufRes);
        free(ctx->llrRes);       free(ctx->llrProcBuf);
        free(ctx);
        return -1;
    }

    // Vote for turbo core clock and maximum bus bandwidth.
    // Without this the cDSP runs at a low-power DCVS corner (~700 MHz).
    // The vote persists for the lifetime of this handle.
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable   = TRUE;
    req.dcvs_v3.dcvs_enable       = FALSE;   // fix clock, disable DCVS scaling
    req.dcvs_v3.set_latency        = TRUE;
    req.dcvs_v3.latency            = 1;       // 1 µs: disable DSP sleep between calls
    req.dcvs_v3.set_core_params    = TRUE;
    req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.set_bus_params     = TRUE;
    req.dcvs_v3.bus_params.min_corner    = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.bus_params.max_corner    = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    int rc = HAP_power_set(NULL, &req);
    if (rc)
        FARF(HIGH, "ldpc_hexagon: HAP_power_set(DCVS_v3/TURBO) returned %d", rc);

    *handle = (remote_handle64)(uintptr_t)ctx;
    FARF(RUNTIME_HIGH, "ldpc_hexagon: session opened, working buffers pre-allocated");
    return 0;
}

int ldpc_hexagon_close(remote_handle64 handle)
{
    ldpc_dsp_ctx_t *ctx = (ldpc_dsp_ctx_t *)(uintptr_t)handle;
    if (ctx) {
        free(ctx->cnProcBuf);    free(ctx->cnProcBufRes);
        free(ctx->bnProcBuf);    free(ctx->bnProcBufRes);
        free(ctx->llrRes);       free(ctx->llrProcBuf);
        free(ctx);
    }
    FARF(RUNTIME_HIGH, "ldpc_hexagon: session closed");
    return 0;
}

// =============================================================================
// Main RPC entry point — called by the qaic-generated skel dispatcher
// =============================================================================

int ldpc_hexagon_decode(remote_handle64 handle,
                        const uint8_t *params,   int paramsLen,
                        const uint8_t *llr,      int llrLen,
                        uint8_t       *llr_out,  int llr_outLen,
                        uint8_t       *meta,     int metaLen)
{
    if (paramsLen < (int)sizeof(ldpc_hex_params_t)) {
        FARF(ERROR, "ldpc_hexagon: params too short (%d < %d)",
             paramsLen, (int)sizeof(ldpc_hex_params_t));
        return -1;
    }

    ldpc_hex_params_t p;
    memcpy(&p, params, sizeof(p));

    FARF(RUNTIME_HIGH, "ldpc_hexagon: BG=%u Z=%u R=%u maxIter=%u numLLR_in=%d",
         p.BG, p.Z, p.R, p.numMaxIter, llrLen);

    // Build LUT on the DSP side (pure pointer assignments into static arrays)
    t_nrLDPC_dec_params dp;
    memset(&dp, 0, sizeof(dp));
    dp.BG         = p.BG;
    dp.Z          = p.Z;
    dp.R          = p.R;
    dp.numMaxIter = p.numMaxIter;

    t_nrLDPC_lut lut;
    uint32_t numLLR = nrLDPC_init(&dp, &lut);
    if (!numLLR || (int)numLLR > llrLen || (int)numLLR > llr_outLen) {
        FARF(ERROR, "ldpc_hexagon: bad numLLR=%u (llrLen=%d out=%d)",
             numLLR, llrLen, llr_outLen);
        return -1;
    }

    // diag=0: raw memcpy passthrough; diag=1: scatter/gather roundtrip; diag>=2: full BP
    if (p.diag == 0) {
        memcpy(llr_out, llr, numLLR);
        if (metaLen >= 4) { uint32_t z = 0; memcpy(meta, &z, 4); }
        return 0;
    }
    if (p.diag == 1) {
        int8_t *llrProcBuf = calloc(NR_LDPC_MAX_NUM_LLR, 1);
        int8_t *llrRes     = calloc(NR_LDPC_MAX_NUM_LLR, 1);
        if (llrProcBuf && llrRes) {
            nrLDPC_llr2llrProcBuf(&lut, (int8_t *)llr, llrProcBuf, p.Z, p.BG);
            memcpy(llrRes, llrProcBuf, NR_LDPC_MAX_NUM_LLR);
            nrLDPC_llrRes2llrOut(&lut, (int8_t *)llr_out, llrRes, p.Z, p.BG);
        }
        free(llrProcBuf);
        free(llrRes);
        if (metaLen >= 4) { uint32_t z = 0; memcpy(meta, &z, 4); }
        return 0;
    }
    // diag >= 2: fall through to full BP below

    ldpc_dsp_ctx_t *ctx = (ldpc_dsp_ctx_t *)(uintptr_t)handle;

    int32_t numIter = ldpc_scalar_core(
        (const int8_t *)llr, llr_out, numLLR,
        &lut, p.BG, p.Z, p.numMaxIter, ctx);

    FARF(RUNTIME_HIGH, "ldpc_hexagon: done in %d iter", numIter);

    if (metaLen >= 4) {
        uint32_t n = (uint32_t)numIter;
        memcpy(meta, &n, 4);
    }

    return 0;
}

#endif // !LDPC_SIM_STANDALONE

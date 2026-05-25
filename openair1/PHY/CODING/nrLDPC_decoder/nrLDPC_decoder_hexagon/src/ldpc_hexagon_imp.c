// SPDX-License-Identifier: LicenseRef-CSSL-1.0
//
// ldpc_hexagon_imp.c — Hexagon DSP LDPC decoder, scalar min-sum.
//
// Compiled with hexagon-clang -O3 and loaded into the cDSP protected domain
// via FastRPC.  All working buffers are heap-allocated (total ~540 KB) to
// stay within the DSP's limited thread stack.
//
// Algorithm:
//   Standard min-sum belief propagation:
//     CN step: for each check-node edge, output = sign(∏ others) × min(|others|)
//     BN step: posterior = channel_LLR + Σ(CN messages);
//              extrinsic = posterior − self CN message
//   Convergence: XOR of sign bits across all connected BNs at each CN = 0
//
// Performance path: replace cnProc_group() with HVX intrinsics
//   (Q6_V_vmux_QVV / Q6_V_vmin_VV etc.) after this scalar baseline is
//   validated.  The mPass and bnProc functions are already memory-bandwidth
//   bound and benefit less from HVX.
//
// To enable HVX when building the skel:
//   add_compile_options(-mhvx -mhvx-length=128B)

// Suppress OAI framework headers (time_meas.h etc.) inside nrLDPC_types.h
#define CODEGEN 1

// nrLDPC_mPass.h needs sizeofArray; provide it without pulling common/utils/utils.h
#ifndef sizeofArray
#define sizeofArray(a) ((int)(sizeof(a) / sizeof((a)[0])))
#endif

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "HAP_farf.h"

// Portable LDPC decoder headers — no SIMD, no OAI framework dependencies
#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_lut.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"

// qaic-generated dispatch header
#include "ldpc_hexagon.h"

// Wire parameter struct — must match ldpc_hexagon_arm.c
typedef struct __attribute__((packed)) {
    uint8_t  BG;
    uint8_t  R;
    uint8_t  numMaxIter;
    uint8_t  pad0;
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
// CN Processing — scalar min-sum
// =============================================================================
// For every (CN i, lifting symbol z) in a group:
//   Pass 1 over all BN positions k: track first/second minimum |v|, XOR signs.
//   Pass 2: write extrinsic message for each position j:
//     mag_j  = min1 if j != min1_k, else min2
//     sign_j = XOR of all signs excluding j  (= sgn_all ^ v_j)
//     result = sign_j < 0 ? -mag_j : mag_j

static void cnProc_group(
    const int8_t *cnProcBuf, int8_t *cnProcBufRes,
    uint32_t startAddr, int numBN, int numCN, uint16_t Z)
{
    uint32_t bitOff = (uint32_t)numCN * NR_LDPC_ZMAX;

    for (int i = 0; i < numCN; i++) {
        for (int z = 0; z < (int)Z; z++) {
            int samp = i * (int)Z + z;

            int8_t  sgn_all = 0;
            uint8_t min1 = 127, min2 = 127;
            int     min1_k = 0;

            for (int k = 0; k < numBN; k++) {
                int8_t v = cnProcBuf[startAddr + (uint32_t)k * bitOff + samp];
                sgn_all ^= v;
                uint8_t av = abs8(v);
                if (av <= min1) { min2 = min1; min1 = av; min1_k = k; }
                else if (av < min2) { min2 = av; }
            }

            for (int j = 0; j < numBN; j++) {
                int8_t v_j  = cnProcBuf[startAddr + (uint32_t)j * bitOff + samp];
                int8_t sgn_j = sgn_all ^ v_j;   // product of signs excluding j
                uint8_t mag  = (j == min1_k) ? min2 : min1;
                cnProcBufRes[startAddr + (uint32_t)j * bitOff + samp] =
                    (sgn_j & 0x80) ? -(int8_t)mag : (int8_t)mag;
            }
        }
    }
}

static void scalar_cnProc(t_nrLDPC_lut *p_lut,
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
// BN Processing — parity-check variant (scalar bnProcPc)
// =============================================================================
// For each BN b in group g (cnidx+1 connected CNs):
//   llrRes[b]         = sat8( llrProcBuf[b] + Σ_k bnProcBuf[k, b] )
//   bnProcBufRes[0,b] = llrProcBuf[b]   (stored for bnProc subtraction)
//
// Buffer layout in bnProcBuf/bnProcBufRes:
//   CN layer k: byte range [startBn + k * numBn * ZMAX  ..  + numBn * ZMAX - 1]
//   BN b within layer k: bytes [k*cnOff + b]  (b in 0..numBn*Z-1)
//
// Groups iterate by connectivity degree (cnidx = CNs-per-BN minus 1).
// lut_numBnInBnGroups has NR_LDPC_NUM_BN_GROUPS_BG1_R13=30 entries; many are
// zero for lower rates and for BG2 — the loop naturally skips them.

static void scalar_bnProcPc(t_nrLDPC_lut *p_lut,
                             const int8_t *bnProcBuf, int8_t *bnProcBufRes,
                             const int8_t *llrProcBuf, int8_t *llrRes,
                             uint16_t Z)
{
    const uint8_t  *numBn = p_lut->numBnInBnGroups;
    const uint32_t *sBn   = p_lut->startAddrBnGroups;
    const uint16_t *sLlr  = p_lut->startAddrBnGroupsLlr;

    // Group 0: BNs connected to exactly 1 CN (no accumulation loop needed)
    {
        uint32_t nb    = (uint32_t)numBn[0];
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[0];
        uint32_t sl    = sLlr[0];
        // Store channel LLR for later bnProc subtraction (extrinsic = channel for 1-CN BN)
        memcpy(&bnProcBufRes[sa], &llrProcBuf[sl], total);
        for (uint32_t b = 0; b < total; b++)
            llrRes[sl + b] = sat8_add(llrProcBuf[sl + b], bnProcBuf[sa + b]);
    }

    // Groups 1+: cnidx = number of connected CNs minus 1
    uint8_t idxBnGroup = 0;
    for (int cnidx = 1; cnidx < NR_LDPC_NUM_BN_GROUPS_BG1_R13; cnidx++) {
        if (numBn[cnidx] == 0) continue;
        idxBnGroup++;

        uint32_t nb    = (uint32_t)numBn[cnidx];
        uint32_t cnOff = nb * NR_LDPC_ZMAX;    // byte stride between CN layers
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[idxBnGroup];
        uint32_t sl    = sLlr[idxBnGroup];

        for (uint32_t b = 0; b < total; b++) {
            int16_t s = (int16_t)llrProcBuf[sl + b];
            for (int k = 0; k <= cnidx; k++)
                s += (int16_t)bnProcBuf[sa + (uint32_t)k * cnOff + b];
            llrRes[sl + b] = s >  127 ?  127 :
                             s < -128 ? -128 : (int8_t)s;
        }
    }
}

// =============================================================================
// BN Processing — extrinsic message (scalar bnProc)
// =============================================================================
// For each BN-to-CN edge (group g, CN layer k, BN b):
//   bnProcBufRes[k, b] = sat8( llrRes[b] - bnProcBuf[k, b] )
// Group 0 was already handled by bnProcPc (1-CN BNs: extrinsic = channel LLR).

static void scalar_bnProc(t_nrLDPC_lut *p_lut,
                           const int8_t *bnProcBuf, int8_t *bnProcBufRes,
                           const int8_t *llrRes, uint16_t Z)
{
    const uint8_t  *numBn = p_lut->numBnInBnGroups;
    const uint32_t *sBn   = p_lut->startAddrBnGroups;
    const uint16_t *sLlr  = p_lut->startAddrBnGroupsLlr;

    uint8_t idxBnGroup = 0;
    for (int cnidx = 1; cnidx < NR_LDPC_NUM_BN_GROUPS_BG1_R13; cnidx++) {
        if (numBn[cnidx] == 0) continue;
        idxBnGroup++;

        uint32_t nb    = (uint32_t)numBn[cnidx];
        uint32_t cnOff = nb * NR_LDPC_ZMAX;
        uint32_t total = nb * (uint32_t)Z;
        uint32_t sa    = sBn[idxBnGroup];
        uint32_t sl    = sLlr[idxBnGroup];

        for (int k = 0; k <= cnidx; k++) {
            uint32_t off = (uint32_t)k * cnOff;
            for (uint32_t b = 0; b < total; b++)
                bnProcBufRes[sa + off + b] =
                    sat8_sub(llrRes[sl + b], bnProcBuf[sa + off + b]);
        }
    }
}

// =============================================================================
// Main decoder core (heap-allocated buffers)
// =============================================================================
// Returns number of BP iterations completed.

static int32_t ldpc_scalar_core(
    const int8_t *p_llr, uint8_t *llr_out, uint32_t numLLR,
    t_nrLDPC_lut *p_lut, uint8_t BG, uint16_t Z, uint8_t numMaxIter)
{
    int8_t *cnProcBuf    = calloc(NR_LDPC_SIZE_CN_PROC_BUF, 1);
    int8_t *cnProcBufRes = calloc(NR_LDPC_SIZE_CN_PROC_BUF, 1);
    int8_t *bnProcBuf    = calloc(NR_LDPC_SIZE_BN_PROC_BUF, 1);
    int8_t *bnProcBufRes = calloc(NR_LDPC_SIZE_BN_PROC_BUF, 1);
    int8_t *llrRes       = calloc(NR_LDPC_MAX_NUM_LLR, 1);
    int8_t *llrProcBuf   = calloc(NR_LDPC_MAX_NUM_LLR, 1);
    int32_t numIter = 0;

    if (!cnProcBuf || !cnProcBufRes || !bnProcBuf ||
        !bnProcBufRes || !llrRes || !llrProcBuf) {
        FARF(ERROR, "ldpc_hexagon: working buffer allocation failed");
        goto cleanup;
    }

    // Scatter input LLRs into the two processing buffers (llrProcBuf and cnProcBuf).
    nrLDPC_llr2llrProcBuf(p_lut, (int8_t *)p_llr, llrProcBuf, Z, BG);
    if (BG == 1)
        nrLDPC_llr2CnProcBuf_BG1(p_lut, (int8_t *)p_llr, cnProcBuf, Z);
    else
        nrLDPC_llr2CnProcBuf_BG2(p_lut, (int8_t *)p_llr, cnProcBuf, Z);

    // First iteration without parity check: the two punctured BN columns are
    // not yet estimable from a single pass, so convergence cannot be declared.
    scalar_cnProc(p_lut, cnProcBuf, cnProcBufRes, Z, BG);

    if (BG == 1)
        nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
    else
        nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);

    scalar_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
    scalar_bnProc  (p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);

    if (BG == 1)
        nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
    else
        nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);

    // BP iteration loop with parity check
    int32_t pcRes = 1;
    while (numIter < (int32_t)numMaxIter && pcRes != 0) {
        scalar_cnProc(p_lut, cnProcBuf, cnProcBufRes, Z, BG);

        if (BG == 1)
            nrLDPC_cn2bnProcBuf_BG1(p_lut, cnProcBufRes, bnProcBuf, Z);
        else
            nrLDPC_cn2bnProcBuf_BG2(p_lut, cnProcBufRes, bnProcBuf, Z);

        scalar_bnProcPc(p_lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, Z);
        scalar_bnProc  (p_lut, bnProcBuf, bnProcBufRes, llrRes, Z);

        if (BG == 1)
            nrLDPC_bn2cnProcBuf_BG1(p_lut, bnProcBufRes, cnProcBuf, Z);
        else
            nrLDPC_bn2cnProcBuf_BG2(p_lut, bnProcBufRes, cnProcBuf, Z);

        pcRes = scalar_cnProcPc(cnProcBuf, p_lut, Z, BG);
        numIter++;
    }

    // Gather posterior LLRs from llrRes into the external output order.
    nrLDPC_llrRes2llrOut(p_lut, (int8_t *)llr_out, llrRes, Z, BG);

cleanup:
    free(cnProcBuf);    free(cnProcBufRes);
    free(bnProcBuf);    free(bnProcBufRes);
    free(llrRes);       free(llrProcBuf);
    return numIter;
}

// =============================================================================
// FastRPC session lifecycle (called by the skel dispatcher on open/close)
// =============================================================================

int ldpc_hexagon_open(const char *uri, remote_handle64 *handle)
{
    void *ctx = malloc(sizeof(int));
    if (!ctx) return -1;
    *handle = (remote_handle64)(uintptr_t)ctx;
    FARF(RUNTIME_HIGH, "ldpc_hexagon: session opened");
    return 0;
}

int ldpc_hexagon_close(remote_handle64 handle)
{
    free((void *)(uintptr_t)handle);
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

    int32_t numIter = ldpc_scalar_core(
        (const int8_t *)llr, llr_out, numLLR,
        &lut, p.BG, p.Z, p.numMaxIter);

    if (metaLen >= 4) {
        uint32_t n = (uint32_t)numIter;
        memcpy(meta, &n, 4);
    }

    FARF(RUNTIME_HIGH, "ldpc_hexagon: done in %d iter", numIter);
    return 0;
}

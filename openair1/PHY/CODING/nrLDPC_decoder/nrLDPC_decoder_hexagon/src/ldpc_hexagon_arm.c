// SPDX-License-Identifier: LicenseRef-CSSL-1.0
//
// ldpc_hexagon_arm.c — ARM-side LDPC decoder stub.
//
// This file is compiled (with the qaic-generated ldpc_hexagon_stub.c) into
// libldpc_hexagon.so, a drop-in replacement for libldpc.so that offloads
// belief-propagation iterations to the Hexagon Compute DSP via FastRPC.
//
// Exported symbols match the standard libldpc interface:
//   LDPCinit()     — open FastRPC session, initialise rpcmem
//   LDPCshutdown() — tear down session and rpcmem
//   LDPCdecoder()  — offload decode call to cDSP, post-process output
//
// The DSP always returns posterior LLRs (int8, numLLR bytes). The ARM stub
// performs the outMode conversion (bit-pack / int8 / LLR pass-through) so
// the DSP side remains format-agnostic.
//
// FastRPC / rpcmem docs: Hexagon SDK → ipc/fastrpc/

#include "ldpc_hexagon.h"   // qaic-generated from ldpc_hexagon.idl
#include "remote.h"         // DSPRPC_CONTROL_UNSIGNED_MODULE, remote_session_control
#include "rpcmem.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// OAI LDPC interface
#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_init.h"
#include "openair1/PHY/CODING/coding_defs.h"

// ---------------------------------------------------------------------------
// Bit conversion (scalar — avoids pulling in the SIMD bnProc headers here)
// ---------------------------------------------------------------------------

// Output matches nrLDPC_llr2bitPacked: bit (7-i) of byte b = sign(llr[b*8+i])
// i.e. MSB-first within each byte — bit 7 = first LLR in group, bit 0 = eighth.
static inline void arm_llr2bitPacked(uint8_t *out, const int8_t *llr, uint32_t numLLR)
{
    uint32_t nbytes = (numLLR + 7) >> 3;
    for (uint32_t b = 0; b < nbytes; b++) {
        uint8_t byte = 0;
        for (int i = 0; i < 8 && b * 8u + i < numLLR; i++)
            if (llr[b * 8 + i] < 0) byte |= (uint8_t)(1u << (7 - i));
        out[b] = byte;
    }
}

// Output: one int8_t per bit (0 or 1)
static inline void arm_llr2bit(uint8_t *out, const int8_t *llr, uint32_t numLLR)
{
    for (uint32_t i = 0; i < numLLR; i++)
        out[i] = llr[i] < 0 ? 1u : 0u;
}

// Wire parameter struct (8 bytes packed, must match ldpc_hexagon_imp.c)
typedef struct __attribute__((packed)) {
    uint8_t  BG;
    uint8_t  R;
    uint8_t  numMaxIter;
    uint8_t  diag;      // 0=memcpy passthrough, 1=scatter/gather roundtrip, 2+=normal BP
    uint16_t Z;
    uint16_t pad1;
} ldpc_hex_params_t;

static remote_handle64 dsp_hdl  = (remote_handle64)-1;
static int             rpcmem_up = 0;

// Pre-allocated ION-backed rpcmem buffers (allocated once in LDPCinit,
// freed in LDPCshutdown).  Reusing avoids per-call alloc/free cycles that
// exhaust the rpcmem pool after a few hundred codewords.
#define LDPC_HEX_MAX_LLR 27648  // NR_LDPC_MAX_NUM_LLR rounded up to 64-byte boundary
static ldpc_hex_params_t *g_rpc_params = NULL;
static int8_t            *g_rpc_llr    = NULL;
static int8_t            *g_rpc_llrout = NULL;
static uint8_t           *g_rpc_meta   = NULL;

// ---------------------------------------------------------------------------
// LDPCinit — open FastRPC session to cDSP and pre-allocate ION buffers.
// ---------------------------------------------------------------------------
int32_t LDPCinit(void)
{
    // ldpctest calls LDPCinit() before every codeword to reset the decoder.
    // Re-opening the FastRPC session and re-allocating ION buffers on every
    // call would exhaust the rpcmem pool within a few codewords.  Return
    // immediately if the session is already open.
    if (dsp_hdl != (remote_handle64)-1)
        return 0;

    // On Linux HLOS with fastrpc ≥ 1.0.4 the secure cDSP device is used by
    // default, which requires signed skels.  Enable unsigned-PD mode so that
    // our custom skel can be loaded without a hardware signature.
    // This must be called before the first remote_handle64_open() call.
    struct remote_rpc_control_unsigned_module um = { .domain = 3, .enable = 1 };
    int rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
    if (rc)
        fprintf(stderr, "ldpc_hexagon: remote_session_control(UNSIGNED_MODULE) returned %d (continuing)\n", rc);

    // rpcmem_init() opens /dev/dma_heap/system so libcdsprpc.so can DMA-transfer
    // fastrpc_shell_unsigned_3 (~1.26 MB) during remote_handle64_open() below.
    // Must be called before ldpc_hexagon_open().
    // NOTE: rpcmem.a is NOT linked (see CMakeLists.txt); this calls the device's
    // real rpcmem_init from libcdsprpc.so, not the SDK's deprecated stub.
    rpcmem_init();
    rpcmem_up = 1;

    // fastrpc ≥ 1.0.4 on Linux HLOS requires "&_dom=cdsp" in the URI to identify
    // the target domain.  The qaic-generated ldpc_hexagon_URI does not include it,
    // so we append it here.
    const char *uri = ldpc_hexagon_URI "&_dom=cdsp";

    int err = ldpc_hexagon_open(uri, &dsp_hdl);
    if (err) {
        fprintf(stderr, "ldpc_hexagon_open failed: %d\n", err);
        rpcmem_deinit();
        rpcmem_up = 0;
        return err;
    }

    // Pre-allocate ION buffers for the maximum possible LLR count.
    // This avoids per-codeword alloc/free cycles that exhaust the rpcmem pool.
    g_rpc_params = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                 sizeof(ldpc_hex_params_t));
    g_rpc_llr    = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                 LDPC_HEX_MAX_LLR);
    g_rpc_llrout = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                 LDPC_HEX_MAX_LLR);
    g_rpc_meta   = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 8);

    if (!g_rpc_params || !g_rpc_llr || !g_rpc_llrout || !g_rpc_meta) {
        fprintf(stderr, "ldpc_hexagon: rpcmem_alloc for pre-allocated buffers failed\n");
        ldpc_hexagon_close(dsp_hdl);
        dsp_hdl = (remote_handle64)-1;
        if (g_rpc_params) { rpcmem_free(g_rpc_params); g_rpc_params = NULL; }
        if (g_rpc_llr)    { rpcmem_free(g_rpc_llr);    g_rpc_llr    = NULL; }
        if (g_rpc_llrout) { rpcmem_free(g_rpc_llrout); g_rpc_llrout = NULL; }
        if (g_rpc_meta)   { rpcmem_free(g_rpc_meta);   g_rpc_meta   = NULL; }
        rpcmem_deinit();
        rpcmem_up = 0;
        return -1;
    }
    fprintf(stderr, "ldpc_hexagon: pre-allocated %d-byte ION buffers (params+llr+llrout+meta)\n",
            (int)(sizeof(ldpc_hex_params_t) + 2 * LDPC_HEX_MAX_LLR + 8));
    return 0;
}

// ---------------------------------------------------------------------------
// LDPCshutdown — close FastRPC session.
// ---------------------------------------------------------------------------
int32_t LDPCshutdown(void)
{
    if (dsp_hdl != (remote_handle64)-1) {
        ldpc_hexagon_close(dsp_hdl);
        dsp_hdl = (remote_handle64)-1;
    }
    if (g_rpc_params) { rpcmem_free(g_rpc_params); g_rpc_params = NULL; }
    if (g_rpc_llr)    { rpcmem_free(g_rpc_llr);    g_rpc_llr    = NULL; }
    if (g_rpc_llrout) { rpcmem_free(g_rpc_llrout); g_rpc_llrout = NULL; }
    if (g_rpc_meta)   { rpcmem_free(g_rpc_meta);   g_rpc_meta   = NULL; }
    if (rpcmem_up) {
        rpcmem_deinit();
        rpcmem_up = 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// LDPCdecoder — offload one code-block decode to the Hexagon cDSP.
//
// p_decParams->check_crc is a function pointer and cannot cross the FastRPC
// boundary. When set, parity is checked via iter count instead (the ARM side
// repeats CRC after the DSP returns; this path is future work).
// ---------------------------------------------------------------------------
int32_t LDPCdecoder(t_nrLDPC_dec_params *p_decParams,
                    int8_t  *p_llr,
                    uint8_t *p_out,
                    t_nrLDPC_time_stats *p_profiler,
                    decode_abort_t *ab)
{
    // Compute numLLR using the standard init path (also validates BG/Z/R).
    t_nrLDPC_lut lut;
    uint32_t numLLR = nrLDPC_init(p_decParams, &lut);
    if (!numLLR) {
        fprintf(stderr, "ldpc_hexagon: nrLDPC_init returned 0\n");
        return -1;
    }

    // Use pre-allocated ION-backed rpcmem buffers (avoids per-call alloc/free).
    ldpc_hex_params_t *rpc_params = g_rpc_params;
    int8_t  *rpc_llr    = g_rpc_llr;
    int8_t  *rpc_llrout = g_rpc_llrout;
    uint8_t *rpc_meta   = g_rpc_meta;

    int32_t ret = -1;
    if (!rpc_params || !rpc_llr || !rpc_llrout || !rpc_meta) {
        fprintf(stderr, "ldpc_hexagon: pre-allocated buffers not ready (numLLR=%u)\n", numLLR);
        return -1;
    }
    if (numLLR > LDPC_HEX_MAX_LLR) {
        fprintf(stderr, "ldpc_hexagon: numLLR=%u exceeds pre-alloc limit %d\n",
                numLLR, LDPC_HEX_MAX_LLR);
        return -1;
    }

    rpc_params->BG         = p_decParams->BG;
    rpc_params->R          = p_decParams->R;
    rpc_params->numMaxIter = p_decParams->numMaxIter;
    rpc_params->diag       = 2;
    rpc_params->Z          = p_decParams->Z;
    rpc_params->pad1       = 0;
    memcpy(rpc_llr, p_llr, numLLR);

    // On first call only: run diag=0 (passthrough) and diag=1 (scatter/gather)
    // to isolate phase costs from the ARM side using wall-clock time.
    // DSP hardware cycle counters are inaccessible from user mode on this device.
    static int arm_perf_done = 0;
    if (!arm_perf_done) {
        arm_perf_done = 1;
        struct timespec ta, tb;
        uint8_t orig_diag = rpc_params->diag;

        rpc_params->diag = 0;  // raw memcpy passthrough
        clock_gettime(CLOCK_MONOTONIC, &ta);
        ldpc_hexagon_decode(dsp_hdl,
            (uint8_t *)rpc_params, sizeof(*rpc_params),
            (uint8_t *)rpc_llr, (int)numLLR,
            (uint8_t *)rpc_llrout, (int)numLLR,
            rpc_meta, 8);
        clock_gettime(CLOCK_MONOTONIC, &tb);
        uint64_t rpc_us = ((uint64_t)(tb.tv_sec - ta.tv_sec) * 1000000ULL +
                           (tb.tv_nsec - ta.tv_nsec) / 1000);
        // Verify diag=0: output must equal input exactly.
        {
            uint32_t mismatches = 0;
            for (uint32_t i = 0; i < numLLR; i++)
                if (rpc_llrout[i] != rpc_llr[i]) mismatches++;
            if (mismatches)
                fprintf(stderr, "  diag=0 MISMATCH: %u/%u bytes differ (data transfer broken!)\n",
                        mismatches, numLLR);
            else
                fprintf(stderr, "  diag=0 OK: output == input (%u bytes match)\n", numLLR);
        }

        rpc_params->diag = 1;  // scatter + gather roundtrip
        clock_gettime(CLOCK_MONOTONIC, &ta);
        ldpc_hexagon_decode(dsp_hdl,
            (uint8_t *)rpc_params, sizeof(*rpc_params),
            (uint8_t *)rpc_llr, (int)numLLR,
            (uint8_t *)rpc_llrout, (int)numLLR,
            rpc_meta, 8);
        clock_gettime(CLOCK_MONOTONIC, &tb);
        uint64_t sg_us = ((uint64_t)(tb.tv_sec - ta.tv_sec) * 1000000ULL +
                          (tb.tv_nsec - ta.tv_nsec) / 1000);
        // Verify diag=1: scatter+gather should be identity transformation.
        {
            uint32_t mismatches = 0;
            for (uint32_t i = 0; i < numLLR; i++)
                if (rpc_llrout[i] != rpc_llr[i]) mismatches++;
            if (mismatches)
                fprintf(stderr, "  diag=1 MISMATCH: %u/%u bytes differ (scatter/gather not identity!)\n",
                        mismatches, numLLR);
            else
                fprintf(stderr, "  diag=1 OK: scatter+gather is identity\n");
        }

        // diag=4: llr2CnProcBuf + LUT header.
        // llr_out[0..63] = 16 × uint32_t diagnostic header from DSP.
        // llr_out[64..]  = cnProcBuf[0..numLLR-65].
        rpc_params->diag = 4;
        ldpc_hexagon_decode(dsp_hdl,
            (uint8_t *)rpc_params, sizeof(*rpc_params),
            (uint8_t *)rpc_llr, (int)numLLR,
            (uint8_t *)rpc_llrout, (int)numLLR,
            rpc_meta, 8);
        {
            uint32_t hdr[16];
            memcpy(hdr, rpc_llrout, 64);
            fprintf(stderr, "  diag=4 LUT header from DSP:\n");
            fprintf(stderr, "    circShift[0].d=0x%08x  .d[0]=%u  dim1=%u  dim2=%u\n",
                    hdr[0], hdr[1], hdr[2], hdr[3]);
            fprintf(stderr, "    numCnInCn[0]=%u  startAddrCn[0]=%u\n",
                    hdr[4], hdr[5]);
            fprintf(stderr, "    numBnInBn[0]=%u  startAddrBn[0]=%u\n",
                    hdr[6], hdr[7]);
            fprintf(stderr, "    cnProcBuf[0] before scatter=%u  after scatter=%u %u\n",
                    hdr[8], hdr[12], hdr[13]);
            fprintf(stderr, "    cnProcBuf[384]=%u  cnProcBuf[768]=%u\n",
                    hdr[14], hdr[15]);
            // Count pos/neg in the body (bytes 64..)
            int llr_pos = 0, llr_neg = 0, llr_zer = 0;
            int cn_pos  = 0, cn_neg  = 0, cn_zer  = 0;
            for (uint32_t i = 0; i < numLLR; i++) {
                int8_t v = rpc_llr[i];
                if (v > 0) llr_pos++; else if (v < 0) llr_neg++; else llr_zer++;
            }
            for (uint32_t i = 64; i < numLLR; i++) {
                int8_t v = rpc_llrout[i];
                if (v > 0) cn_pos++; else if (v < 0) cn_neg++; else cn_zer++;
            }
            fprintf(stderr, "    llr(pos=%d neg=%d zer=%d) cnBuf[64..](pos=%d neg=%d zer=%d)\n",
                    llr_pos, llr_neg, llr_zer, cn_pos, cn_neg, cn_zer);
            // expected: llr[332] for cnBuf[0] with cshift=332
            fprintf(stderr, "    llr[332]=%d (expected at cnBuf[0])\n",
                    (int)(int8_t)rpc_llr[332]);
        }

        // diag=3: full BP but return raw llrRes (before gather) to check if posteriors are sane.
        rpc_params->diag = 3;
        ldpc_hexagon_decode(dsp_hdl,
            (uint8_t *)rpc_params, sizeof(*rpc_params),
            (uint8_t *)rpc_llr, (int)numLLR,
            (uint8_t *)rpc_llrout, (int)numLLR,
            rpc_meta, 8);
        {
            int pos = 0, neg = 0, zer = 0;
            for (uint32_t i = 0; i < numLLR; i++) {
                int8_t v = rpc_llrout[i];
                if (v > 0) pos++; else if (v < 0) neg++; else zer++;
            }
            fprintf(stderr, "  diag=3 raw llrRes: pos=%d neg=%d zero=%d [0..7]: %d %d %d %d %d %d %d %d\n",
                    pos, neg, zer,
                    (int)(int8_t)rpc_llrout[0], (int)(int8_t)rpc_llrout[1],
                    (int)(int8_t)rpc_llrout[2], (int)(int8_t)rpc_llrout[3],
                    (int)(int8_t)rpc_llrout[4], (int)(int8_t)rpc_llrout[5],
                    (int)(int8_t)rpc_llrout[6], (int)(int8_t)rpc_llrout[7]);
        }

        fprintf(stderr, "DSP phase breakdown (ARM wall-clock):\n");
        fprintf(stderr, "  RPC overhead (diag=0):  %6llu us\n", (unsigned long long)rpc_us);
        fprintf(stderr, "  scatter+gather (diag=1):%6llu us  compute=%llu us\n",
                (unsigned long long)sg_us,
                (unsigned long long)(sg_us > rpc_us ? sg_us - rpc_us : 0));
        fprintf(stderr, "  full decode (diag=2):    (see ldpctest Decoding time)\n");

        rpc_params->diag = orig_diag;
    }

    int err = ldpc_hexagon_decode(dsp_hdl,
                                  (uint8_t *)rpc_params, sizeof(*rpc_params),
                                  (uint8_t *)rpc_llr,    (int)numLLR,
                                  (uint8_t *)rpc_llrout, (int)numLLR,
                                  rpc_meta, 8);
    if (err) {
        fprintf(stderr, "ldpc_hexagon_decode RPC call failed: %d\n", err);
        return -1;
    }

    uint32_t numIter;
    memcpy(&numIter, rpc_meta, 4);
    ret = (int32_t)numIter;

    // Print DSP clock on first decode (bytes 4-7 of meta, set during ldpc_hexagon_open).
    static int dsp_clk_printed = 0;
    if (!dsp_clk_printed) {
        dsp_clk_printed = 1;
        uint32_t clk_hz = 0;
        memcpy(&clk_hz, rpc_meta + 4, 4);
        fprintf(stderr, "ldpc_hexagon: DSP core clock = %u Hz (%u MHz)\n",
                clk_hz, clk_hz / 1000000u);
    }

    // One-shot diagnostic: count pos/neg in first decode to check DSP output quality.
    static int arm_diag_done = 0;
    if (!arm_diag_done) {
        arm_diag_done = 1;
        int pos = 0, neg = 0, zer = 0;
        for (uint32_t i = 0; i < numLLR; i++) {
            int8_t v = rpc_llrout[i];
            if (v > 0) pos++; else if (v < 0) neg++; else zer++;
        }
        fprintf(stderr, "ldpc_hexagon diag: numIter=%u numLLR=%u pos=%d neg=%d zero=%d llrout[0..7]: %d %d %d %d %d %d %d %d\n",
                numIter, numLLR, pos, neg, zer,
                (int)rpc_llrout[0], (int)rpc_llrout[1], (int)rpc_llrout[2], (int)rpc_llrout[3],
                (int)rpc_llrout[4], (int)rpc_llrout[5], (int)rpc_llrout[6], (int)rpc_llrout[7]);
    }

    if (ret >= (int32_t)p_decParams->numMaxIter)
        set_abort(ab, true);

    // Convert posterior LLRs to the requested output format.
    switch (p_decParams->outMode) {
    case nrLDPC_outMode_BIT:
        arm_llr2bitPacked(p_out, rpc_llrout, numLLR);
        break;
    case nrLDPC_outMode_BITINT8:
        arm_llr2bit(p_out, rpc_llrout, numLLR);
        break;
    case nrLDPC_outMode_LLRINT8:
    default:
        memcpy(p_out, rpc_llrout, numLLR);
        break;
    }

    return ret;
}

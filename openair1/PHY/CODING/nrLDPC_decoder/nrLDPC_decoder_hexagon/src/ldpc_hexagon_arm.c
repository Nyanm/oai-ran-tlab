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

// Output: byte b bit i = 1 iff llr[b*8+i] < 0  (LSB = first LLR in group of 8)
static inline void arm_llr2bitPacked(uint8_t *out, const int8_t *llr, uint32_t numLLR)
{
    uint32_t nbytes = (numLLR + 7) >> 3;
    for (uint32_t b = 0; b < nbytes; b++) {
        uint8_t byte = 0;
        for (int i = 0; i < 8 && b * 8u + i < numLLR; i++)
            if (llr[b * 8 + i] < 0) byte |= (uint8_t)(1u << i);
        out[b] = byte;
    }
}

// Output: one int8_t per bit (0 or 1)
static inline void arm_llr2bit(uint8_t *out, const int8_t *llr, uint32_t numLLR)
{
    for (uint32_t i = 0; i < numLLR; i++)
        out[i] = llr[i] < 0 ? 1u : 0u;
}

#define LDPC_HEX_DOMAIN  3   // cDSP domain index on SA9000P

// Wire parameter struct (8 bytes packed, must match ldpc_hexagon_imp.c)
typedef struct __attribute__((packed)) {
    uint8_t  BG;
    uint8_t  R;
    uint8_t  numMaxIter;
    uint8_t  pad0;
    uint16_t Z;
    uint16_t pad1;
} ldpc_hex_params_t;

static remote_handle64 dsp_hdl  = (remote_handle64)-1;
static int             rpcmem_up = 0;

// ---------------------------------------------------------------------------
// LDPCinit — open FastRPC session to cDSP.
// ---------------------------------------------------------------------------
int32_t LDPCinit(void)
{
    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=%d", ldpc_hexagon_URI, LDPC_HEX_DOMAIN);

    int err = ldpc_hexagon_open(uri, &dsp_hdl);
    if (err) {
        fprintf(stderr, "ldpc_hexagon_open failed: %d\n", err);
        return err;
    }
    rpcmem_init();
    rpcmem_up = 1;
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

    // Allocate ION-backed rpcmem for zero-copy DMA to DSP.
    ldpc_hex_params_t *rpc_params =
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                     sizeof(ldpc_hex_params_t));
    int8_t  *rpc_llr     =
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, numLLR);
    int8_t  *rpc_llrout  =
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, numLLR);
    uint8_t *rpc_meta    =
        rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 4);

    int32_t ret = -1;
    if (!rpc_params || !rpc_llr || !rpc_llrout || !rpc_meta) {
        fprintf(stderr, "ldpc_hexagon: rpcmem_alloc failed (numLLR=%u)\n", numLLR);
        goto cleanup;
    }

    rpc_params->BG         = p_decParams->BG;
    rpc_params->R          = p_decParams->R;
    rpc_params->numMaxIter = p_decParams->numMaxIter;
    rpc_params->pad0       = 0;
    rpc_params->Z          = p_decParams->Z;
    rpc_params->pad1       = 0;
    memcpy(rpc_llr, p_llr, numLLR);

    int err = ldpc_hexagon_decode(dsp_hdl,
                                  (uint8_t *)rpc_params, sizeof(*rpc_params),
                                  (uint8_t *)rpc_llr,    (int)numLLR,
                                  (uint8_t *)rpc_llrout, (int)numLLR,
                                  rpc_meta, 4);
    if (err) {
        fprintf(stderr, "ldpc_hexagon_decode RPC call failed: %d\n", err);
        goto cleanup;
    }

    // Extract iteration count from metadata.
    uint32_t numIter;
    memcpy(&numIter, rpc_meta, 4);
    ret = (int32_t)numIter;

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

cleanup:
    if (rpc_params) rpcmem_free(rpc_params);
    if (rpc_llr)    rpcmem_free(rpc_llr);
    if (rpc_llrout) rpcmem_free(rpc_llrout);
    if (rpc_meta)   rpcmem_free(rpc_meta);
    return ret;
}

// SPDX-License-Identifier: LicenseRef-CSSL-1.0
//
// ldpc_sim_main.c — Standalone Hexagon simulator profiling harness.
//
// Runs the LDPC BP phases (cnProc, permutations, bnProc) in the Hexagon
// simulator and reports per-phase cycle counts.  Hardware cycle counters
// work in the simulator even though they return 0 on the device in
// unsigned-PD mode.
//
// Build:
//   cd sim/
//   make          (uses Makefile in this directory)
//
// Run:
//   make sim      (invokes hexagon-sim and prints phase breakdown)

// Tell ldpc_hexagon_imp.c to skip FastRPC / HAP-framework sections.
#define LDPC_SIM_STANDALONE 1

// Pull in the full DSP implementation — gives access to all static functions.
#include "../src/ldpc_hexagon_imp.c"

// Simulator cycle counter (in Hexagon toolchain standard headers, linked via -lhexagon).
#include <hexagon_sim_timer.h>

#define N_WARMUP 2    // un-timed warm-up iterations (prime caches)
#define N_ITER   8    // timed iterations to average

// Weak LLRs that don't converge easily — ensures all N_ITER run.
static void fill_llr(int8_t *buf, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        buf[i] = (int8_t)((i & 3) ? 4 : -4);
}

int main(void)
{
    t_nrLDPC_dec_params dp;
    memset(&dp, 0, sizeof(dp));
    dp.BG = 1; dp.Z = 384; dp.R = 13; dp.numMaxIter = N_WARMUP + N_ITER + 4;

    t_nrLDPC_lut lut;
    uint32_t numLLR = nrLDPC_init(&dp, &lut);
    if (!numLLR) { printf("nrLDPC_init failed\n"); return 1; }

    ldpc_dsp_ctx_t ctx;
    ctx.cnProcBuf    = calloc(NR_LDPC_SIZE_CN_PROC_BUF, 1);
    ctx.cnProcBufRes = calloc(NR_LDPC_SIZE_CN_PROC_BUF, 1);
    ctx.bnProcBuf    = calloc(NR_LDPC_SIZE_BN_PROC_BUF, 1);
    ctx.bnProcBufRes = calloc(NR_LDPC_SIZE_BN_PROC_BUF, 1);
    ctx.llrRes       = calloc(NR_LDPC_MAX_NUM_LLR, 1);
    ctx.llrProcBuf   = calloc(NR_LDPC_MAX_NUM_LLR, 1);
    int8_t  *llr    = malloc(numLLR);
    uint8_t *llrout = malloc(numLLR);

    if (!ctx.cnProcBuf || !ctx.cnProcBufRes || !ctx.bnProcBuf ||
        !ctx.bnProcBufRes || !ctx.llrRes || !ctx.llrProcBuf || !llr || !llrout) {
        printf("alloc failed\n"); return 1;
    }
    fill_llr(llr, numLLR);

    int8_t *cnProcBuf    = ctx.cnProcBuf;
    int8_t *cnProcBufRes = ctx.cnProcBufRes;
    int8_t *bnProcBuf    = ctx.bnProcBuf;
    int8_t *bnProcBufRes = ctx.bnProcBufRes;
    int8_t *llrRes       = ctx.llrRes;
    int8_t *llrProcBuf   = ctx.llrProcBuf;

    // Initial scatter (one-time, not part of per-iteration timing)
    memset(cnProcBuf,    0, NR_LDPC_SIZE_CN_PROC_BUF);
    memset(cnProcBufRes, 0, NR_LDPC_SIZE_CN_PROC_BUF);
    memset(bnProcBuf,    0, NR_LDPC_SIZE_BN_PROC_BUF);
    memset(bnProcBufRes, 0, NR_LDPC_SIZE_BN_PROC_BUF);
    nrLDPC_llr2llrProcBuf(&lut, llr, llrProcBuf, 384, 1);
    nrLDPC_llr2CnProcBuf_BG1(&lut, llr, cnProcBuf, 384);

    // Warm-up (prime caches, not timed)
    for (int w = 0; w < N_WARMUP; w++) {
        cnProc(&lut, cnProcBuf, cnProcBufRes, 384, 1);
        nrLDPC_cn2bnProcBuf_BG1(&lut, cnProcBufRes, bnProcBuf, 384);
        hvx_bnProcPc(&lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, 384);
        hvx_bnProc(&lut, bnProcBuf, bnProcBufRes, llrRes, 384);
        nrLDPC_bn2cnProcBuf_BG1(&lut, bnProcBufRes, cnProcBuf, 384);
    }

    // Timed iterations
    uint64_t cyc_cnProc = 0, cyc_cn2bn = 0, cyc_bnProcPc = 0;
    uint64_t cyc_bnProc = 0, cyc_bn2cn = 0, cyc_pc = 0;
    uint64_t t0, t1;

    for (int it = 0; it < N_ITER; it++) {
        t0 = hexagon_sim_read_pcycles();
        cnProc(&lut, cnProcBuf, cnProcBufRes, 384, 1);
        t1 = hexagon_sim_read_pcycles();
        cyc_cnProc += t1 - t0;

        t0 = hexagon_sim_read_pcycles();
        nrLDPC_cn2bnProcBuf_BG1(&lut, cnProcBufRes, bnProcBuf, 384);
        t1 = hexagon_sim_read_pcycles();
        cyc_cn2bn += t1 - t0;

        t0 = hexagon_sim_read_pcycles();
        hvx_bnProcPc(&lut, bnProcBuf, bnProcBufRes, llrProcBuf, llrRes, 384);
        t1 = hexagon_sim_read_pcycles();
        cyc_bnProcPc += t1 - t0;

        t0 = hexagon_sim_read_pcycles();
        hvx_bnProc(&lut, bnProcBuf, bnProcBufRes, llrRes, 384);
        t1 = hexagon_sim_read_pcycles();
        cyc_bnProc += t1 - t0;

        t0 = hexagon_sim_read_pcycles();
        nrLDPC_bn2cnProcBuf_BG1(&lut, bnProcBufRes, cnProcBuf, 384);
        t1 = hexagon_sim_read_pcycles();
        cyc_bn2cn += t1 - t0;

        t0 = hexagon_sim_read_pcycles();
        scalar_cnProcPc(cnProcBuf, &lut, 384, 1);
        t1 = hexagon_sim_read_pcycles();
        cyc_pc += t1 - t0;
    }

    uint64_t total = cyc_cnProc + cyc_cn2bn + cyc_bnProcPc + cyc_bnProc + cyc_bn2cn + cyc_pc;
    uint64_t n = (uint64_t)N_ITER;

    printf("\n=== LDPC per-phase profile: BG1 Z=384, avg over %d iters ===\n", N_ITER);
    printf("  cnProc    (HVX min-sum)  : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_cnProc/n),   100.0*cyc_cnProc/total);
    printf("  cn2bn     (permutation)  : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_cn2bn/n),    100.0*cyc_cn2bn/total);
    printf("  bnProcPc  (BN+posterior) : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_bnProcPc/n), 100.0*cyc_bnProcPc/total);
    printf("  bnProc    (BN extrinsic) : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_bnProc/n),   100.0*cyc_bnProc/total);
    printf("  bn2cn     (permutation)  : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_bn2cn/n),    100.0*cyc_bn2cn/total);
    printf("  cnProcPc  (parity check) : %8llu  (%3.0f%%)\n",
           (unsigned long long)(cyc_pc/n),       100.0*cyc_pc/total);
    printf("  per-iteration total       : %8llu  (simulator cycles)\n",
           (unsigned long long)(total/n));
    return 0;
}

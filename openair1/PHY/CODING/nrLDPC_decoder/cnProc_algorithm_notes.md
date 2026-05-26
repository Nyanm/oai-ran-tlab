# cnProc Min-Sum: Algorithm Notes

## Two Approaches

### OAI reference (`nrLDPC_cnProc.h`) — LUT exclude-self

For each output BN position `j`, directly load only the **other** (numBN−1) BN positions
using pre-computed LUT offsets. Min and sign are accumulated in a single inner loop and
the result is written immediately. No second loop needed because self is never loaded.

Works correctly for all group degrees but reads `numBN × (numBN−1)` values per CN×chunk.

### Two-pass min1/min2 (Hexagon DSP, `ldpc_hexagon_imp.c`)

- **Pass 1** over all k: find `min1`, `min2`, `sgn_all`
- **Pass 2** over all k: select magnitude (`|vk|==min1` → use `min2`, else `min1`) and
  remove self sign (`sgn_j = sgn_all XOR vk`)

Pass 2 cannot be merged into pass 1 because the magnitude decision requires the fully
finalized `min1`. Both passes must re-read `vk`. Total reads: `2 × numBN` per CN×chunk.

**Tie approximation:** when multiple BNs share `min1`, all get `min2` as their extrinsic
magnitude. This is a standard approximation with negligible effect on BER convergence.

## Memory Traffic

| numBN | OAI LUT reads | Two-pass reads | Ratio |
|------:|:-------------:|:--------------:|------:|
|     3 |     6         |     6          |  1.0× |
|     4 |    12         |     8          |  1.5× |
|    10 |    90         |    20          |  4.5× |
|    19 |   342         |    38          |  9.0× |

## Why It Matters for the DSP

DSP L1 cache is 32 KB. The full `cnProcBuf` working set is ~530 KB (BG1, Z=384).
Every extra read is an L1 miss. Two-pass is the correct choice for any cache-constrained
environment, especially for BG1 CNG19 (degree-19 group, most numerous).

## Potential CPU Optimization (Future Work)

The OAI AVX2/NEON reference uses the LUT approach for **all** degrees, including CNG19.
A generic two-pass implementation for the CPU decoder could reduce memory traffic 9× for
BG1 CNG19 and may improve throughput at large lifting factors (Z=384) on all CPU targets.

To investigate:
1. Implement `cnProc_group_2pass(numBN, numCN, Z)` with AVX2/NEON intrinsics
2. Call it from the dispatcher for degrees ≥ 4 (degree-3 is a wash)
3. Benchmark on RK3588, GH200, x86 AVX2 targets

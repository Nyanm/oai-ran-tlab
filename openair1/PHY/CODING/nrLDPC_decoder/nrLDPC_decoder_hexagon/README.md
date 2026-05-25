# LDPC Decoder — Hexagon DSP Offload

Offloads 5G NR LDPC belief-propagation decoding from the A78 application cores
to the Qualcomm Hexagon Compute DSP (cDSP) on the DragonWing SA9000P via
FastRPC.

## Directory layout

```
nrLDPC_decoder_hexagon/
├── inc/
│   └── ldpc_hexagon.idl      FastRPC IDL (qaic → stub + skel + shared header)
└── src/
    ├── ldpc_hexagon_arm.c    ARM stub  → libldpc_hexagon.so
    └── ldpc_hexagon_imp.c    DSP skel  → libldpc_hexagon_skel.so
```

## Interface

The IDL defines a single RPC function:

```
long decode(in   sequence<uint8> params,    // 8-byte packed t_nrLDPC_dec_params
            in   sequence<uint8> llr,        // input LLRs (numLLR × int8)
            rout sequence<uint8> llr_out,    // posterior LLRs (numLLR × int8)
            rout sequence<uint8> meta);      // 4-byte LE uint32 = iterations completed
```

`params` wire format (8 bytes):

| Offset | Field      | Type    | Notes                                  |
|--------|------------|---------|----------------------------------------|
| 0      | BG         | uint8   | Base graph 1 or 2                      |
| 1      | R          | uint8   | Rate: 13, 23, 89 (BG1); 15, 13, 23 (BG2) |
| 2      | numMaxIter | uint8   | Maximum BP iterations                  |
| 3      | —          | uint8   | Reserved                               |
| 4–5    | Z          | uint16  | Lifting size (2–384), little-endian    |
| 6–7    | —          | uint16  | Reserved                               |

The DSP always returns posterior LLRs in `llr_out`. The ARM stub converts to the
requested `outMode` (bit-packed / int8 / LLR pass-through) locally, and the
`check_crc` function pointer never crosses the FastRPC boundary.

## ARM stub (`ldpc_hexagon_arm.c` + `ldpc_encoder_optim8segmulti.c`)

`libldpc_hexagon.so` exports the same four symbols as `libldpc.so` so it is a
drop-in replacement for testing with `ldpctest` and the segmentation layer:

| Symbol         | Source                               | Description                       |
|----------------|--------------------------------------|-----------------------------------|
| `LDPCinit()`   | ldpc_hexagon_arm.c                   | Opens FastRPC session, rpcmem init |
| `LDPCshutdown()` | ldpc_hexagon_arm.c                 | Closes session, releases rpcmem   |
| `LDPCdecoder()` | ldpc_hexagon_arm.c                  | Offloads decode to cDSP            |
| `LDPCencoder()` | ldpc_encoder_optim8segmulti.c       | Runs on ARM (not offloaded)        |

The encoder is included unchanged from the standard `libldpc.so` build and
links against `ldpc_segment` and `ldpc_gen_HEADERS` exactly as it does there.
Offloading the encoder to the DSP may be evaluated separately.

`LDPCdecoder` flow:
1. Calls `nrLDPC_init` to derive `numLLR` from BG/Z/R.
2. Allocates four ION-backed rpcmem buffers (params, LLR in, LLR out, meta).
3. Copies input LLRs, calls `ldpc_hexagon_decode` via FastRPC.
4. Reads `numIter` from meta; sets abort flag if `numIter >= numMaxIter`.
5. Converts `llr_out` to the requested output format with a scalar loop.

## DSP implementation (`ldpc_hexagon_imp.c`)

Scalar min-sum belief propagation using the existing portable OAI headers:

| Header reused        | Role                                             |
|----------------------|--------------------------------------------------|
| `nrLDPC_types.h`     | Struct definitions (`t_nrLDPC_lut`, etc.)        |
| `nrLDPCdecoder_defs.h` | Constants (ZMAX, buffer sizes, group counts)   |
| `nrLDPC_lut.h`       | All static LUT arrays (circShift, BN addresses…) |
| `nrLDPC_init.h`      | LUT pointer setup (`nrLDPC_init`)                |
| `nrLDPC_mPass.h`     | Circular memcpy (`nrLDPC_llr2CnProcBuf`, etc.)   |

`#define CODEGEN 1` suppresses the `time_meas.h` dependency inside
`nrLDPC_types.h`. `sizeofArray` is defined locally before including
`nrLDPC_mPass.h` to avoid pulling in `common/utils/utils.h`.

### Algorithm

Standard flooding-schedule min-sum BP:

**Initialisation**
- `nrLDPC_llr2llrProcBuf` — scatter channel LLRs into LLR processing buffer
- `nrLDPC_llr2CnProcBuf_BG{1,2}` — scatter channel LLRs into CN processing buffer

**Each iteration**
1. `scalar_cnProc` — CN step (min-sum, scalar)
2. `nrLDPC_cn2bnProcBuf_BG{1,2}` — CN results → BN processing buffer (circular memcpy)
3. `scalar_bnProcPc` — BN posterior sum: `llrRes = sat8(llrProcBuf + Σ_k bnProcBuf[k])`
4. `scalar_bnProc` — BN extrinsic: `bnProcBufRes[k] = sat8(llrRes − bnProcBuf[k])`
5. `nrLDPC_bn2cnProcBuf_BG{1,2}` — extrinsics → CN processing buffer
6. `scalar_cnProcPc` — parity check: XOR of sign bits at each CN; returns 0 if all pass

**Output**
- `nrLDPC_llrRes2llrOut` — gather `llrRes` into external LLR order

### CN processing detail

`cnProc_group` processes one CN group (e.g. G3, G4, …, G19 for BG1):

```
for each (CN i, lifting symbol z):
    Pass 1 over all numBN positions k:
        sgn_all ^= cnProcBuf[startAddr + k*bitOff + i*Z + z]
        track min1, min2, min1_k
    Pass 2 — write extrinsic for each position j:
        mag_j  = (j == min1_k) ? min2 : min1
        sgn_j  = sgn_all ^ cnProcBuf[..j..]   // product of signs excluding j
        result = (sgn_j & 0x80) ? -mag_j : mag_j
```

`bitOff = numCN × NR_LDPC_ZMAX` — stride between BN position layers (bytes).

### Buffer layout in `bnProcBuf` / `bnProcBufRes`

BN groups are indexed by connectivity degree (`cnidx` = number of connected CNs − 1):

```
bnProcBuf[lut_startAddrBnGroups[g] + k * numBn * ZMAX + b]
                                      ↑ CN layer k         ↑ BN sample b ∈ [0, numBn*Z)
```

Only `numBn × Z` bytes are valid within each `numBn × ZMAX`-byte layer; the
extra `(ZMAX − Z) × numBn` bytes provide alignment for all lifting sizes without
recomputing offsets.

### Working buffers

All six processing buffers are heap-allocated on the DSP to stay within the
cDSP thread stack limit (~64 KB):

| Buffer         | Size (bytes)              |
|----------------|---------------------------|
| cnProcBuf      | 316 × 384 = 121 344       |
| cnProcBufRes   | 316 × 384 = 121 344       |
| bnProcBuf      | 316 × 384 = 121 344       |
| bnProcBufRes   | 316 × 384 = 121 344       |
| llrRes         | 27 000                    |
| llrProcBuf     | 27 000                    |
| **Total**      | **~540 KB**               |

## Build

### ARM stub — integrated OAI build (recommended)

Add `-DHEXAGON_LDPC=ON` to the normal DragonWing top-level cmake invocation:

```sh
cd openairinterface5g
cmake -B build-dragonwing \
    -DCMAKE_TOOLCHAIN_FILE=cmake_targets/cross-arm-dragonwing.cmake \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DHEXAGON_LDPC=ON \
    .
cmake --build build-dragonwing --target ldpc_hexagon
```

`libldpc_hexagon.so` is placed alongside the other OAI loadable modules in
`build-dragonwing/`.

### ARM stub — standalone build

```sh
cmake -B build-arm \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DOAI_SRC_ROOT=/path/to/openairinterface5g \
    -DCMAKE_TOOLCHAIN_FILE=<oai>/cmake_targets/cross-arm-dragonwing.cmake \
    -DOS_TYPE=HLOS \
    <oai>/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_hexagon
cmake --build build-arm
```

### DSP skel (always built separately with hexagon-clang)

```sh
cmake -B build-dsp \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DOAI_SRC_ROOT=/path/to/openairinterface5g \
    -DCMAKE_TOOLCHAIN_FILE=/opt/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake \
    -DDSP_VERSION=v73 \
    <oai>/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_hexagon
cmake --build build-dsp
```

Confirm the correct DSP ISA version for the SA9000P:
```sh
adb shell cat /sys/devices/soc0/soc_id
```

### Deploy

```sh
adb push build-dragonwing/libldpc_hexagon.so   /data/oai/lib/
adb push build-dsp/libldpc_hexagon_skel.so     /vendor/lib/rfsa/adsp/
```

## HVX upgrade path

The `cnProc_group` inner loops are the compute bottleneck. Each iteration
touches `numBN × numCN × Z` int8 samples with `numBN` passes. The HVX
equivalent replaces the two scalar passes with 3–4 vector instructions per
128-byte chunk:

| Scalar operation | HVX intrinsic |
|-----------------|---------------|
| `abs8(v)` | `Q6_V_vabs_VB` |
| `min(min1, av)` | `Q6_Vub_vmin_VubVub` |
| `sgn ^= v` (sign accumulation) | `Q6_V_vxor_VV` |
| `(sgn & 0x80) ? -mag : mag` | `Q6_Q_vcmp_gt_VbVb` + `Q6_V_vmux_QVV` |

Compile the skel with `-mhvx -mhvx-length=128B` and replace the body of
`cnProc_group` to unlock full cDSP throughput.

## Limitations / future work

- `check_crc` (function pointer) cannot cross FastRPC. Convergence is currently
  determined solely by the min-sum parity check on the DSP.
- `rpcmem_alloc` is called on every `LDPCdecoder` invocation; for production,
  allocate persistent rpcmem buffers at `LDPCinit` time and reuse them.
- The scalar baseline has not yet been benchmarked on hardware; expected
  throughput on the cDSP scalar pipeline is comparable to A78 NEON before HVX
  vectorisation.

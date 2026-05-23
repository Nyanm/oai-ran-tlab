<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Hexagon DSP Offload Template

This directory is a starting point for offloading computation from the OAI ARM
cores to the Qualcomm Hexagon DSP on the DragonWing IQ-9075.

The pattern is the same as `libldpc.so` or `libdfts.so`: OAI loads a shared
library at runtime via `dlopen`, calls functions through typed function
pointers, and is unaware of how the work is performed inside.  Here the .so
is a thin glue layer on the ARM side; the actual computation runs on the
Compute DSP (cDSP) via Qualcomm's **FastRPC** mechanism.

[[_TOC_]]

---

## 1  How FastRPC works

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ARM cores (aarch64, Linux/HLOS)                                    │
  │                                                                     │
  │   OAI process                                                       │
  │   ├─ dlopen("libhexdsp_offload.so")                                 │
  │   └─ hexdsp_offload_process(in, out)                                │
  │           │                                                         │
  │           │  1. rpcmem_alloc() — allocate ION buffer (physically    │
  │           │     contiguous, visible to both ARM MMU and DSP SMMU)   │
  │           │  2. memcpy(in → rpc_in)                                 │
  │           │  3. hexdsp_offload_process() ─── FastRPC syscall ──►   │
  │           │                                                    │    │
  │  libhexdsp_offload.so                                          │    │
  │  ├─ hexdsp_offload_stub.c  (qaic-generated marshalling)        │    │
  │  └─ hexdsp_offload_arm.c   (rpcmem + OAI glue)                │    │
  └────────────────────────────────────────────────────────────────│────┘
                                                                   │
                              kernel: ion_map / smmu_map           │
                                                                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Hexagon cDSP (QuRT RTOS, protected domain)                         │
  │                                                                     │
  │   FastRPC dispatcher                                                │
  │   └─ unmarshal args → call hexdsp_offload_process()                │
  │                                                                     │
  │  libhexdsp_offload_skel.so  (loaded from /vendor/lib/rfsa/adsp/)   │
  │  ├─ hexdsp_offload_skel.c  (qaic-generated dispatch)               │
  │  └─ hexdsp_offload_imp.c   (your compute kernel — fill this in)    │
  │                                                                     │
  │   4. compute on in_buf[], write out_buf[]                           │
  │   5. return 0                                                       │
  └─────────────────────────────────────────────────────────────────────┘
                                                                   │
                              kernel: flush cache / unmap          │
                                                                   ▼
  ARM side resumes:
  │   6. memcpy(rpc_out → out)
  │   7. rpcmem_free()
```

### 1.1  The IDL and qaic

The interface between ARM and DSP is declared in a single `.idl` file
(`inc/hexdsp_offload.idl`).  The SDK tool `qaic` compiles it into three
files that are never hand-edited:

| Generated file            | Used by  | Purpose                                      |
|---------------------------|----------|----------------------------------------------|
| `hexdsp_offload_stub.c`   | ARM .so  | Marshals arguments into a FastRPC message    |
| `hexdsp_offload_skel.c`   | DSP .so  | Unmarshals the message, calls `_imp.c`       |
| `hexdsp_offload.h`        | both     | C function prototypes visible on both sides  |

IDL types map to C as follows:

| IDL type            | C type (ARM stub)               | C type (DSP skel)        |
|---------------------|---------------------------------|--------------------------|
| `in sequence<uint8>`| `const uint8_t *, int len`      | `const uint8_t *, int`   |
| `rout sequence<uint8>`| `uint8_t *, int len`          | `uint8_t *, int`         |
| `long`              | `int`                           | `int`                    |
| `remote_handle64`   | opaque handle (open/close)      | `remote_handle64`        |

`in` means ARM → DSP (read-only on DSP).  `rout` means DSP → ARM
(written by DSP, read by ARM after the call returns).

### 1.2  Memory: ION and rpcmem

The DSP has its own MMU (SMMU).  Ordinary `malloc` buffers on the ARM side
are **not visible** to the DSP.  Data must be in an ION buffer — a
physically-contiguous allocation that both the ARM MMU and the DSP SMMU can
map.

`rpcmem_alloc()` (from the Hexagon SDK) handles this transparently:

```c
uint8_t *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
// use buf normally from ARM ...
// FastRPC maps it into DSP address space automatically when passed as
// an in/rout sequence argument.
rpcmem_free(buf);
```

If you pass a plain `malloc` buffer to a FastRPC call the kernel will make
an extra copy; for large blocks (e.g. a full LDPC codeblock) this kills
performance.  The ARM glue in `hexdsp_offload_arm.c` therefore allocates
rpcmem bounce buffers internally and copies once on each side, keeping the
OAI caller agnostic of this requirement.

For zero-copy operation the caller can allocate rpcmem directly and skip
the bounce; this requires exposing `rpcmem_alloc` to the OAI layer.

### 1.3  FastRPC domains

The SA9000P has several DSP subsystems.  For compute offload the relevant
one is the **cDSP** (Compute DSP), domain ID 3.  The domain is encoded in
the URI passed to `hexdsp_offload_open()`:

```c
snprintf(uri, sizeof(uri), "%s&_dom=3", hexdsp_offload_URI);
```

`hexdsp_offload_URI` is a string constant generated by `qaic` from the
interface name.

### 1.4  The skel library on the device

The DSP side (`libhexdsp_offload_skel.so`) is a Hexagon shared library
loaded by the FastRPC runtime from a fixed search path on the device:

```
/vendor/lib/rfsa/adsp/
```

The ARM stub looks for it there automatically; no path configuration is
needed.  The library runs inside a **protected domain** (PD) on the cDSP —
a sandboxed QuRT process.  A crash in the skel library does not crash the
ARM process; FastRPC returns an error code.

---

## 2  Directory structure

```
hexagon_offload_template/
├── CMakeLists.txt          ← single file, builds ARM stub or DSP skel
├── inc/
│   └── hexdsp_offload.idl  ← interface definition (qaic input)
└── src/
    ├── hexdsp_offload_arm.c ← ARM glue: rpcmem bounce + OAI dlopen exports
    └── hexdsp_offload_imp.c ← DSP compute kernel (fill this in)
```

### 2.1  `inc/hexdsp_offload.idl`

Declares the `process()` method.  Edit this to add new methods or change
the argument types.  Run `qaic` (or let cmake do it via `build_idl()`) after
any change.

### 2.2  `src/hexdsp_offload_arm.c`

Compiled into **`libhexdsp_offload.so`** together with the qaic-generated
`hexdsp_offload_stub.c`.  Exports three symbols that OAI resolves via
`dlsym`:

| Symbol                   | Called by OAI when          |
|--------------------------|-----------------------------|
| `hexdsp_offload_init`    | module is first loaded      |
| `hexdsp_offload_shutdown`| OAI exits or unloads module |
| `hexdsp_offload_process` | each compute invocation     |

### 2.3  `src/hexdsp_offload_imp.c`

Compiled into **`libhexdsp_offload_skel.so`** with hexagon-clang.  This is
the file a student replaces with the actual Hexagon kernel.  It has access
to:

- **HAP_farf** — DSP-side logging (`FARF(RUNTIME_HIGH, "...")`), visible
  via `adb logcat -s adsprpc`
- **HVX intrinsics** — 128-byte SIMD vectors; enable with
  `-mhvx -mhvx-length=128B` and `#include <hexagon_types.h>`
- **HAP_mem** — DSP heap if local scratch is needed
- **QuRT** — threading, mutexes, timers (advanced use)

---

## 3  Build

Both builds are invoked from the OAI repository root.  They are independent
and can run in any order.

### 3.1  Prerequisites

```shell
# Hexagon SDK (already installed for DragonWing ARM build)
source /opt/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
```

### 3.2  ARM stub (libhexdsp_offload.so)

Cross-compiled with the DragonWing ARM toolchain.

```shell
cmake -B build-hexdsp-arm \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DCMAKE_TOOLCHAIN_FILE=cmake_targets/cross-arm-dragonwing.cmake \
    -DOS_TYPE=HLOS \
    common/utils/hexagon_offload_template
cmake --build build-hexdsp-arm
```

Output: `build-hexdsp-arm/libhexdsp_offload.so`

### 3.3  DSP skel (libhexdsp_offload_skel.so)

Compiled with hexagon-clang targeting the cDSP instruction set.

```shell
cmake -B build-hexdsp-dsp \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DCMAKE_TOOLCHAIN_FILE=/opt/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake \
    -DDSP_VERSION=v73 \
    common/utils/hexagon_offload_template
cmake --build build-hexdsp-dsp
```

Output: `build-hexdsp-dsp/libhexdsp_offload_skel.so`

> **DSP_VERSION:** must match the Hexagon ISA version of the cDSP on the
> device.  To find it:
> ```shell
> adb shell cat /sys/devices/soc0/soc_id
> ```
> Cross-reference the SoC ID with the Qualcomm product documentation.
> Known values: `v73` (SM8550 era), `v75`, `v79`, `v81`.

### 3.4  Enabling HVX in the DSP build

Add the following to the DSP cmake invocation:

```shell
-DCMAKE_C_FLAGS="-mhvx -mhvx-length=128B"
```

Then in `hexdsp_offload_imp.c`:

```c
#include <hexagon_types.h>

// HVX_Vector is 128 bytes wide.  Pointers must be 128-byte aligned.
HVX_Vector *vin  = (HVX_Vector *)in_buf;
HVX_Vector *vout = (HVX_Vector *)out_buf;
int nvec = in_bufLen / sizeof(HVX_Vector);
for (int i = 0; i < nvec; i++)
    vout[i] = Q6_Vub_vsat_VhVh(vin[i], vin[i]);  // example: saturate
```

---

## 4  Deploy

```shell
# ARM stub: alongside the other OAI shared libs
adb push build-hexdsp-arm/libhexdsp_offload.so   /data/oai/lib/

# DSP skel: FastRPC searches this path automatically
adb shell mkdir -p /vendor/lib/rfsa/adsp
adb push build-hexdsp-dsp/libhexdsp_offload_skel.so  /vendor/lib/rfsa/adsp/
```

---

## 5  OAI integration

OAI's `load_module_shlib` infrastructure (`common/utils/load_module_shlib.c`)
loads offload libraries at runtime using `dlopen` + `dlsym`.  To wire in
the hexdsp offload, add a loader similar to `nrLDPC_load.c`:

```c
#include "load_module_shlib.h"

// Function pointer types matching hexdsp_offload_arm.c exports
typedef int (*hexdsp_init_f)(void);
typedef int (*hexdsp_shutdown_f)(void);
typedef int (*hexdsp_process_f)(const uint8_t *, uint32_t, uint8_t *, uint32_t);

typedef struct {
    hexdsp_init_f     init;
    hexdsp_shutdown_f shutdown;
    hexdsp_process_f  process;
} hexdsp_interface_t;

int load_hexdsp(hexdsp_interface_t *itf)
{
    loader_shlibfunc_t fdesc[] = {
        {.fname = "hexdsp_offload_init"},
        {.fname = "hexdsp_offload_shutdown"},
        {.fname = "hexdsp_offload_process"},
    };
    int ret = load_module_version_shlib("hexdsp_offload", "", fdesc,
                                        sizeofArray(fdesc), NULL);
    if (ret) return ret;
    itf->init     = (hexdsp_init_f)    fdesc[0].fptr;
    itf->shutdown = (hexdsp_shutdown_f)fdesc[1].fptr;
    itf->process  = (hexdsp_process_f) fdesc[2].fptr;
    return itf->init();
}
```

The loader searches for `libhexdsp_offload.so` on `LD_LIBRARY_PATH` or the
path configured under `loader.hexdsp_offload.shlibpath` in the OAI config
file.

---

## 6  Adapting the template

To implement a specific kernel (e.g. LDPC decoding):

1. **Rename** — replace `hexdsp_offload` with your module name throughout
   (IDL, source files, CMakeLists.txt).

2. **Extend the IDL** — add methods matching the ARM function signatures OAI
   expects.  Each `in sequence<T>` argument becomes a `(const T*, int)` pair
   in C.  Each `rout sequence<T>` becomes a `(T*, int)` pair.

3. **ARM glue** — update `hexdsp_offload_arm.c` to allocate rpcmem buffers
   sized for your data (LLR arrays, code blocks, etc.) and call the new IDL
   methods.

4. **DSP kernel** — implement the compute in `hexdsp_offload_imp.c`.  Start
   scalar, verify correctness, then vectorise with HVX.

5. **OAI loader** — write a `load_<module>.c` following the pattern in §5
   above, using the same function-pointer types OAI already uses for LDPC or
   DFT offload.

---

## 7  Debugging

### 7.1  DSP logs

`FARF()` messages from the skel library appear in the Android/Linux log:

```shell
adb logcat -s adsprpc
```

Log levels: `RUNTIME_HIGH` (always on), `HIGH`, `MED`, `LOW`.

### 7.2  FastRPC error codes

Negative return values from `hexdsp_offload_open()` or `_process()` are
FastRPC error codes.  Common ones:

| Code   | Meaning                                                      |
|--------|--------------------------------------------------------------|
| `-1`   | Generic failure (check logcat for detail)                    |
| `3`    | Skel library not found in `/vendor/lib/rfsa/adsp/`          |
| `6`    | DSP subsystem not available or not powered                   |
| `-14`  | Bad address — buffer not allocated with `rpcmem_alloc`       |

### 7.3  Simulator

The DSP skel can be tested on the build host without a device using the
Hexagon simulator:

```shell
# Build with the simulator target (no device needed)
cmake -B build-hexdsp-sim \
    -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
    -DCMAKE_TOOLCHAIN_FILE=/opt/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake \
    -DDSP_VERSION=v73 \
    -DNO_QURT_INC=1 \
    common/utils/hexagon_offload_template
cmake --build build-hexdsp-sim
# The SDK's hexagon_fun.cmake adds a runHexagonSim() target automatically.
cmake --build build-hexdsp-sim --target run
```

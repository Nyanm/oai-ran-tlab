<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Building OAI for Qualcomm DragonWing IQ-9/X (ARM)

This document covers cross-compiling OAI for the ARM cores of the Qualcomm
DragonWing IQ-9/X platform (SA9000P SoC, Kryo 780 / Cortex-A78).
The Hexagon DSP component is not covered here.

The toolchain file is `cmake_targets/cross-arm-dragonwing.cmake`, which uses the
GCC cross-compiler bundled with the Hexagon SDK rather than the Ubuntu package.
The build procedure otherwise follows the same two-step pattern as the generic
ARM64 cross-compile described in `cross-compile.md`.

> **Note on the compiler:** The Hexagon SDK includes two compilers.
> `hexagon-clang` (LLVM 19) targets the **Hexagon DSP only** and cannot produce
> ARM binaries.  The ARM cross-compiler is `aarch64-none-linux-gnu-gcc`
> (Arm GNU Toolchain 11.3.1) located under
> `tools/gcc_tools_64/bin/` in the SDK tree.

[[_TOC_]]

---

## 1 Prerequisites on the build host

### 1.1 Hexagon SDK

The SDK must be installed, defaulting to `/opt/Hexagon_SDK/6.4.0.2`.
Run the SDK's environment script once per shell session (or add to your profile):

```shell
source /opt/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
```

If the SDK is installed elsewhere, pass `-DHEXAGON_SDK_ROOT=<path>` to every
`cmake` invocation in this guide.

### 1.2 OAI native build dependencies

The host machine needs OAI's standard build tools.  If not installed yet:

```shell
cmake_targets/build_oai -I
```

### 1.3 ARM64 target libraries

OAI links against several libraries that must be available for the `aarch64`
architecture at link time.  The standard way is Ubuntu multiarch.

> **Caveat:** The `dpkg --add-architecture` approach requires that your
> apt sources serve `aarch64` packages.  Ubuntu 22.04 / 24.04 on a standard
> x86 host works out of the box via `ports.ubuntu.com`.  If you are on a
> non-standard host, inside a corporate mirror, or inside a container without
> network access, you may need to adapt the sources list below — or provide
> the libraries from the DragonWing device filesystem instead (see
> [Section 4](#4-alternative-sysroot-from-device)).

#### 1.3.1 Enable the arm64 architecture in apt

**Ubuntu 24.04 (Noble)** — the sources are in `ubuntu.sources` format:

```shell
sudo dpkg --add-architecture arm64

# Restrict the existing ubuntu.sources to amd64 only
sudo sed -i '/^Components:/a Architectures: amd64' \
    /etc/apt/sources.list.d/ubuntu.sources

# Add a separate arm64 source pointing to ports.ubuntu.com
sudo tee /etc/apt/sources.list.d/arm-cross-compile-sources.list <<'EOF'
deb [arch=arm64] http://ports.ubuntu.com/ noble main restricted
deb [arch=arm64] http://ports.ubuntu.com/ noble-updates main restricted
deb [arch=arm64] http://ports.ubuntu.com/ noble universe
deb [arch=arm64] http://ports.ubuntu.com/ noble-updates universe
deb [arch=arm64] http://ports.ubuntu.com/ noble multiverse
deb [arch=arm64] http://ports.ubuntu.com/ noble-updates multiverse
deb [arch=arm64] http://ports.ubuntu.com/ noble-backports main restricted universe multiverse
EOF
```

**Ubuntu 22.04 (Jammy)** — sources are still in the old `sources.list` format:

```shell
sudo dpkg --add-architecture arm64

sudo cp /etc/apt/sources.list "/etc/apt/sources.list.$(date).backup"
sudo sed -i -E "s/(deb)\ (http:.+)/\1\ [arch=amd64]\ \2/" \
    /etc/apt/sources.list

sudo tee /etc/apt/sources.list.d/arm-cross-compile-sources.list <<'EOF'
deb [arch=arm64] http://ports.ubuntu.com/ jammy main restricted
deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates main restricted
deb [arch=arm64] http://ports.ubuntu.com/ jammy universe
deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates universe
deb [arch=arm64] http://ports.ubuntu.com/ jammy multiverse
deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates multiverse
deb [arch=arm64] http://ports.ubuntu.com/ jammy-backports main restricted universe multiverse
EOF
```

#### 1.3.2 Install the arm64 packages

```shell
sudo apt-get update
sudo apt-get install --yes \
    libc6-dev-i386 \
    libreadline-dev:arm64 \
    libgnutls28-dev:arm64 \
    libconfig-dev:arm64 \
    libsctp-dev:arm64 \
    libssl-dev:arm64 \
    libtool:arm64 \
    zlib1g-dev:arm64 \
    libyaml-cpp-dev:arm64
```

> **Note:** `libc6-dev-i386` is for the host (code-generation tools), not the
> target.  All other packages with `:arm64` suffix are for the cross-linked
> ARM binaries.

> The list above is believed to be complete but may grow as new OAI features
> are enabled.  If cmake reports a missing package, install `<package>:arm64`
> and re-run cmake.

---

## 2 Build

Both build directories live directly under the OAI repository root.
**Each directory must be created fresh** — `CMAKE_TOOLCHAIN_FILE` is only
honoured on the very first cmake run in a directory.  If a `CMakeCache.txt`
already exists, cmake ignores the toolchain file and silently produces a
native x86 build.

### 2.1 Step 1 — native host tools

These are x86 binaries that cmake runs during the cross-compile step to
generate LDPC processing code and the T-tracer event IDs.

```shell
cd <oai-root>

mkdir build        # must not contain a prior CMakeCache.txt
cd build
cmake .. -DAVOID_SIGN=ON
make -j$(nproc) ldpc_generators generate_T
cd ..
```

> **Note:** `-DAVOID_SIGN=ON` tells the LDPC code generator to emit
> `simde_mm_xor_si128` instead of `simde_mm_sign_epi8` for sign extraction.
> SIMDe's translation of `mm_sign_epi8` is incorrect for the MSB-only pattern
> used in the CN processing kernel; `mm_xor_si128` is equivalent and
> translates correctly on aarch64.  This flag has no effect on Step 2.

### 2.2 Step 2 — cross-compile for DragonWing

```shell
# Remove any previous attempt first — a stale CMakeCache.txt will cause
# CMAKE_TOOLCHAIN_FILE to be ignored with a warning but no error.
rm -rf dragonwing_build
mkdir dragonwing_build
cd dragonwing_build

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake_targets/cross-arm-dragonwing.cmake \
    -DNATIVE_DIR=../build

# Example targets — add or remove as needed
make -j$(nproc) nr-softmodem nr-cuup nr-uesoftmodem \
                params_libconfig coding rfsimulator
```

If the Hexagon SDK is not at the default path, add:

```
-DHEXAGON_SDK_ROOT=/your/path/to/Hexagon_SDK/6.4.0.2
```

The `QUALCOMM_DRAGONWING=1` flag is set automatically by the toolchain file and
causes CMake to use `-mcpu=cortex-a78` instead of the generic `-march=armv8.2-a`.

### 2.3 Deploy via adb

```shell
# Verify the device is reachable
adb devices

# Create a destination directory on the device
adb shell mkdir -p /data/oai

# Push the binaries
adb push dragonwing_build/nr-softmodem          /data/oai/
adb push dragonwing_build/nr-cuup               /data/oai/
adb push dragonwing_build/nr-uesoftmodem        /data/oai/

# Push shared libraries that OAI loads at runtime
adb push dragonwing_build/libparams_libconfig.so /data/oai/
adb push dragonwing_build/libcoding.so           /data/oai/
adb push dragonwing_build/librfsimulator.so      /data/oai/

# On the device, set LD_LIBRARY_PATH before running
adb shell "export LD_LIBRARY_PATH=/data/oai && /data/oai/nr-softmodem --help"
```

If the standard system libraries (libgnutls, libssl, libconfig, …) are not
present on the device, they must be pushed alongside the OAI binaries:

```shell
# Example: find and push the arm64 shared libs from the build host
for lib in libgnutls libssl libcrypto libconfig libsctp; do
    find /usr/lib/aarch64-linux-gnu -name "${lib}.so*" -exec \
        adb push {} /data/oai/ \;
done
```

---

## 3 Building for O-RAN FHI 7.2 (DPDK + libxran + armral)

This section covers cross-compiling the three additional libraries needed for
the O-RAN 7.2 fronthaul interface (`-DOAI_FHI72=ON`).  They must be built in
order: DPDK → libxran → armral → OAI.

All libraries are installed into a common staging directory (`~/dw-sysroot`)
so they stay isolated from the host system and can be pushed to the device
together with the OAI binaries.

```shell
export DW_TC=/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/bin/aarch64-none-linux-gnu
export DW_SYSROOT=$HOME/dw-sysroot
mkdir -p $DW_SYSROOT
```

### 3.1 DPDK

OAI supports two xran release tracks:

| xran release | xran version | DPDK version |
|---|---|---|
| F | 6.1.9  | 20.11.9 |
| K | 11.1.1 | 24.11.4 (minimum 22) |

The K release is recommended for new deployments.  The commands below show
the K release; swap version strings for F where noted.

#### 3.1.1 Build tools

```shell
sudo apt-get install -y meson ninja-build python3-pyelftools
```

#### 3.1.2 Create a meson cross-file

```shell
cat > ~/aarch64-dragonwing.ini <<'EOF'
[binaries]
c          = '/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/bin/aarch64-none-linux-gnu-gcc'
cpp        = '/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/bin/aarch64-none-linux-gnu-g++'
ar         = '/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/bin/aarch64-none-linux-gnu-ar'
strip      = '/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/bin/aarch64-none-linux-gnu-strip'
pkg-config = 'pkg-config'

[host_machine]
system     = 'linux'
cpu_family = 'aarch64'
cpu        = 'armv8.2-a'
endian     = 'little'

[properties]
# DPDK reads this via meson.get_external_property() when cross-compiling;
# -Dplatform=generic on the command line is ignored in that path.
platform = 'generic'

[built-in options]
c_args        = ['-I/usr/include/aarch64-linux-gnu', '-I/usr/include']
cpp_args      = ['-I/usr/include/aarch64-linux-gnu', '-I/usr/include']
c_link_args   = ['-L/usr/lib/aarch64-linux-gnu']
cpp_link_args = ['-L/usr/lib/aarch64-linux-gnu']
EOF
```

#### 3.1.3 Download, build and install

```shell
cd ~
wget http://fast.dpdk.org/rel/dpdk-24.11.4.tar.xz   # K release
# wget http://fast.dpdk.org/rel/dpdk-20.11.9.tar.xz # F release
tar xf dpdk-24.11.4.tar.xz
cd dpdk-stable-24.11.4

# Always wipe the build directory — meson caches pkg-config results and will
# silently reuse x86 libelf paths if the previous setup run was bad.
rm -rf build-dragonwing

PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu/pkgconfig \
meson setup build-dragonwing \
    --cross-file ~/aarch64-dragonwing.ini \
    --prefix=$HOME/dw-sysroot \
    -Dplatform=generic \
    -Ddisable_libs=bpf \
    -Ddisable_drivers=net/ice,net/i40e,net/iavf,net/ixgbe
ninja -C build-dragonwing
ninja -C build-dragonwing install
```

> **Note:** `PKG_CONFIG_LIBDIR` replaces (not appends) the pkg-config search
> path, ensuring meson finds arm64 `.pc` files rather than x86 ones.
> `-Ddisable_libs=bpf` drops the libelf dependency (only needed for eBPF
> packet filtering, not for fronthaul use), avoiding the x86/arm64 libelf
> mismatch.  The `-Ddisable_drivers` list trims x86-only NIC drivers;
> add DragonWing-specific PMDs with `-Denable_drivers=...` if needed.

Verify pkg-config can see it:

```shell
PKG_CONFIG_PATH=$DW_SYSROOT/lib/pkgconfig \
    pkg-config --modversion libdpdk
```

### 3.2 libxran

#### 3.2.1 Clone and patch

```shell
git clone https://github.com/openairinterface/o-du-phy.git ~/phy
cd ~/phy

# K release — tag must match K_VERSION in radio/fhi_72/CMakeLists.txt
git checkout oran_k_release_v1.1
# F release alternative:
# git checkout oran_f_release_v1.9
# git apply ~/openairinterface5g/cmake_targets/tools/oran_fhi_integration_patches/F/oaioran_F.patch
```

#### 3.2.2 Cross-compile

```shell
sudo apt-get install -y libnuma-dev:arm64

cd ~/phy/fhi_lib/lib
make clean

CPATH=/usr/include/aarch64-linux-gnu:/usr/include \
PKG_CONFIG_PATH=$DW_SYSROOT/lib/pkgconfig \
make -j$(nproc) XRAN_LIB_SO=1 \
    CC=${DW_TC}-gcc \
    CPP=${DW_TC}-g++ \
    AR=${DW_TC}-ar \
    AS=${DW_TC}-as \
    LD=${DW_TC}-gcc \
    WIRELESS_SDK_TOOLCHAIN=gcc \
    TARGET=armv8 \
    RTE_SDK=$DW_SYSROOT \
    XRAN_DIR=~/phy/fhi_lib
```

The output is `~/phy/fhi_lib/lib/build/libxran.so`.

> **Note:** The xran Makefile for `TARGET=armv8` hardcodes `CC := gcc`,
> `CPP := g++` etc., so the cross-compiler must be passed as **make
> command-line arguments** (after `make`) — not as shell environment variables
> — so they override the Makefile assignments.  `PKG_CONFIG_PATH` and `CPATH`
> must remain shell environment variables: the Makefile invokes pkg-config via
> `$(shell ...)` which inherits the shell environment, and GCC reads `CPATH`
> independently to prepend extra include directories (bridging the gap between
> the SDK GCC's minimal built-in sysroot and the Ubuntu arm64 system headers).
> `RTE_SDK` must point at the **installed** DPDK prefix so pkg-config finds
> `libdpdk.pc` and resolves the correct arm64 include paths.

### 3.3 armral (Arm RAN Acceleration Library)

armral is required by OAI on aarch64 for BFP compression.

```shell
git clone https://git.gitlab.arm.com/networking/ral.git ~/ral
cd ~/ral
git checkout armral-25.01
mkdir build && cd build

cmake .. -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=~/openairinterface5g/cmake_targets/cross-arm-dragonwing.cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=$DW_SYSROOT \
    -DCMAKE_C_FLAGS="-mcpu=cortex-a78+crypto" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-a78+crypto"

ninja
ninja install
```

### 3.4 OAI gNB with FHI 7.2

```shell
cd ~/openairinterface5g

# Step 1 — native host tools (skip if already done)
mkdir -p build && cd build && cmake .. -DAVOID_SIGN=ON && make -j$(nproc) ldpc_generators generate_T && cd ..

# Step 2 — DragonWing cross-compile with FHI 7.2
rm -rf dragonwing_build && mkdir dragonwing_build && cd dragonwing_build

PKG_CONFIG_PATH=$DW_SYSROOT/lib/pkgconfig \
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake_targets/cross-arm-dragonwing.cmake \
    -DNATIVE_DIR=../build \
    -DOAI_FHI72=ON \
    -Dxran_LOCATION=$HOME/phy/fhi_lib/lib \
    -Darmral_LOCATION=$DW_SYSROOT

make -j$(nproc) nr-softmodem nr-cuup params_libconfig coding oran_fhlib_5g
```

> **Note:** `PKG_CONFIG_PATH` (not `PKG_CONFIG_LIBDIR`) is used here so that
> the ARM DPDK `.pc` file is found *in addition to* the existing system arm64
> packages, rather than replacing them.

### 3.5 linuxptp (ptp4l / phc2sys)

S-plane synchronization requires `ptp4l` and `phc2sys` from linuxptp 3.1.1.
They are not present on the DragonWing by default and must be cross-compiled.

```shell
git clone https://git.code.sf.net/p/linuxptp/code ~/linuxptp
cd ~/linuxptp
git checkout v3.1.1

KBUILD_OUTPUT=/opt/Hexagon_SDK/6.4.0.2/tools/gcc_tools_64/aarch64-none-linux-gnu/libc \
CROSS_COMPILE=${DW_TC}- \
make -j$(nproc) EXTRA_CFLAGS="-mcpu=cortex-a78"
```

> **Note — two common pitfalls when cross-compiling linuxptp:**
>
> 1. Use `EXTRA_CFLAGS`, **not** `CFLAGS`, to add compiler flags.
>    The linuxptp `makefile` defines `CFLAGS = -Wall $(VER) $(incdefs) $(EXTRA_CFLAGS)`.
>    Passing `CFLAGS=...` on the make command line replaces that entire definition,
>    discarding `$(incdefs)` — which contains `-DHAVE_ONESTEP_SYNC` and `-D_GNU_SOURCE`.
>    Without `-DHAVE_ONESTEP_SYNC`, `missing.h` redeclares the enum that the system
>    `net_tstamp.h` already defines, causing a compile error.
>
> 2. Set `KBUILD_OUTPUT` to the cross-compiler sysroot so `incdefs.sh` picks up the
>    ARM `linux/net_tstamp.h` instead of the host x86 one.  `CROSS_COMPILE` must be
>    an environment variable (not a `make CC=` argument) because `incdefs.sh` calls
>    `${CROSS_COMPILE}cpp` in a subshell to discover include paths.

Deploy:

```shell
adb push ~/linuxptp/ptp4l   /usr/sbin/ptp4l
adb push ~/linuxptp/phc2sys /usr/sbin/phc2sys
adb shell chmod +x /usr/sbin/ptp4l /usr/sbin/phc2sys
```

Follow the ptp4l / phc2sys configuration in `doc/ORAN_FHI7.2_Tutorial.md`
(§ "PTP configuration") once the RTL 10G NIC is available.

### 3.6 RTL8127 kernel module

The IQ-9075 EVK uses a Realtek RTL8127A (10 GbE) NIC for the fronthaul connection.
The vendor kernel (`6.18.12-g06f8ab99cf04-dirty`) does not include any RTL8127 driver.
Two options are available, depending on whether hardware timestamping is needed:

| Option | Module | Hardware timestamping | Notes |
|--------|--------|-----------------------|-------|
| A (§ 3.6.1–3.6.5) | mainline `r8169` | **No** — software only | Simpler build |
| B (§ 3.6.6) | Realtek out-of-tree `r8127` | **Yes** — PHC registered | Required for ptp4l hardware mode |

For S-plane synchronisation with `ptp4l`, **use option B**.  Option A is adequate if
only basic connectivity is needed (e.g. rfsimulator, debug access).

Because the vendor kernel has `CONFIG_MODVERSIONS=n` and `CONFIG_MODULE_SIG=n`, the only
requirement for any module to load is that its **version magic string matches exactly**:
`6.18.12-g06f8ab99cf04-dirty SMP preempt mod_unload aarch64`

> **Note:** If QCOM provides a kernel SDK or `kernel-devsrc` package for the IQ-9075
> build, use that instead — it will produce a better-matched module.  The procedure
> below is a best-effort workaround using the mainline source.

#### 3.6.1 Download and configure the kernel source

```shell
cd ~
# Download mainline stable 6.18.12
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.18.12.tar.xz
tar xf linux-6.18.12.tar.xz
cd linux-6.18.12

# Seed config from the running device kernel
adb pull /proc/config.gz /tmp/kernel-config.gz
gunzip -f /tmp/kernel-config.gz
cp /tmp/kernel-config .config

# Enable r8169 and its Realtek PHY driver as loadable modules
scripts/config --module CONFIG_R8169
scripts/config --module CONFIG_REALTEK_PHY

# Set LOCALVERSION so vermagic matches the device exactly
scripts/config --set-str CONFIG_LOCALVERSION "-g06f8ab99cf04-dirty"
scripts/config --disable CONFIG_LOCALVERSION_AUTO

# Resolve any new Kconfig options introduced since the device config was made.
# CROSS_COMPILE must be set here so compiler/assembler capability checks (MTE,
# shadow call stack, etc.) are evaluated against the ARM toolchain, not the host.
# Omitting CROSS_COMPILE causes interactive questions during modules_prepare.
make ARCH=arm64 CROSS_COMPILE=${DW_TC}- olddefconfig
```

> **Note:** `olddefconfig` may set `CONFIG_HAVE_RUST=y` if the build host has `rustc`
> installed.  This does not affect the r8169 module build, but if `modules_prepare`
> fails with a Rust-related error, disable it and re-run olddefconfig:
> ```shell
> scripts/config --disable CONFIG_HAVE_RUST
> make ARCH=arm64 CROSS_COMPILE=${DW_TC}- olddefconfig
> ```

#### 3.6.2 Prepare the kernel build tree

```shell
# Builds host scripts (modpost etc.) and generates arch headers.
# Does NOT compile the full kernel.
make ARCH=arm64 CROSS_COMPILE=${DW_TC}- modules_prepare
```

#### 3.6.3 Build r8169.ko and realtek.ko

Two modules are needed: the MAC driver (`r8169`) and the Realtek PHY driver (`realtek`).
The r8169 probe fails with "no dedicated PHY driver found for PHY ID 0x001cc890" if
`realtek.ko` is not loaded first.

```shell
make ARCH=arm64 CROSS_COMPILE=${DW_TC}- M=drivers/net/ethernet/realtek KBUILD_MODPOST_WARN=1 modules
make ARCH=arm64 CROSS_COMPILE=${DW_TC}- M=drivers/net/phy/realtek      KBUILD_MODPOST_WARN=1 modules
```

Outputs: `drivers/net/ethernet/realtek/r8169.ko` and `drivers/net/phy/realtek/realtek.ko`.

> **Note:** `KBUILD_MODPOST_WARN=1` is required because `Module.symvers` does not exist
> (only generated by a full kernel build).  Without it, `modpost` errors on unresolved
> symbols such as `napi_alloc_skb`, `free_irq`, etc.  These are all standard kernel
> exports that are present in the running kernel; since `CONFIG_MODVERSIONS=n` on the
> device there is no CRC check at load time, so the warnings are safe to ignore.

#### 3.6.4 Prepare the firmware

The RTL8127A requires firmware `rtl_nic/rtl8127a-1.fw`.  The build host has it in
compressed form (`linux-firmware` package); decompress before pushing because the vendor
kernel does not support `CONFIG_FW_LOADER_COMPRESS`.

```shell
zstd -d /lib/firmware/rtl_nic/rtl8127a-1.fw.zst -o ~/rtl8127a-1.fw
```

#### 3.6.5 Deploy and load

The modules depend on `libphy.ko` and `mdio-bus.ko`, which are already present in the
device's module tree.  Load order matters: `libphy` → `realtek` → `r8169`.

```shell
# Push modules and firmware
adb push ~/linux-6.18.12/drivers/net/ethernet/realtek/r8169.ko /data/local/tmp/
adb push ~/linux-6.18.12/drivers/net/phy/realtek/realtek.ko    /data/local/tmp/
adb shell mkdir -p /lib/firmware/rtl_nic
adb push ~/rtl8127a-1.fw /lib/firmware/rtl_nic/

# Load in dependency order
adb shell "modprobe libphy && insmod /data/local/tmp/realtek.ko && insmod /data/local/tmp/r8169.ko"

# Verify
adb shell "dmesg | grep -i r8169"
adb shell "ip link show"
```

If the NIC is detected, assign an IP and verify link before proceeding to the DPDK /
ptp4l configuration.  To make the modules persistent across reboots, copy them into the
device module tree:

```shell
adb shell mkdir -p /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/ethernet/realtek
adb shell mkdir -p /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/phy/realtek
adb push ~/linux-6.18.12/drivers/net/ethernet/realtek/r8169.ko \
    /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/ethernet/realtek/
adb push ~/linux-6.18.12/drivers/net/phy/realtek/realtek.ko \
    /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/phy/realtek/
adb shell depmod -a
```

#### 3.6.6 Option B — Realtek out-of-tree r8127 driver (hardware timestamping)

Realtek publishes a standalone driver with full PTP/PHC support
(`ptp_clock_info`, hardware timestamp registers).  It replaces both `r8169.ko` and
`realtek.ko` from option A with a single `r8127.ko`.

```shell
cd ~
git clone https://github.com/openwrt/rtl8127.git
cd rtl8127

make KERNELDIR=~/linux-6.18.12 \
     ARCH=arm64 \
     CROSS_COMPILE=${DW_TC}- \
     KBUILD_MODPOST_WARN=1 \
     ENABLE_PTP_SUPPORT=y \
     modules
```

> **Kernel 6.18 patch required:** The Realtek driver uses the removed `hrtimer_init` API.
> Before building, edit `r8127_ptp.c` around line 754:
> ```diff
> -        hrtimer_init(&tp->pps_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
> -        tp->pps_timer.function = rtl8127_hrtimer_for_pps;
> +        hrtimer_setup(&tp->pps_timer, rtl8127_hrtimer_for_pps, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
> ```
> `hrtimer_init` + separate `.function` assignment was replaced by `hrtimer_setup` in
> kernel 6.18.

If option A modules are already loaded, unload them first:

```shell
adb shell "rmmod r8169; rmmod realtek"
```

Deploy and load:

```shell
adb push ~/rtl8127/r8127.ko /data/local/tmp/
adb shell "modprobe libphy && insmod /data/local/tmp/r8127.ko"

# Verify hardware timestamping is now available
adb shell "ethtool -T enP1p1s0"
```

Expected output should include `PTP Hardware Clock: 0` and hardware transmit/receive
timestamp modes.  To make persistent across reboots:

```shell
adb shell mkdir -p /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/ethernet/realtek
adb push ~/rtl8127/r8127.ko \
    /lib/modules/6.18.12-g06f8ab99cf04-dirty/kernel/drivers/net/ethernet/realtek/
adb shell depmod -a
```

#### 3.7 Deploy FHI 7.2 binaries via adb

```shell
adb shell mkdir -p /data/oai/lib

# OAI executables
adb push dragonwing_build/nr-softmodem           /data/oai/
adb push dragonwing_build/liboran_fhlib_5g.so    /data/oai/lib/
adb push dragonwing_build/libparams_libconfig.so /data/oai/lib/
adb push dragonwing_build/libcoding.so           /data/oai/lib/

# DPDK shared libraries
find $DW_SYSROOT/lib -name "librte_*.so*" -exec adb push {} /data/oai/lib/ \;

# dpdk-devbind.py — needed to bind VFs to vfio-pci before starting the gNB.
# Install to /usr/local/bin to match the path used in ORAN_FHI7.2_Tutorial.md.
# Python 3 is already present on the device.
adb push $DW_SYSROOT/bin/dpdk-devbind.py /data/oai/
adb shell "cp /data/oai/dpdk-devbind.py /usr/local/bin/dpdk-devbind.py && chmod +x /usr/local/bin/dpdk-devbind.py"

# xran
adb push ~/phy/fhi_lib/lib/build/libxran.so      /data/oai/lib/

# armral
adb push $DW_SYSROOT/lib/libarmral.so            /data/oai/lib/

# Run with combined LD_LIBRARY_PATH
adb shell "export LD_LIBRARY_PATH=/data/oai/lib && /data/oai/nr-softmodem --help"
```

---

## 4 Alternative: sysroot from device

If the Ubuntu multiarch packages are not available on the build host, you can
extract the device's root filesystem and use it as a sysroot instead.

```shell
# Pull relevant library directories from the device
adb pull /usr/lib       dragonwing-sysroot/usr/lib
adb pull /usr/include   dragonwing-sysroot/usr/include
adb pull /lib           dragonwing-sysroot/lib

# Then configure cmake with an explicit sysroot (from the repo root)
rm -rf dragonwing_build && mkdir dragonwing_build && cd dragonwing_build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake_targets/cross-arm-dragonwing.cmake \
    -DNATIVE_DIR=../build \
    -DCMAKE_SYSROOT=$(pwd)/../dragonwing-sysroot \
    -DCMAKE_FIND_ROOT_PATH=$(pwd)/../dragonwing-sysroot
```

You will also need to regenerate the pkg-config search path:

```shell
export PKG_CONFIG_LIBDIR=$(pwd)/dragonwing-sysroot/usr/lib/pkgconfig:$(pwd)/dragonwing-sysroot/usr/lib/aarch64-linux-gnu/pkgconfig
```

---

## 5 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Could NOT find GnuTLS` | arm64 packages not installed or wrong `PKG_CONFIG_LIBDIR` | Check Section 1.3; verify `pkg-config --list-all` finds arm64 packages |
| `cannot find -lgnutls` at link | Linker not searching `/usr/lib/aarch64-linux-gnu` | Verify that toolchain file `CMAKE_EXE_LINKER_FLAGS_INIT` is being applied |
| `Unsupported architecture` from apt | Host apt sources not configured for arm64 | Redo Section 1.3.1 |
| Binary crashes with `SIGILL` on device | Wrong `-mcpu` / `-march` flag | The toolchain defaults to `cortex-a78`; if running on a Silver (A55) core, add `-DCMAKE_C_FLAGS=-mcpu=cortex-a55` |
| `adb: error: failed to copy` | `/data/oai` directory doesn't exist or no permission | `adb shell mkdir -p /data/oai` or use a writable path like `/data/local/tmp/oai` |

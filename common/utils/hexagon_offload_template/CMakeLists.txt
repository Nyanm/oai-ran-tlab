# SPDX-License-Identifier: LicenseRef-CSSL-1.0
#
# CMakeLists.txt — Hexagon DSP offload template.
#
# Builds two targets from a single file depending on OS_TYPE:
#
#   OS_TYPE=HLOS    → libhexdsp_offload.so   (ARM stub, loaded by OAI)
#   (default)       → libhexdsp_offload_skel.so  (DSP skeleton, pushed to device)
#
# ── ARM (HLOS) build ────────────────────────────────────────────────────────
#
#   cmake -B build-arm \
#       -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
#       -DCMAKE_TOOLCHAIN_FILE=../../cmake_targets/cross-arm-dragonwing.cmake \
#       -DOS_TYPE=HLOS \
#       common/utils/hexagon_offload_template
#   cmake --build build-arm
#
# ── DSP (Hexagon) build ──────────────────────────────────────────────────────
#
#   cmake -B build-dsp \
#       -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
#       -DCMAKE_TOOLCHAIN_FILE=/opt/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake \
#       -DDSP_VERSION=v73 \
#       common/utils/hexagon_offload_template
#   cmake --build build-dsp
#
# ── Deploy ──────────────────────────────────────────────────────────────────
#
#   # ARM stub: push alongside the other OAI shared libs
#   adb push build-arm/libhexdsp_offload.so   /data/oai/lib/
#
#   # DSP skel: FastRPC loads it from the adsp library path
#   adb push build-dsp/libhexdsp_offload_skel.so  /vendor/lib/rfsa/adsp/
#
# ── OAI integration ──────────────────────────────────────────────────────────
#
# OAI loads the ARM stub via load_module_shlib (common/utils/load_module_shlib.c):
#
#   loader_shlibfunc_t fdesc[] = {
#       {.fname = "hexdsp_offload_init"},
#       {.fname = "hexdsp_offload_shutdown"},
#       {.fname = "hexdsp_offload_process"},
#   };
#   load_module_version_shlib("hexdsp_offload", "", fdesc, 3, NULL);

cmake_minimum_required(VERSION 3.22)
project(hexdsp_offload C)

# ---------------------------------------------------------------------------
# Hexagon SDK root — defaults to the standard install path.
# Override with -DHEXAGON_SDK_ROOT=<path> if needed.
# ---------------------------------------------------------------------------
if(NOT HEXAGON_SDK_ROOT)
    set(HEXAGON_SDK_ROOT "/opt/Hexagon_SDK/6.4.0.2")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

# ---------------------------------------------------------------------------
# Include paths shared by both ARM stub and DSP skel.
# ---------------------------------------------------------------------------
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_BINARY_DIR}           # qaic writes generated .h here
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
)

# ---------------------------------------------------------------------------
# ARM (HLOS) side — stub shared library loaded by OAI via dlopen.
# ---------------------------------------------------------------------------
if(${OS_TYPE} MATCHES "HLOS")

    add_library(hexdsp_offload SHARED
        # qaic-generated stub (file created at configure time by build_idl)
        ${CMAKE_CURRENT_BINARY_DIR}/hexdsp_offload_stub
        # ARM glue: rpcmem bounce buffers + OAI-compatible function exports
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hexdsp_offload_arm
    )

    # Run qaic on the IDL file; adds hexdsp_offload_stub.c and hexdsp_offload.h
    # to the build as side-effects.
    build_idl(inc/hexdsp_offload.idl hexdsp_offload)

    include_directories(${common_incs})

    # Pick the right cdsprpc library for the target OS (Android vs Ubuntu ARM).
    choose_dsprpc(cdsp dsprpc)
    link_custom_library(hexdsp_offload ${dsprpc})
    link_custom_library(hexdsp_offload rpcmem)

    # Applies SDK-standard compile flags (PIC, warnings, …).
    set_common_compile_and_link_options(hexdsp_offload)

    install(TARGETS hexdsp_offload LIBRARY DESTINATION lib)

# ---------------------------------------------------------------------------
# DSP (Hexagon) side — skel shared library pushed to /vendor/lib/rfsa/adsp/.
# ---------------------------------------------------------------------------
else()

    # DSP_VERSION selects the Hexagon ISA revision; must match the hardware.
    # Known values: v73 (SM8550 era), v75, v79, v81.
    # For the DragonWing SA9000P confirm the exact version with:
    #   adb shell cat /sys/devices/soc0/soc_id
    # and cross-reference with Qualcomm product documentation.
    if(NOT DSP_VERSION)
        set(DSP_VERSION "v73")
        message(STATUS "DSP_VERSION not set; defaulting to ${DSP_VERSION}. "
                       "Override with -DDSP_VERSION=v75 etc.")
    endif()

    add_library(hexdsp_offload_skel SHARED
        # qaic-generated skel dispatch layer
        ${CMAKE_CURRENT_BINARY_DIR}/hexdsp_offload_skel
        # DSP-side compute implementation (student fills this in)
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hexdsp_offload_imp
    )

    build_idl(inc/hexdsp_offload.idl hexdsp_offload_skel)

    include_directories(${common_incs})

    # copy_binaries puts the .so into the SDK ship/ staging directory.
    copy_binaries(hexdsp_offload_skel)

endif()

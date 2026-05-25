# SPDX-License-Identifier: LicenseRef-CSSL-1.0
#
# CMakeLists.txt — Hexagon DSP LDPC decoder offload.
#
# Builds two distinct targets depending on build type:
#
#   OS_TYPE=HLOS (ARM cross-compile):
#     → libldpc_hexagon.so   ARM stub loaded by OAI via dlopen in place of libldpc.so
#
#   Default (Hexagon cross-compile):
#     → libldpc_hexagon_skel.so  DSP skeleton pushed to /vendor/lib/rfsa/adsp/
#
# ── ARM (HLOS) build ────────────────────────────────────────────────────────
#
#   cmake -B build-arm \
#       -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
#       -DOAI_SRC_ROOT=/path/to/openairinterface5g \
#       -DCMAKE_TOOLCHAIN_FILE=<oai>/cmake_targets/cross-arm-dragonwing.cmake \
#       -DOS_TYPE=HLOS \
#       <oai>/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_hexagon
#   cmake --build build-arm
#
# ── DSP (Hexagon) build ──────────────────────────────────────────────────────
#
#   cmake -B build-dsp \
#       -DHEXAGON_SDK_ROOT=/opt/Hexagon_SDK/6.4.0.2 \
#       -DOAI_SRC_ROOT=/path/to/openairinterface5g \
#       -DCMAKE_TOOLCHAIN_FILE=/opt/Hexagon_SDK/6.4.0.2/build/cmake/hexagon_toolchain.cmake \
#       -DDSP_VERSION=v73 \
#       <oai>/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_hexagon
#   cmake --build build-dsp
#
# ── Deploy ──────────────────────────────────────────────────────────────────
#
#   adb push build-arm/libldpc_hexagon.so          /data/oai/lib/
#   adb push build-dsp/libldpc_hexagon_skel.so     /vendor/lib/rfsa/adsp/
#
# ── OAI integration ─────────────────────────────────────────────────────────
#
#   OAI loads libldpc_hexagon.so instead of libldpc.so when the loader finds
#   it under the OAI lib directory.  The three exported symbols are identical:
#     LDPCinit(), LDPCshutdown(), LDPCdecoder()

cmake_minimum_required(VERSION 3.22)
project(ldpc_hexagon C)

# ---------------------------------------------------------------------------
# Hexagon SDK root
# ---------------------------------------------------------------------------
if(NOT HEXAGON_SDK_ROOT)
    set(HEXAGON_SDK_ROOT "/opt/Hexagon_SDK/6.4.0.2")
    message(STATUS "HEXAGON_SDK_ROOT not set; defaulting to ${HEXAGON_SDK_ROOT}")
endif()

# ---------------------------------------------------------------------------
# OAI source root — required for LDPC decoder headers
# ---------------------------------------------------------------------------
if(NOT OAI_SRC_ROOT)
    # Try to detect from our own path: this file lives at
    # <OAI_SRC_ROOT>/openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_hexagon
    get_filename_component(OAI_SRC_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
    message(STATUS "OAI_SRC_ROOT auto-detected as ${OAI_SRC_ROOT}")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

# ---------------------------------------------------------------------------
# Include paths shared by ARM stub and DSP skel
# ---------------------------------------------------------------------------
set(ldpc_dec_dir ${OAI_SRC_ROOT}/openair1/PHY/CODING/nrLDPC_decoder)

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/inc        # ldpc_hexagon.idl lives here
    ${CMAKE_CURRENT_BINARY_DIR}            # qaic writes generated .h here
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc
    ${ldpc_dec_dir}                        # nrLDPCdecoder_defs.h, nrLDPC_types.h, …
    ${OAI_SRC_ROOT}                        # for openair1/PHY/CODING/coding_defs.h
)

# ---------------------------------------------------------------------------
# ARM (HLOS) side — stub .so loaded by OAI via dlopen
# ---------------------------------------------------------------------------
if(${OS_TYPE} MATCHES "HLOS")

    add_library(ldpc_hexagon SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/ldpc_hexagon_stub
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ldpc_hexagon_arm
    )

    build_idl(inc/ldpc_hexagon.idl ldpc_hexagon)

    include_directories(${common_incs})

    choose_dsprpc(cdsp dsprpc)
    link_custom_library(ldpc_hexagon ${dsprpc})
    link_custom_library(ldpc_hexagon rpcmem)

    set_common_compile_and_link_options(ldpc_hexagon)

    install(TARGETS ldpc_hexagon LIBRARY DESTINATION lib)

# ---------------------------------------------------------------------------
# DSP (Hexagon) side — skel .so pushed to /vendor/lib/rfsa/adsp/
# ---------------------------------------------------------------------------
else()

    if(NOT DSP_VERSION)
        set(DSP_VERSION "v73")
        message(STATUS "DSP_VERSION not set; defaulting to ${DSP_VERSION}. "
                       "Check SA9000P ISA version with: "
                       "adb shell cat /sys/devices/soc0/soc_id")
    endif()

    add_library(ldpc_hexagon_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/ldpc_hexagon_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ldpc_hexagon_imp
    )

    build_idl(inc/ldpc_hexagon.idl ldpc_hexagon_skel)

    include_directories(${common_incs})

    # scalar-only baseline: no HVX flags needed
    # To add HVX: target_compile_options(ldpc_hexagon_skel PRIVATE -mhvx -mhvx-length=128B)

    copy_binaries(ldpc_hexagon_skel)

endif()

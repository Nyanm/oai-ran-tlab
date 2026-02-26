add_library(coding MODULE
            3gpplte_sse.c
            3gpplte.c
            3gpplte_turbo_decoder_sse_8bit.c
            3gpplte_turbo_decoder_sse_16bit.c
            3gpplte_turbo_decoder_avx2_16bit.c
            3gpplte_turbo_decoder.c
)
set_target_properties(coding PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(coding PRIVATE log_headers)

add_library(ldpc_orig MODULE
            nrLDPC_decoder/nrLDPC_decoder.c
            nrLDPC_encoder/ldpc_encoder.c
            nrLDPC_encoder/ldpc_encoder32.c
)
set_target_properties(ldpc_orig PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc_orig PRIVATE ldpc_segment ldpc_gen_HEADERS)

add_library(ldpc MODULE
            nrLDPC_decoder/nrLDPC_decoder.c
            nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
            nrLDPC_encoder/ldpc_encoder32.c
)
set_target_properties(ldpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc PRIVATE ldpc_segment ldpc_gen_HEADERS)

add_dependencies(ldpctest ldpc ldpc_orig)
add_dependencies(nr-softmodem ldpc ldpc_orig)
add_dependencies(nr-uesoftmodem ldpc ldpc_orig)
add_dependencies(nr_ulsim ldpc ldpc_orig)
add_dependencies(nr_ulschsim ldpc ldpc_orig)
add_dependencies(nr_dlsim ldpc ldpc_orig)
add_dependencies(nr_dlschsim ldpc ldpc_orig)

##############################################
# Base CUDA setting
##############################################

add_boolean_option(ENABLE_LDPC_CUDA OFF "Build support for CUDA" OFF)
if (ENABLE_LDPC_CUDA)
  enable_language(CUDA)

  # it would be possible to auto-detect compute capailities: https://stackoverflow.com/a/68223399
  #include(FindCUDA/select_compute_arch)
  #CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)

  # Nvidia recommends to build against all targeted archs: https://forums.developer.nvidia.com/t/261162
  # this is for cmake 3.16 (OAI minimum), starting at cmake 3.18 there is another syntax, see below

  SET(CUDA_VERBOSE_BUILD ON)

  # version using NVLink-C2C in Grace Hopper
  add_library(ldpc_cuda MODULE
                   nrLDPC_decoder/nrLDPC_decoder_cuda.c
                   nrLDPC_decoder/nrLDPC_decoder.c
		   nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
		   nrLDPC_decoder/nrLDPC_decoder_BG1_cuda.cu
		   nrLDPC_encoder/ldpc_encoder_cuda32.c
		   nrLDPC_encoder/ldpc_BG1_Zc384_32bit.cu
		   nrLDPC_encoder/ldpc_input.cu
		   nrLDPC_encoder/ldpc_input32.c
                   # The slot coding layer cannot be linked with
                   # target_link_libraries like above
                   # because of cuda_add_library
                   # which already uses target_link_libraries
                   #nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
                   nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder_cuda.c
                   nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
                   nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder_cuda.c
                   nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
                   nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.c
		   nrLDPC_coding/nrLDPC_coding_segment/ldpc_deinterleave.cu
		   nrLDPC_coding/nrLDPC_coding_segment/ldpc_rate_matching.cu
  )
  set_target_properties(ldpc_cuda PROPERTIES CUDA_ARCHITECTURES "80;87;90")
  # we need to find the right include directory, but ti seems that
  # enable_language(CUDA) does not make this available(?), so also use the CUDA package
  find_package(CUDA REQUIRED)
  target_include_directories(ldpc_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
  
  target_link_libraries(ldpc_cuda ldpc_gen_HEADERS ${T_LIB})
  set_target_properties(ldpc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(ldpc_cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  add_dependencies(ldpctest ldpc_cuda)
  add_dependencies(nr-softmodem ldpc_cuda)
  add_dependencies(nr-uesoftmodem ldpc_cuda)
  add_dependencies(nr_ulsim ldpc_cuda)
  add_dependencies(nr_ulschsim ldpc_cuda)
  add_dependencies(nr_dlsim ldpc_cuda)
  add_dependencies(nr_dlschsim ldpc_cuda)

  target_compile_definitions(ldpc_cuda PRIVATE USE_CUDA)

endif()
add_boolean_option(ENABLE_LDPC_HIP OFF "Build support for HIP (AMD ROCm)" OFF)

if (ENABLE_LDPC_HIP)
# ROCm typically installs HIP CMake modules under:
#   /opt/rocm/lib/cmake/hip
# or /opt/rocm/hip/cmake, depending on version.
#
# Prefer config package if available:
   find_package(HIP QUIET)

   if (NOT HIP_FOUND)
    # Fallback: try to find FindHIP.cmake
    # (You can also hardcode /opt/rocm if you want.)
      list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
      find_package(HIP REQUIRED)
   endif()
# If your ROCm provides hip_add_library, you need this include:
# (Some installs export it through HIP package, some require explicit include)
   if (EXISTS "/opt/rocm/lib/cmake/hip/hip-config.cmake")
# ok
   endif()

# You can set targets here (your AMD set)
# MI210 = gfx90a, Ryzen AI MAX+ 395 = gfx1151
   set(LDPC_HIP_ARCHS "gfx90a;gfx1151")

# HIP builds a MODULE similar to CUDA case
# Note: hipcc can compile .cu as HIP (nvcc-style extension) *or* you can rename to .cu
# Keep your C sources as-is; for the GPU sources, you will likely create HIP variants.
   add_library(ldpc_hip MODULE
	      nrLDPC_decoder/nrLDPC_decoder_cuda.c
              nrLDPC_decoder/nrLDPC_decoder.c
              nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
												    # HIP versions of your CUDA kernels (recommended to duplicate/rename)
												  	          nrLDPC_decoder/nrLDPC_decoder_BG1_hip.cu 
	      nrLDPC_encoder/ldpc_BG1_Zc384_32bit_hip.cu nrLDPC_encoder/ldpc_input_hip.cu
	      nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder_cuda.c
	      nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder.c
	      nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_decoder_cuda.c
	      nrLDPC_coding/nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
	      nrLDPC_coding/nrLDPC_coding_segment/nr_rate_matching.c
              nrLDPC_coding/nrLDPC_coding_segment/ldpc_deinterleave_hip.cu
	      nrLDPC_coding/nrLDPC_coding_segment/ldpc_rate_matching_hip.cu)
# Tell CMake to compile .cu sources with HIP
# If you keep .cu extensions, you can still force LANGUAGE HIP.
    set_source_files_properties(
	      nrLDPC_decoder/nrLDPC_decoder_BG1_hip.cu
              nrLDPC_encoder/ldpc_BG1_Zc384_32bit_hip.cu
	      nrLDPC_encoder/ldpc_input_hip.cu
              nrLDPC_coding/nrLDPC_coding_segment/ldpc_deinterleave_hip.cu
	      nrLDPC_coding/nrLDPC_coding_segment/ldpc_rate_matching_hip.cu
              PROPERTIES LANGUAGE HIP)

# Include dirs: HIP package usually provides HIP_INCLUDE_DIRS
# If not, /opt/rocm/include is standard.
    if (DEFINED HIP_INCLUDE_DIRS)
         target_include_directories(ldpc_hip PRIVATE ${HIP_INCLUDE_DIRS})
    else()
         target_include_directories(ldpc_hip PRIVATE /opt/rocm/include)
    endif()

# HIP arch flags for CMake 3.16:
# The portable way is to pass hipcc flags directly.
# These are understood by hipcc/clang for AMD targets.
    target_compile_options(ldpc_hip PRIVATE "--offload-arch=gfx90a" "--offload-arch=gfx1151")
    set_target_properties(ldpc_hip PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    add_dependencies(ldpctest ldpc_hip)
    add_dependencies(nr-softmodem ldpc_hip)
    add_dependencies(nr-uesoftmodem ldpc_hip)
    add_dependencies(nr_ulsim ldpc_hip)
    add_dependencies(nr_ulschsim ldpc_hip)
    add_dependencies(nr_dlsim ldpc_hip)
    add_dependencies(nr_dlschsim ldpc_hip)
    target_compile_definitions(ldpc_hip PRIVATE USE_HIP)
endif()
add_subdirectory(nrLDPC_coding)
add_subdirectory(nrLDPC_decoder)

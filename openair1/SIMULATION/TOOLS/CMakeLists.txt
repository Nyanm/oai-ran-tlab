# SPDX-License-Identifier: LicenseRef-CSSL-1.0

if(CUDA_ENABLE)
    add_library(oai_cuda_lib STATIC
        multipath_channel.cu
        phase_noise.cu
        channel_pipeline.cu
    )

    target_link_libraries(oai_cuda_lib PUBLIC CUDA::toolkit)

    target_include_directories(oai_cuda_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

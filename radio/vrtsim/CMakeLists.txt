add_library(noise_device STATIC noise_device.c)

set(VRTSIM_SOURCES vrtsim.c)
if(CUDA_ENABLE)
    list(APPEND VRTSIM_SOURCES vrtsim_channel_pipeline.cu)
endif()

add_library(vrtsim MODULE ${VRTSIM_SOURCES})
add_library(vrtsim_static STATIC ${VRTSIM_SOURCES})

set(VRTSIM_COMMON_LIBS PRIVATE SIMU shm_td_iq_channel actor noise_device)
target_link_libraries(vrtsim ${VRTSIM_COMMON_LIBS})
target_link_libraries(vrtsim_static ${VRTSIM_COMMON_LIBS})

set(VRTSIM_COMMON_INCLUDES PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../openair1/PHY/defs
)
target_include_directories(vrtsim ${VRTSIM_COMMON_INCLUDES})
target_include_directories(vrtsim_static ${VRTSIM_COMMON_INCLUDES})

set_target_properties(vrtsim PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_dependencies(vrtsim generate_T)

if(CUDA_ENABLE)
    set_target_properties(vrtsim vrtsim_static PROPERTIES 
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80"
    )
    target_link_libraries(vrtsim PRIVATE oai_cuda_lib)
    target_link_libraries(vrtsim_static PRIVATE oai_cuda_lib)
    target_compile_definitions(vrtsim PRIVATE ENABLE_CUDA)
    target_compile_definitions(vrtsim_static PRIVATE ENABLE_CUDA)
endif()

add_boolean_option(OAI_VRTSIM_TAPS_CLIENT OFF "Enable taps client" ON)
if (OAI_VRTSIM_TAPS_CLIENT)
  target_link_libraries(vrtsim PRIVATE taps_client)
  file(DOWNLOAD
    "https://gitlab.eurecom.fr/oai/raytracing-channel-emulator/-/raw/59589ad2a534881810b2aa7dd81e63066245bb8d/server/api/taps.fbs?inline=false"
    ${CMAKE_SOURCE_DIR}/taps.fbs
  )

  set(TAPS_IF_PATH ${CMAKE_SOURCE_DIR}/taps.fbs)
  set(TAPS_API_HEADER taps_generated.h)

  add_custom_command(OUTPUT ${TAPS_API_HEADER}
    COMMAND flatc -c ${TAPS_IF_PATH}
    DEPENDS ${TAPS_IF_PATH}
    COMMENT "Generating flatbuffers API from ${TAPS_IF_PATH}"
  )
  add_library(taps_api INTERFACE)
  target_include_directories(taps_api INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

  add_library(taps_client taps_client.cpp ${TAPS_API_HEADER})
  target_include_directories(taps_client PUBLIC .)
  find_package(Flatbuffers REQUIRED)
  target_link_libraries(taps_client PUBLIC flatbuffers taps_api SIMU nanomsg)
endif()

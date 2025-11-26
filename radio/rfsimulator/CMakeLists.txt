add_library(rfsimulator MODULE
            simulator.c
            apply_channelmod.c
            ../../openair1/PHY/TOOLS/signal_energy.c
)
target_link_libraries(rfsimulator PRIVATE SIMU log_headers)
set_target_properties(rfsimulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(replay_node stored_node.c)
target_link_libraries (replay_node minimal_lib)

# External taps client for channel emulation via socket interface
add_boolean_option(OAI_RFSIM_TAPS_CLIENT OFF "Enable external taps client for RFSim" ON)
if (OAI_RFSIM_TAPS_CLIENT)
  target_compile_definitions(rfsimulator PRIVATE ENABLE_TAPS_CLIENT=1)

  # Download FlatBuffers schema for taps interface
  file(DOWNLOAD
    "https://gitlab.eurecom.fr/oai/raytracing-channel-emulator/-/raw/59589ad2a534881810b2aa7dd81e63066245bb8d/server/api/taps.fbs?inline=false"
    ${CMAKE_SOURCE_DIR}/taps.fbs
  )

  set(TAPS_IF_PATH ${CMAKE_SOURCE_DIR}/taps.fbs)
  set(TAPS_API_HEADER taps_generated.h)

  # Generate FlatBuffers C++ header from schema
  add_custom_command(OUTPUT ${TAPS_API_HEADER}
    COMMAND flatc -c ${TAPS_IF_PATH}
    DEPENDS ${TAPS_IF_PATH}
    COMMENT "Generating flatbuffers API from ${TAPS_IF_PATH}"
  )

  add_library(taps_api INTERFACE)
  target_include_directories(taps_api INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

  # Build taps client library
  add_library(rfsim_taps_client ../vrtsim/taps_client.cpp ${TAPS_API_HEADER})
  target_include_directories(rfsim_taps_client PUBLIC ../vrtsim)
  
  # Handle different FlatBuffers package naming conventions across distributions
  find_package(FlatBuffers QUIET)
  if(NOT FlatBuffers_FOUND)
    find_package(Flatbuffers REQUIRED)
    set(FLATBUFFERS_TARGET flatbuffers)
  else()
    set(FLATBUFFERS_TARGET flatbuffers::flatbuffers)
  endif()
  
  target_link_libraries(rfsim_taps_client PUBLIC ${FLATBUFFERS_TARGET} taps_api SIMU nanomsg)
  target_link_libraries(rfsimulator PRIVATE rfsim_taps_client)
  
  message(STATUS "RFSim: External taps client support enabled")
endif()

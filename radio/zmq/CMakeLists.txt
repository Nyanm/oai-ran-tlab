# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_library(ring_buffer ring_buffer.cpp)
target_include_directories(ring_buffer PUBLIC ./)
add_library(zmq_radio SHARED zmq_radio.cpp)
target_link_libraries(zmq_radio PUBLIC zmq ring_buffer)
set_target_properties(zmq_radio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(zmq_radio_static STATIC zmq_radio.cpp)
target_link_libraries(zmq_radio_static PUBLIC zmq ring_buffer)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

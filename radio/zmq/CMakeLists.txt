
add_library(circular_buffer circular_buffer.cpp)
target_include_directories(circular_buffer PUBLIC ./)
add_library(zmq_radio SHARED zmq_radio.cpp)
target_link_libraries(zmq_radio PUBLIC zmq circular_buffer)
set_target_properties(zmq_radio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(zmq_radio_static STATIC zmq_radio.cpp)
target_link_libraries(zmq_radio_static PUBLIC zmq circular_buffer)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_executable(test_wraparound_buffer test_wraparound_buffer.cpp)
add_dependencies(tests test_wraparound_buffer)
target_link_libraries(test_wraparound_buffer PRIVATE wraparound_buffer pthread LOG minimal_lib GTest::gtest)
add_test(NAME test_wraparound_buffer
  COMMAND ./test_wraparound_buffer)

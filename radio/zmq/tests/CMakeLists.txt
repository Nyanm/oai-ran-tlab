add_executable(test_circular_buffer test_circular_buffer.cpp)
target_link_libraries(test_circular_buffer PRIVATE circular_buffer GTest::gtest)
add_dependencies(tests test_circular_buffer)
add_test(NAME test_circular_buffer
  COMMAND ./test_circular_buffer)

add_executable(test_zmq test_zmq.cpp)
target_link_libraries(test_zmq PRIVATE zmq_radio_static UTIL CONFIG_LIB GTest::gtest zmq)
add_dependencies(tests test_zmq)
add_test(NAME test_zmq
    COMMAND ./test_zmq)

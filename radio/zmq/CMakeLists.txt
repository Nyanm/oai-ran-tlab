add_library(oai_zmq MODULE zmq.c circular_buffer.c zmq_threads.c)
set_target_properties(oai_zmq PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(oai_zmq PRIVATE zmq)
add_dependencies(oai_zmq generate_T)

add_library(fd_rfsim MODULE fd_rfsim.c)
target_link_libraries(fd_rfsim PRIVATE atomic UTIL)
set_target_properties(fd_rfsim PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(NOT OAI_FHI72)
  add_custom_command(TARGET fd_rfsim POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E create_symlink libfd_rfsim.so liboai_transpro.so
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

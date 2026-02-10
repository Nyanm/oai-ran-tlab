##########################################################
# LDPC tee library - tee to two libraries
##########################################################

add_boolean_option(ENABLE_LDPC_TEE OFF "Build support for LDPC tee to two LDPC libraries" OFF)

if (ENABLE_LDPC_TEE)
  add_library(ldpc_tee MODULE nrLDPC_coding_tee.c)

  set_target_properties(ldpc_tee PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  add_dependencies(nr-softmodem ldpc_tee)
  add_dependencies(nr-uesoftmodem ldpc_tee)
  add_dependencies(nr_ulsim ldpc_tee)
  add_dependencies(nr_ulschsim ldpc_tee)
  add_dependencies(nr_dlsim ldpc_tee)
  add_dependencies(nr_dlschsim ldpc_tee)

endif()

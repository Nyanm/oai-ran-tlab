##########################################################

# LDPC coding library - ArmRAN Acceleration Library
##########################################################

add_boolean_option(ENABLE_LDPC_ARMRAL OFF "Build support for LDPC coding with the Arm RAN Acceleration Library" OFF)

if (ENABLE_LDPC_ARMRAL)

  find_package(armral REQUIRED)
  add_library(ldpc_armral MODULE
    ../nrLDPC_coding_segment/nr_rate_matching.c
    nrLDPC_coding_armral_decoder.c
    ../nrLDPC_coding_segment/nrLDPC_coding_segment_encoder.c
    ../../nrLDPC_encoder/ldpc_encoder_optim8segmulti.c
  )
  set_target_properties(ldpc_armral PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(ldpc_armral PRIVATE ldpc_gen_HEADERS)
  target_link_libraries(ldpc_armral PRIVATE armral)
  
  add_dependencies(ldpctest ldpc_armral)
  add_dependencies(nr-softmodem ldpc_armral)
  add_dependencies(nr-uesoftmodem ldpc_armral)
  add_dependencies(nr_ulsim ldpc_armral)
  add_dependencies(nr_ulschsim ldpc_armral)
  add_dependencies(nr_dlsim ldpc_armral)
  add_dependencies(nr_dlschsim ldpc_armral)
  
  #ensure that the T header files are generated before targets depending on them
  if (${T_TRACER})
    add_dependencies(ldpc_armral generate_T)
  endif (${T_TRACER})

endif()

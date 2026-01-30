add_library(xnap_lib OBJECT
            xnap_lib_common.c
            xnap_gNB_interface_management.c
)
target_link_libraries(xnap_lib PUBLIC ngap nr_rrc asn1_xnap ds)
target_include_directories(xnap_lib PUBLIC
                           ${OPENAIR2_DIR}/XNAP
                           ${CMAKE_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_TESTS)
  target_compile_definitions(xnap_lib PRIVATE ENABLE_TESTS)
endif()


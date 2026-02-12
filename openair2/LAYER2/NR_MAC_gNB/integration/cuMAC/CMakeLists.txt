if (OAI_AERIAL_CUMAC)
    message(STATUS "Building with support for Aerial cuMAC")
    add_library(aerial_cumac_lib
            cumac_msg.c
            cumac_msg_funcs.c
            cumac_nvipc.c
    )

    find_library(NVIPC_LIB nvipc REQUIRED)
    if (NOT NVIPC_LIB)
        message(FATAL_ERROR "-- nvipc library not found")
    else ()
        message(STATUS "  nvipc library found in ${NVIPC_LIB}")
    endif ()

    find_library(NVLOG_LIB nvlog REQUIRED)
    if (NOT NVLOG_LIB)
        message(FATAL_ERROR "-- nvlog library not found")
    else ()
        message(STATUS "  nvlog library found in ${NVLOG_LIB}")
    endif ()
    #target_compile_definitions(aerial_cumac_lib PUBLIC ENABLE_L2_SLT_RSP)
    target_link_libraries(aerial_cumac_lib PRIVATE asn1_lte_rrc_hdrs asn1_nr_rrc_hdrs)
    #target_link_libraries(aerial_cumac_lib PRIVATE nr_fapi_p5 nr_fapi_p7)
    target_link_libraries(aerial_cumac_lib PRIVATE "${NVLOG_LIB}" "${NVIPC_LIB}")
    find_package(CUDA REQUIRED)
    target_link_libraries(aerial_cumac_lib PUBLIC cudart)

    #add_executable(cumac_test
    #        cumac_test.c
    #        cumac_msg_funcs.c
    #        cumac_msg.c)
    #target_link_libraries(cumac_test PRIVATE "${NVLOG_LIB}" "${NVIPC_LIB}")
    ##target_link_libraries(cumac_test PRIVATE asn1_lte_rrc_hdrs asn1_nr_rrc_hdrs)
    #target_link_libraries(cumac_test PRIVATE nr_fapi_p5 nr_fapi_p7)
    #target_link_libraries(cumac_test PRIVATE  utils pthread UTIL ${T_LIB} minimal_lib )
    ##flag for nvIPC
    #target_compile_definitions(cumac_test PUBLIC ENABLE_L2_SLT_RSP)
    ## This makes it so cumac_test is always compiled whenever nr-softmodem is compiled
    #add_dependencies(nr-softmodem cumac_test )

else ()
    message(STATUS "No Support for Aerial cuMAC")
endif ()

add_boolean_option(OAI_AW2SORI OFF "Activate OAI's AW2S driver" OFF)
if(OAI_AW2SORI)
  add_subdirectory(AW2SORI)
endif()

add_boolean_option(OAI_ETHERNET OFF "Activate OAI's Ethernet transport driver" OFF)
if(OAI_ETHERNET)
  add_subdirectory(ETHERNET)
endif()

add_boolean_option(OAI_BLADERF OFF "Activate OAI's BladeRF driver" OFF)
if(OAI_BLADERF)
  add_subdirectory(BLADERF)
endif()

set(E_VERSION 5.1.6)
set(F_VERSION 6.1.5)
add_boolean_option(OAI_FHI72 OFF "Activate OAI's FHI 7.2 (xran/fhi_lib) driver" OFF)
if (OAI_FHI72)
  set(XRAN_E_RELEASE_EOL 91e7030cf8adefce502f0701fb7aaa1d512379df)
  set(XRAN_F_RELEASE_EOL 91e7030cf8adefce502f0701fb7aaa1d512379df)
  find_package(xran REQUIRED)
  if(xran_VERSION VERSION_EQUAL E_VERSION)
    if(DEFINED XRAN_E_RELEASE_EOL)
      message(WARNING "Support XRAN release E is being eclipsed. Consider upgrading to release K. Replacing radio/fhi_72 directory with contents from commit ${XRAN_E_RELEASE_EOL}.")
      execute_process(
        COMMAND git checkout ${XRAN_E_RELEASE_EOL} -- radio/fhi_72/
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_CHECKOUT_RESULT
      )
      if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to checkout fhi_72 directory from git")
      endif()
    endif()
  elseif(xran_VERSION VERSION_EQUAL F_VERSION)
    if(DEFINED XRAN_F_RELEASE_EOL)
      message(WARNING "Support XRAN release F is being eclipsed. Consider upgrading to release K. Replacing radio/fhi_72 directory with contents from commit ${XRAN_F_RELEASE_EOL}.")
      execute_process(
        COMMAND git checkout ${XRAN_F_RELEASE_EOL} -- radio/fhi_72/
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_CHECKOUT_RESULT
      )
      if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to checkout fhi_72 directory from git")
      endif()
    endif()
  endif()
  add_subdirectory(fhi_72)
endif()

add_boolean_option(OAI_IRIS OFF "Activate OAI's IRIS/SoapySDR driver" OFF)
if(OAI_IRIS)
  add_subdirectory(IRIS)
endif()

add_boolean_option(OAI_LMSSDR OFF "Activate OAI's LimeSDR driver" OFF)
if(OAI_LMSSDR)
  add_subdirectory(LMSSDR)
endif()

add_boolean_option(OAI_SIMU ON "Activate OAI's rfsimulator driver" OFF)
if(OAI_SIMU)
  add_subdirectory(rfsimulator)
endif()

add_boolean_option(OAI_USRP OFF "Activate OAI's USRP driver" OFF)
if(OAI_USRP)
  add_subdirectory(USRP)
endif()

add_boolean_option(OAI_VRTSIM ON "Activate OAI's shared memory radio driver" OFF)
if(OAI_VRTSIM)
  add_subdirectory(vrtsim)
endif()

add_boolean_option(OAI_RF_EMULATOR ON "Activate OAI's RF emulator" OFF)
if(OAI_RF_EMULATOR)
  add_subdirectory(emulator)
endif()

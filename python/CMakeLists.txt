#/*
# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about the OpenAirInterface (OAI) Software Alliance:
# *      contact@openairinterface.org
# */

cmake_minimum_required (VERSION 3.19)
project (OpenAirInterface LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(OAI_VERSION 2.4.0)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

add_library(liboaipy MODULE
    src/oaipylib_module.c
    src/oaipylib_wrappers.c
    src/oaipylib_top.c
    ${NFAPI_USER_DIR}/nfapi.c
    ${NFAPI_USER_DIR}/gnb_ind_vars.c
    ${PHY_INTERFACE_DIR}/queue_t.c
)

target_include_directories(liboaipy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(liboaipy PRIVATE ${Python3_INCLUDE_DIRS})

target_link_libraries(liboaipy PRIVATE
	-Wl,--start-group UTIL SIMU PHY_COMMON PHY_NR_COMMON PHY_NR PHY_NR_UE SCHED_NR_LIB SCHED_NR_UE_LIB MAC_UE_NR MAC_NR_COMMON CONFIG_LIB L2_NR -Wl,--end-group
  m pthread ${T_LIB} ITTI dl nr_ue_phy_meas physim_common
  softmodem_common
    Python3::Module
)

set_target_properties(liboaipy PROPERTIES
    PREFIX ""
    OUTPUT_NAME "liboaipy"
)

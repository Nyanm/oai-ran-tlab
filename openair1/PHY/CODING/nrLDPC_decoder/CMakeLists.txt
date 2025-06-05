cmake_minimum_required(VERSION 3.10)
project(ldpc_decoder_test C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(ldpc_decoder_test
  nrLDPC_decoder.c
)

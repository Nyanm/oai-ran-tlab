add_executable(test_adcdac_utils test_adcdac_utils.cpp)
target_link_libraries(test_adcdac_utils PRIVATE adcdac_utils GTest::gtest)
add_dependencies(tests test_adcdac_utils)
add_test(NAME test_adcdac_utils
  COMMAND ./test_adcdac_utils)

add_executable(bench_adcdac_utils bench_adcdac_utils.cpp)
target_link_libraries(bench_adcdac_utils PRIVATE adcdac_utils benchmark::benchmark)
target_compile_options(bench_adcdac_utils PRIVATE -Wno-error=maybe-uninitialized)
add_dependencies(tests bench_adcdac_utils)

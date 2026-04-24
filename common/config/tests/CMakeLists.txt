# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_config_cmdline test_config_cmdline.cpp)
target_link_libraries(test_config_cmdline PRIVATE CONFIG_LIB GTest::gtest)

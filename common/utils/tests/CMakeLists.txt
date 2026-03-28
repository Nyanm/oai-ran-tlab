# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_tpool_vs_actors test_tpool_vs_actors.c)
target_link_libraries(test_tpool_vs_actors PRIVATE thread-pool pthread LOG minimal_lib actor)

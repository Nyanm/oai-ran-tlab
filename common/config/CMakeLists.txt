add_library(CONFIG_LIB
            config_common.c
            config_load_configmodule.c
            config_userapi.c
            config_cmdline.c
)
target_link_libraries(CONFIG_LIB PRIVATE dl LOG)
target_link_libraries(CONFIG_LIB PUBLIC ${T_LIB}) # TODO: remove: was provided through UTILS
target_link_libraries(CONFIG_LIB PUBLIC log_headers)

# CONFIG_LIB is a wrapper around a particular config implementation, default is
# libconfig in params_libconfig below
add_subdirectory(libconfig)
add_subdirectory(yaml)

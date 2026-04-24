/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"
extern "C" {
#include "common/config/config_userapi.h"
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (assert) {
    abort();
  } else {
    exit(EXIT_SUCCESS);
  }
}
}

configmodule_interface_t *uniqCfg;

TEST(cmdline, cmdline_new_array)
{
  char *array[2] = {strdup("prefix.[0].test"), strdup("1")};
  if ((uniqCfg = load_configmodule(2, array, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("");
  }
  paramdef_t params[] = {UINT16PARAM("test", "", 0, NULL, 0)};
  paramlist_def_t list = {"prefix", NULL, 0};
  config_getlist(uniqCfg, &list, params, sizeofArray(params), NULL);
  EXPECT_EQ(list.numelt, 1);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

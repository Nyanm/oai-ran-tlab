#ifndef RAN_FUNC_RC_CTRL_SLICE_H
#define RAN_FUNC_RC_CTRL_SLICE_H

#include <stdint.h>
#include <stdlib.h>

#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/lst_ran_param.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_param_list.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_parameter_value.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/ie/ir/ran_parameter_value_type.h"
#include "openair2/E2AP/flexric/src/sm/slice_sm/ie/slice_data_ie.h"
#include "openair2/E2AP/flexric/src/sm/slice_sm/ie/slice_data_ie.h"
#include "openair2/E2AP/flexric/src/lib/sm/ie/ue_id.h"

typedef enum {
  RRM_Policy_Ratio_List_8_4_3_6 = 1,
  RRM_Policy_Ratio_Group_8_4_3_6 = 2,
  RRM_Policy_8_4_3_6 = 3,
  RRM_Policy_Member_List_8_4_3_6 = 5,
  RRM_Policy_Member_8_4_3_6 = 6,
  PLMN_Identity_8_4_3_6 = 7,
  S_NSSAI_8_4_3_6 = 8,
  SST_8_4_3_6 = 9,
  SD_8_4_3_6 = 10,
  Min_PRB_Policy_Ratio_8_4_3_6 = 11,
  Max_PRB_Policy_Ratio_8_4_3_6 = 12,
  Dedicated_PRB_Policy_Ratio_8_4_3_6 = 13,
} slice_level_PRB_quota_param_id_e;

bool add_mod_rc_slice(int mod_id, size_t slices_len, ran_param_list_t* lst);


#endif // RAN_FUNC_RC_CTRL_SLICE_H

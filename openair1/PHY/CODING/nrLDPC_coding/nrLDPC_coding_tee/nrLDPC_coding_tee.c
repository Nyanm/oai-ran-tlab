/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_tee/nrLDPC_coding_tee.c
 * \brief LDPC coding library acting as a tee between two LDPC libraries.
 *        It distributes the TBs between the two libraries according to some criterion.
 */

#include "common/config/config_userapi.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"

nrLDPC_coding_interface_t nrLDPC_coding_interface_default, nrLDPC_coding_interface_offload;

bool choose_offload_decode(nrLDPC_TB_decoding_parameters_t *TB)
{
  return TB->Z >= 128 && TB->BG == 1;
}

int32_t nrLDPC_coding_decoder(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  uint8_t nb_tb = nrLDPC_slot_decoding_parameters->nb_TBs;
  nrLDPC_TB_decoding_parameters_t *TBs = nrLDPC_slot_decoding_parameters->TBs;

  // split and count TBs
  nrLDPC_TB_decoding_parameters_t TBs_default[nb_tb];
  uint8_t tb_id_default = 0;
  nrLDPC_TB_decoding_parameters_t TBs_offload[nb_tb];
  uint8_t tb_id_offload = 0;
  for (uint8_t tb_id = 0; tb_id < nb_tb; tb_id++) {
    if (choose_offload_decode(&TBs[tb_id])) {
      TBs_offload[tb_id_offload] = TBs[tb_id];
      tb_id_offload++;
    } else {
      TBs_default[tb_id_default] = TBs[tb_id];
      tb_id_default++;
    }
  }

  // TODO parallelize
  int ret_decoder = 0;
  if (tb_id_default > 0) {
    nrLDPC_slot_decoding_parameters_t slot_parameters_default = {.frame = nrLDPC_slot_decoding_parameters->frame,
                                                                 .slot = nrLDPC_slot_decoding_parameters->slot,
                                                                 .nb_TBs = tb_id_default,
                                                                 .threadPool = nrLDPC_slot_decoding_parameters->threadPool,
                                                                 .TBs = TBs_default};
    int ret_default = nrLDPC_coding_interface_default.nrLDPC_coding_decoder(&slot_parameters_default);
    ret_decoder = ret_default == 0 ? ret_decoder : ret_default;
  }

  if (tb_id_offload > 0) {
    nrLDPC_slot_decoding_parameters_t slot_parameters_offload = {.frame = nrLDPC_slot_decoding_parameters->frame,
                                                                 .slot = nrLDPC_slot_decoding_parameters->slot,
                                                                 .nb_TBs = tb_id_offload,
                                                                 .threadPool = nrLDPC_slot_decoding_parameters->threadPool,
                                                                 .TBs = TBs_offload};
    int ret_offload = nrLDPC_coding_interface_offload.nrLDPC_coding_decoder(&slot_parameters_offload);
    ret_decoder = ret_offload == 0 ? ret_decoder : ret_offload;
  }
  return ret_decoder;
}

bool choose_offload_encode(nrLDPC_TB_encoding_parameters_t *TB)
{
  return TB->BG == 1 && TB->C > 8 && TB->Z == 384;
}

int32_t nrLDPC_coding_encoder(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  uint8_t nb_tb = nrLDPC_slot_encoding_parameters->nb_TBs;
  nrLDPC_TB_encoding_parameters_t *TBs = nrLDPC_slot_encoding_parameters->TBs;

  // split and count TBs
  nrLDPC_TB_encoding_parameters_t TBs_default[nb_tb];
  uint8_t tb_id_default = 0;
  nrLDPC_TB_encoding_parameters_t TBs_offload[nb_tb];
  uint8_t tb_id_offload = 0;
  for (uint8_t tb_id = 0; tb_id < nb_tb; tb_id++) {
    if (choose_offload_encode(&TBs[tb_id])) {
      TBs_offload[tb_id_offload] = TBs[tb_id];
      tb_id_offload++;
    } else {
      TBs_default[tb_id_default] = TBs[tb_id];
      tb_id_default++;
    }
  }

  // TODO parallelize
  int ret_encoder = 0;
  if (tb_id_default > 0) {
    nrLDPC_slot_encoding_parameters_t slot_parameters_default = {.frame = nrLDPC_slot_encoding_parameters->frame,
                                                                 .slot = nrLDPC_slot_encoding_parameters->slot,
                                                                 .nb_TBs = tb_id_default,
                                                                 .threadPool = nrLDPC_slot_encoding_parameters->threadPool,
                                                                 .TBs = TBs_default};
    int ret_default = nrLDPC_coding_interface_default.nrLDPC_coding_encoder(&slot_parameters_default);
    ret_encoder = ret_default == 0 ? ret_encoder : ret_default;
  }

  if (tb_id_offload > 0) {
    nrLDPC_slot_encoding_parameters_t slot_parameters_offload = {.frame = nrLDPC_slot_encoding_parameters->frame,
                                                                 .slot = nrLDPC_slot_encoding_parameters->slot,
                                                                 .nb_TBs = tb_id_offload,
                                                                 .threadPool = nrLDPC_slot_encoding_parameters->threadPool,
                                                                 .TBs = TBs_offload};
    int ret_offload = nrLDPC_coding_interface_offload.nrLDPC_coding_encoder(&slot_parameters_offload);
    ret_encoder = ret_offload == 0 ? ret_encoder : ret_offload;
  }
  return ret_encoder;
}

int32_t nrLDPC_coding_init(int max_num_pxsch)
{
  int ret_init = 0;

  // load LDPC default library
  // First query configmodule to know which default library was provided
  char *shlibversion_default = NULL;
  // clang-format off
  paramdef_t LoaderParams_default[] = {
    {"shlibversion", NULL, 0, .strptr = &shlibversion_default, .defstrval = "", TYPE_STRING, 0, NULL}
  };
  // clang-format on
  char *cfgprefix_default = "nrLDPC_coding_tee.default";
  int ret_cfgmodule = config_get(config_get_if(), LoaderParams_default, sizeofArray(LoaderParams_default), cfgprefix_default);
  if (ret_cfgmodule < 0) {
    fprintf(stderr, "[LOADER]  %s %d couldn't retrieve config from section %s\n", __FILE__, __LINE__, cfgprefix_default);
  }
  // load
  int ret_loader = load_nrLDPC_coding_interface(shlibversion_default, &nrLDPC_coding_interface_default, max_num_pxsch);
  ret_init = ret_loader == 0 ? ret_init : ret_loader;

  // load LDPC offload library
  // First query configmodule to know which offload library was provided
  char *shlibversion_offload = NULL;
  // clang-format off
  paramdef_t LoaderParams_offload[] = {
    {"shlibversion", NULL, 0, .strptr = &shlibversion_offload, .defstrval = "", TYPE_STRING, 0, NULL}
  };
  // clang-format on
  char *cfgprefix_offload = "nrLDPC_coding_tee.offload";
  ret_cfgmodule = config_get(config_get_if(), LoaderParams_offload, sizeofArray(LoaderParams_offload), cfgprefix_offload);
  if (ret_cfgmodule < 0) {
    fprintf(stderr, "[LOADER]  %s %d couldn't retrieve config from section %s\n", __FILE__, __LINE__, cfgprefix_offload);
  }
  // load
  ret_loader = load_nrLDPC_coding_interface(shlibversion_offload, &nrLDPC_coding_interface_offload, max_num_pxsch);
  ret_init = ret_loader == 0 ? ret_init : ret_loader;

  return ret_init;
}

int32_t nrLDPC_coding_shutdown(void)
{
  int ret_shutdown = 0;

  int ret_default = nrLDPC_coding_interface_default.nrLDPC_coding_shutdown();
  ret_shutdown = ret_default == 0 ? ret_shutdown : ret_default;

  int ret_offload = nrLDPC_coding_interface_offload.nrLDPC_coding_shutdown();
  ret_shutdown = ret_offload == 0 ? ret_shutdown : ret_offload;

  return ret_shutdown;
}

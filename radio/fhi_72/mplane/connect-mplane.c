/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

#include "connect-mplane.h"
#include "init-mplane.h"
#include "common/utils/assertions.h"

#include <libyang/libyang.h>
#include <nc_client.h>

#define CLI_CH_TIMEOUT 60

static int my_auth_hostkey_check(const char *hostname, ssh_session session, void *priv)
{
  (void)hostname;
  (void)session;
  (void)priv;

  return 0;
}

bool listen_mplane(ru_session_t *ru_session)
{
  int port = NC_PORT_CH_SSH;
  char *host = "0.0.0.0";       // better IPv4
  int timeout = CLI_CH_TIMEOUT;

  /* create the session */
  nc_client_ssh_ch_set_username(ru_session->username);
  nc_client_ssh_ch_add_bind_listen(host, port);

  nc_client_ssh_ch_set_auth_pref(NC_SSH_AUTH_PASSWORD, 1);
  nc_client_ssh_ch_set_auth_pref(NC_SSH_AUTH_PUBLICKEY, -1);  // ssh-key identification
  nc_client_ssh_ch_set_auth_pref(NC_SSH_AUTH_INTERACTIVE, -1);

  //int keypair_ret = nc_client_ssh_ch_add_keypair(pub_key, priv_key);
  //assert(keypair_ret == 0 && "Unable to authenticate RU\n");
  //nc_client_ssh_ch_set_auth_hostkey_check_clb(my_auth_hostkey_check, "DATA");  // host-key identification

  printf("Waiting %ds for a SSH Call Home connection on port %u...\n", timeout, port);

  struct nc_session *ru_session_ptr = NULL;
  int ret = nc_accept_callhome(timeout * 1000, NULL, &ru_session_ptr); // check the right session; maybe session[0] is already ongoing, and need to create session[1]
  assert(ret == 1 && "SSH Call Home failed.");

  ru_session->session = (void *)ru_session_ptr;
  nc_client_ssh_ch_del_bind(host, port);

  //const char *ru_ip_add = nc_session_get_host(ru_session->session);
  //ru_session->ru_ip_add = malloc(strlen(ru_ip_add) + 1);
  //memcpy(ru_session->ru_ip_add, ru_ip_add, strlen(ru_ip_add) + 1);
  printf("Successfuly connected to RU\n"); // with IP address %s\n", ru_session->ru_ip_add);

  return true;
}

bool connect_mplane(ru_session_t *ru_session)
{
  int port = NC_PORT_SSH;

  nc_client_ssh_set_username(ru_session->username);

  nc_client_ssh_set_auth_pref(NC_SSH_AUTH_PUBLICKEY, 1);  // ssh-key identification
  nc_client_ssh_set_auth_pref(NC_SSH_AUTH_PASSWORD, -1);
  nc_client_ssh_set_auth_pref(NC_SSH_AUTH_INTERACTIVE, -1);

  nc_client_ssh_set_auth_hostkey_check_clb(my_auth_hostkey_check, "DATA");  // host-key identification

  MP_LOG_I("RPC request to RU \"%s\" = <connect> with username \"%s\" and port ID \"%d\".\n", ru_session->ru_ip_add, ru_session->username, port);
  ru_session->session = nc_connect_ssh(ru_session->ru_ip_add, port, NULL);
  AssertError(ru_session->session != NULL, return false, "[MPLANE] RU IP address %s unreachable. Maybe M-plane is disabled on the RU?\n", ru_session->ru_ip_add);

  MP_LOG_I("Successfuly connected to RU \"%s\" with username \"%s\" and port ID \"%d\".\n", ru_session->ru_ip_add, ru_session->username, port);

  return true;
}

void disconnect_mplane(void *rus_disconnect)
{
  ru_session_list_t *ru_session_list = (ru_session_list_t *)rus_disconnect;

  for (size_t i = 0; i <ru_session_list->num_rus; i++) {
    ru_session_t *ru_session = &ru_session_list->ru_session[i];
    if (ru_session->session == NULL)
      continue;
    MP_LOG_I("Sending PM de-activation request for RU \"%s\".\n", ru_session->ru_ip_add);
    bool success = pm_conf(ru_session, "false");
    if (success)
      MP_LOG_I("Successfully de-activated PM for RU \"%s\".\n", ru_session->ru_ip_add);
    MP_LOG_I("Disconnecting from RU \"%s\".\n", ru_session->ru_ip_add);
    nc_session_free(ru_session->session, NULL);
    ru_session->session = NULL;
#ifdef MPLANE_V1
    ly_ctx_destroy((struct ly_ctx *)ru_session->ctx, NULL);
#elif defined MPLANE_V2
    ly_ctx_destroy((struct ly_ctx *)ru_session->ctx);
#endif
  }

  nc_client_destroy();
}

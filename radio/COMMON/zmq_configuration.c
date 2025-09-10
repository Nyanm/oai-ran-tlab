#include "radio/zmq/zmq_configuration.h"

#include "common/utils/LOG/log.h"
#include "common/config/config_userapi.h"

zmq_configuration_t *get_zmq_configuration(void)
{
  /* nothing to configure if zmq not selected */
  uint32_t use_zmq = 0;

  paramdef_t zmq_enabled_params[] = ZMQ_ENABLED_PARAMS_DESC;

  config_process_cmdline(config_get_if(), zmq_enabled_params, sizeofArray(zmq_enabled_params), NULL);

  if (!use_zmq)
    return 0;

  char *rx_mode = 0;
  char *tx_mode = 0;
  char *rx_address = 0;
  char *tx_address = 0;

  paramdef_t zmq_params[] = ZMQ_PARAMS_DESC;

  config_get(config_get_if(), zmq_params, sizeofArray(zmq_params), ZMQ_SECTION);

  AssertFatal(rx_mode && tx_mode && rx_address && tx_address,
              "ZeroMQ error: you must provide rx-mode, tx-mode, rx-address and tx-address\n");

  zmq_connection_mode_t zmq_rx_mode;
  if (!strcmp(rx_mode, "connect"))
    zmq_rx_mode = ZMQ_CONNECT;
  else if (!strcmp(rx_mode, "bind"))
    zmq_rx_mode = ZMQ_BIND;
  else
    AssertFatal(0, "zmq rx-mode must be \"connect\" or \"bind\"\n");

  zmq_connection_mode_t zmq_tx_mode;
  if (!strcmp(tx_mode, "connect"))
    zmq_tx_mode = ZMQ_CONNECT;
  else if (!strcmp(tx_mode, "bind"))
    zmq_tx_mode = ZMQ_BIND;
  else
    AssertFatal(0, "zmq tx-mode must be \"connect\" or \"bind\"\n");

  zmq_configuration_t *ret = calloc_or_fail(1, sizeof(*ret));

  LOG_I(HW, "ZeroMQ configuration:\n");
  LOG_I(HW, "  rx-mode: %s\n", rx_mode);
  LOG_I(HW, "  tx-mode: %s\n", tx_mode);
  LOG_I(HW, "  rx-address: %s\n", rx_address);
  LOG_I(HW, "  tx-address: %s\n", tx_address);

  ret->rx_mode = zmq_rx_mode;
  ret->tx_mode = zmq_tx_mode;
  ret->rx_address = rx_address;
  ret->tx_address = tx_address;

  return ret;
}

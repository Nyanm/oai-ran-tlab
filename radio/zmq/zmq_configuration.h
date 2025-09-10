#ifndef RADIO_ZMQ_CONFIGURATION_H
#define RADIO_ZMQ_CONFIGURATION_H

#define ZMQ_SECTION "device.zmq"

#define ZMQ_ENABLED_PARAMS_DESC { \
  { "zmq", "use ZeroMQ to input/output radio data\n", PARAMFLAG_BOOL, .uptr = &use_zmq, .defintval = 0, TYPE_UINT, 0 } \
}

#define ZMQ_PARAMS_DESC { \
  { "rx-mode", "ZeroMQ RX port mode (\"connect\" or \"bind\")\n", 0, .strptr = &rx_mode, .defstrval = 0, TYPE_STRING, 0 }, \
  { "tx-mode", "ZeroMQ TX port mode (\"connect\" or \"bind\")\n", 0, .strptr = &tx_mode, .defstrval = 0, TYPE_STRING, 0 }, \
  { "rx-address", "ZeroMQ RX address (for example \"tcp://localhost:5555\"\n", 0, .strptr = &rx_address, .defstrval = 0, TYPE_STRING, 0 }, \
  { "tx-address", "ZeroMQ TX address (for example \"tcp://localhost:5555\"\n", 0, .strptr = &tx_address, .defstrval = 0, TYPE_STRING, 0 }, \
}

typedef enum {
  ZMQ_CONNECT,
  ZMQ_BIND
} zmq_connection_mode_t;

typedef struct {
  zmq_connection_mode_t rx_mode;
  zmq_connection_mode_t tx_mode;
  char *rx_address;
  char *tx_address;
} zmq_configuration_t;

zmq_configuration_t *get_zmq_configuration(void);

#endif /* RADIO_ZMQ_CONFIGURATION_H */

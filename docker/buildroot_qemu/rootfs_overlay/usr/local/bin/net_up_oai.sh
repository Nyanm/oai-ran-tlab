#!/bin/sh
set -eu

# Bring link up (harmless if already up)
ip link set dev eth0 up || true

# Make it idempotent:
ip addr replace 192.168.71.140/26 dev eth0
ip route replace default via 192.168.71.141 dev eth0

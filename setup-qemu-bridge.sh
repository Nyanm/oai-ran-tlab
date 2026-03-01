#!/bin/bash
set -euo pipefail

#note this script should be launched inside the oran-build (or oai-gnb) 
# container that launches QEMU later

echo "[*] Detecting container network configuration..."

# Detect container IP and gateway as assigned by Docker
IPCIDR=$(ip -4 -o addr show dev eth0 | awk '{print $4}' | head -n1)
GW=$(ip route show default | awk '{print $3}' | head -n1)

if [[ -z "$IPCIDR" || -z "$GW" ]]; then
  echo "ERROR: Could not detect IP or gateway on eth0"
  exit 1
fi

echo "    Container IP: $IPCIDR"
echo "    Gateway     : $GW"

echo "[*] Creating bridge br0 (if needed)..."
ip link show br0 >/dev/null 2>&1 || ip link add br0 type bridge
ip link set br0 up

echo "[*] Creating tap interface tap0 (if needed)..."
ip link show tap0 >/dev/null 2>&1 || ip tuntap add dev tap0 mode tap user root
ip link set tap0 up

echo "[*] Preparing eth0 for bridging..."
ip link set eth0 promisc on

echo "[*] Attaching eth0 and tap0 to br0..."
ip link set eth0 master br0
ip link set tap0 master br0
ip link set eth0 up

echo "[*] Moving container IP from eth0 to br0..."
ip addr flush dev eth0
ip addr add "$IPCIDR" dev br0

echo "[*] Replacing default route..."
ip route del default || true
ip route add default via "$GW" dev br0

echo "[✓] Bridge setup complete."
echo "    br0  <- eth0 + tap0"
echo "    You may now start QEMU using tap0."


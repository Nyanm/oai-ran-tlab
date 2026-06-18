
#!/usr/bin/env bash
set -euo pipefail

IFACE="enp2s0f0np0"
IP="192.168.40.1/24"
RMEM_MAX=33554432
WMEM_MAX=33554432
MTU=9000

echo "=== Configuring $IFACE for 10 GigE ==="

# Set IP address
echo "[1/3] Setting IP address $IP on $IFACE..."
sudo ip addr flush dev "$IFACE"
sudo ip addr add "$IP" dev "$IFACE"
sudo ip link set "$IFACE" up

# Set MTU
echo "[2/3] Setting MTU to $MTU..."
sudo ip link set "$IFACE" mtu "$MTU"

# Set socket buffers
echo "[3/3] Configuring socket buffers..."
sudo sysctl -w net.core.rmem_max="$RMEM_MAX"
sudo sysctl -w net.core.wmem_max="$WMEM_MAX"

# Persist socket buffer settings
SYSCTL_CONF="/etc/sysctl.conf"
for KEY in net.core.rmem_max net.core.wmem_max; do
    VAL="${RMEM_MAX}"
    if grep -q "^${KEY}" "$SYSCTL_CONF" 2>/dev/null; then
        sudo sed -i "s/^${KEY}.*/${KEY}=${VAL}/" "$SYSCTL_CONF"
    else
        echo "${KEY}=${VAL}" | sudo tee -a "$SYSCTL_CONF" > /dev/null
    fi
done

echo ""
echo "=== Done. Current interface state: ==="
ip addr show "$IFACE"
echo ""
echo "MTU: $(cat /sys/class/net/$IFACE/mtu)"
echo "rmem_max: $(sysctl -n net.core.rmem_max)"
echo "wmem_max: $(sysctl -n net.core.wmem_max)"
#!/usr/bin/env bash
set -euo pipefail

PREFIX=/opt/oai-gnb
CONFIGFILE="$PREFIX/etc/gnb.conf"
if [[ ! -f "$CONFIGFILE" ]]; then
  YAML_CONFIGFILE="$PREFIX/etc/gnb.yaml"
  if [[ ! -f "$YAML_CONFIGFILE" ]]; then
    echo "No $CONFIGFILE or $YAML_CONFIGFILE in container (9p export)."
    exit 255
  fi
  CONFIGFILE="$YAML_CONFIGFILE"
fi

echo "[*] Selected config in container: $CONFIGFILE"

VM_CONFIG="/mnt/oai-gnb${CONFIGFILE#/opt/oai-gnb}"

SETUP_SCRIPT="./setup-qemu-bridge.sh"
QEMU_SCRIPT="./start-qemu-tap-aarch64.sh"

echo "======================================"
echo " OAI QEMU Container Entrypoint"
echo "======================================"

# Ensure required tools exist
command -v ip >/dev/null || { echo "ERROR: iproute2 not installed"; exit 1; }
command -v qemu-system-aarch64 >/dev/null || { echo "ERROR: qemu-system-aarch64 not installed"; exit 1; }

# Ensure /dev/net/tun exists
if [[ ! -e /dev/net/tun ]]; then
  echo "ERROR: /dev/net/tun not available. Run container with:"
  echo "  --privileged"
  echo "  OR cap_add: NET_ADMIN + device: /dev/net/tun"
  exit 1
fi

# Run bridge setup if enabled
if [[ "${SETUP_BRIDGE:-1}" == "1" ]]; then
  echo "[*] Running bridge setup..."
  "${SETUP_SCRIPT}"
else
  echo "[*] Skipping bridge setup (SETUP_BRIDGE=0)"
fi

echo "[*] Starting QEMU..."

# Forward signals to QEMU cleanly
_term() {
  echo "[*] Caught termination signal. Stopping QEMU..."
  kill -TERM "$QEMU_PID" 2>/dev/null || true
  wait "$QEMU_PID"
  exit 0
}
trap _term SIGTERM SIGINT

# Start QEMU in background
"${QEMU_SCRIPT}" "$@" &
QEMU_PID=$!

echo "[*] Waiting for SSH..."

GUEST_IP=192.168.71.140

until nc -z "$GUEST_IP" 22; do
  sleep 1
done

echo "[*] SSH ready. Starting nr-softmodem..."

sshpass -p '1234' ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$GUEST_IP" <<EOF
set -e
mkdir -p /mnt/oai-gnb
mountpoint -q /mnt/oai-gnb || mount -t 9p -o trans=virtio,version=9p2000.L oai_gnb /mnt/oai-gnb
cd /mnt/oai-gnb/bin
exec ./nr-softmodem -O "$VM_CONFIG" ${USE_ADDITIONAL_OPTIONS:-}
EOF

# Wait on QEMU
wait "$QEMU_PID"

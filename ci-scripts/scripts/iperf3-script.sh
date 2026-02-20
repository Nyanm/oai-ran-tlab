#!/bin/bash

SERVER=10.45.0.1
IFACE=oai_oaitun_ue1
BITRATE=10M
DURATION=60

ZERO_LIMIT=30
zero_count=0

export LC_NUMERIC=C

# ---- get UE IP ----
UE_IP=$(ip -o -4 addr show "$IFACE" | awk '{print $4}' | cut -d/ -f1)

if [ -z "$UE_IP" ]; then
  echo "Error: No IP found on $IFACE"
  exit 1
fi

echo "Using UE IP: $UE_IP"
echo "Starting iperf3 UDP test..."

# ---- run iperf3 with watchdog ----
iperf3 -c "$SERVER" -B "$UE_IP" -u -b "$BITRATE" -t "$DURATION" -i 1 -J |
jq -c '.intervals[].sum' |
while read line; do

  bps=$(echo "$line" | jq '.bits_per_second')
  mbps=$(awk "BEGIN {printf \"%.2f\", $bps/1000000}")

  printf "Throughput: %8.2f Mbps\n" "$mbps"

  # ---- watchdog ----
  if (( ${bps%.*} == 0 )); then
    ((zero_count++))
    echo "No traffic interval: $zero_count/$ZERO_LIMIT"
  else
    zero_count=0
  fi

  if (( zero_count >= ZERO_LIMIT )); then
    echo "No traffic → stopping iperf3"
    pkill -INT iperf3
    break
  fi
done
echo "Test finished"

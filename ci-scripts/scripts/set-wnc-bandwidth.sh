#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <bandwidth>"
    exit 1
fi

BANDWIDTH="$1"
HOST="root@10.10.0.115"

LOGFILE="radio_config_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting configuration at $(date) ===" | tee "$LOGFILE"
echo "Target bandwidth: $BANDWIDTH" | tee -a "$LOGFILE"

expect <<EOF | tee -a "$LOGFILE" > /dev/null 2>&1
set timeout 10

spawn ssh -tt $HOST

expect "#"
send "radio disable\r"

expect "#"
send "config\r"

expect "(config)#"
send "radio 1\r"

expect "(conf-rf 1)#"
send "bandwidth ${BANDWIDTH}\r"

expect "(conf-rf 1)#"
send "exit\r"

expect "(config)#"
send "exit\r"

expect "#"
send "radio enable\r"

expect "#"
send "show running-config\r"

expect "#"
send "exit\r"
EOF

echo "=== Checking bandwidth in running-config ===" | tee -a "$LOGFILE"

# Check bandwidth in the log
FOUND_BW=$(grep -E "bandwidth[[:space:]]+[0-9]+" "$LOGFILE" \
    | head -n 1 \
    | awk '{print $2}' \
    | tr -d '\r[:space:]')

if [ "$FOUND_BW" -eq "$BANDWIDTH" ]; then
    echo "✔ SUCCESS: Bandwidth correctly set to $FOUND_BW" | tee -a "$LOGFILE"
    exit 0
else
    echo "✖ ERROR: Expected bandwidth $BANDWIDTH but found: $FOUND_BW" | tee -a "$LOGFILE"
    echo "Check the device logs above for details." | tee -a "$LOGFILE"
    exit 1
fi

echo "Log saved to $LOGFILE"


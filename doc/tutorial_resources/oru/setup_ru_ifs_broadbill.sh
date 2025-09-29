#!/bin/bash
set -x
# Set these in the DU config
RU_U_PLANE_MAC=00:11:22:33:64:66
RU_C_PLANE_MAC=00:11:22:33:64:67
UNRELATED_MAC=00:11:22:33:64:70
UNRELATED_MAC2=00:11:22:33:64:71
VLAN=3
MTU=9600
IF=ens2f1np1
# this next 2 lines is for C/U planes
sudo ip link set $IF vf 8 mac $RU_U_PLANE_MAC vlan $VLAN qos 0 spoofchk off mtu $MTU
sudo ip link set $IF vf 9 mac $RU_C_PLANE_MAC vlan $VLAN qos 0 spoofchk off mtu $MTU
sleep 1
sudo /usr/local/bin/dpdk-devbind.py --unbind 41:12.0
sudo /usr/local/bin/dpdk-devbind.py --unbind 41:12.1
sudo modprobe vfio-pci
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 41:12.0
sudo /usr/local/bin/dpdk-devbind.py --bind vfio-pci 41:12.1
sleep 5

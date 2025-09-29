#!/bin/sh
#### 100G interface --> enp1s0f0 | VFs pcie-address --> 01:01.0, 01:01.1
#### 25G interface (PTP) --> enp193s0f1 | VFs pcie-address --> c1:11.0,c1:11.1

set -x
IF_NAME=enp193s0f1
NUM_VFs=2
U_PLANE_MAC_ADD=00:11:22:33:54:00
C_PLANE_MAC_ADD=00:11:22:33:54:01
VLAN=3
MTU=9600
U_PLANE_PCI=0000:c1:11.0
C_PLANE_PCI=0000:c1:11.1
## It will be something like this --> $DPDK_INST/bin
DPDK_DEVBIND_PREFIX=
sudo ethtool -G $IF_NAME rx 8160 tx 8160
sudo sh -c "echo 0 > /sys/class/net/$IF_NAME/device/sriov_numvfs"
sudo sh -c "echo $NUM_VFs > /sys/class/net/$IF_NAME/device/sriov_numvfs"
sudo modprobe -r iavf
sudo modprobe iavf
# this next 2 lines is for C/U planes
sudo ip link set $IF_NAME vf 0 mac $U_PLANE_MAC_ADD vlan $VLAN spoofchk off mtu $MTU
sudo ip link set $IF_NAME vf 1 mac $C_PLANE_MAC_ADD vlan $VLAN spoofchk off mtu $MTU
sleep 1
sudo dpdk-devbind.py --unbind $U_PLANE_PCI
sudo dpdk-devbind.py --unbind $C_PLANE_PCI
sudo modprobe vfio-pci
sudo dpdk-devbind.py --bind vfio-pci $U_PLANE_PCI
sudo dpdk-devbind.py --bind vfio-pci $C_PLANE_PCI

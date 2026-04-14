#!/bin/sh
# SPDX-License-Identifier: MIT

set -e
# Set these in the DU config
RU_U_PLANE_MAC=00:11:22:33:64:66
RU_C_PLANE_MAC=00:11:22:33:64:67
DU_U_PLANE_MAC=00:11:22:33:64:68
DU_C_PLANE_MAC=00:11:22:33:64:69
U_VLAN=3
C_VLAN=4

MTU=9216
IF_NAME=ens1f0np0
DPDK_DEVBIND_PREFIX=/usr/local/bin/

pci_addr()
{
        PF_IF=$1
        VF_INDEX=$2
        SYSFS_PATH="/sys/class/net/${PF_IF}/device/virtfn${VF_INDEX}"

        if [ ! -e "$SYSFS_PATH" ]; then
                    echo "VF $VF_INDEX not found for interface $PF_IF"
                        exit 1
        fi
        PCI_ADDR=$(basename "$(readlink "$SYSFS_PATH")")
        echo "$PCI_ADDR"
}

ethtool -G $IF_NAME rx 8160 tx 8160
sh -c "echo 0 > /sys/class/net/$IF_NAME/device/sriov_numvfs"
sh -c "echo 4 > /sys/class/net/$IF_NAME/device/sriov_numvfs"

# this next 2 lines is for C/U planes
ip link set $IF_NAME vf 0 mac $RU_U_PLANE_MAC vlan $U_VLAN spoofchk off mtu $MTU
ip link set $IF_NAME vf 1 mac $RU_C_PLANE_MAC vlan $C_VLAN spoofchk off mtu $MTU
# this next 2 lines is for DU C/U planes
ip link set $IF_NAME vf 2 mac $DU_U_PLANE_MAC vlan $U_VLAN spoofchk off mtu $MTU
ip link set $IF_NAME vf 3 mac $DU_C_PLANE_MAC vlan $C_VLAN spoofchk off mtu $MTU

C_U_PLANE_PCI0=$(pci_addr $IF_NAME 0)
C_U_PLANE_PCI1=$(pci_addr $IF_NAME 1)
C_U_PLANE_PCI2=$(pci_addr $IF_NAME 2)
C_U_PLANE_PCI3=$(pci_addr $IF_NAME 3)

sleep 1
modprobe iavf
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --unbind $C_U_PLANE_PCI0
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --unbind $C_U_PLANE_PCI1
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --unbind $C_U_PLANE_PCI2
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --unbind $C_U_PLANE_PCI3
modprobe vfio-pci
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --bind vfio-pci $C_U_PLANE_PCI0
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --bind vfio-pci $C_U_PLANE_PCI1
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --bind vfio-pci $C_U_PLANE_PCI2
${DPDK_DEVBIND_PREFIX}dpdk-devbind.py --bind vfio-pci $C_U_PLANE_PCI3
exit 0

#!/usr/bin/env bash
# SR-IOV setup for Broadbill (O-DU machine)
# NIC: ens2f1np1 (Intel E810, PCI 0000:41:00.1)
# VF0: 0000:41:11.0  MAC=00:11:22:44:54:00
# VF1: 0000:41:11.1  MAC=00:11:22:44:54:01
# Run after every reboot before starting nr-softmodem

set -e

echo 0 | sudo tee /sys/bus/pci/devices/0000:41:00.1/sriov_numvfs
echo 2 | sudo tee /sys/bus/pci/devices/0000:41:00.1/sriov_numvfs

sudo ip link set ens2f1np1 vf 0 mac 00:11:22:44:54:00 vlan 3
sudo ip link set ens2f1np1 vf 1 mac 00:11:22:44:54:01 vlan 3

sudo modprobe vfio-pci

echo vfio-pci | sudo tee /sys/bus/pci/devices/0000:41:11.0/driver_override
echo 0000:41:11.0 | sudo tee /sys/bus/pci/drivers/iavf/unbind 2>/dev/null || true
echo 0000:41:11.0 | sudo tee /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo vfio-pci | sudo tee /sys/bus/pci/devices/0000:41:11.1/driver_override
echo 0000:41:11.1 | sudo tee /sys/bus/pci/drivers/iavf/unbind 2>/dev/null || true
echo 0000:41:11.1 | sudo tee /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo "SR-IOV setup complete on Broadbill"

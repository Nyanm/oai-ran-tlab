#!/usr/bin/env bash
# SR-IOV setup for Peafowl (O-RU machine)
# NIC: enp1s0f0 (Intel E810, PCI 0000:01:00.0)
# VF0: 0000:01:01.0  MAC=00:11:22:44:64:66
# VF1: 0000:01:01.1  MAC=00:11:22:44:64:67
# Run after every reboot before starting nr-oru

set -e

echo 0 | sudo tee /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs
echo 2 | sudo tee /sys/bus/pci/devices/0000:01:00.0/sriov_numvfs

sudo ip link set enp1s0f0 vf 0 mac 00:11:22:44:64:66 vlan 3
sudo ip link set enp1s0f0 vf 1 mac 00:11:22:44:64:67 vlan 3

sudo modprobe vfio-pci

echo vfio-pci | sudo tee /sys/bus/pci/devices/0000:01:01.0/driver_override
echo 0000:01:01.0 | sudo tee /sys/bus/pci/drivers/iavf/unbind 2>/dev/null || true
echo 0000:01:01.0 | sudo tee /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo vfio-pci | sudo tee /sys/bus/pci/devices/0000:01:01.1/driver_override
echo 0000:01:01.1 | sudo tee /sys/bus/pci/drivers/iavf/unbind 2>/dev/null || true
echo 0000:01:01.1 | sudo tee /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo "SR-IOV setup complete on Peafowl"

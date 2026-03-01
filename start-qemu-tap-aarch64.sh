#!/usr/bin/env bash

cd ./qemu-rootfs/arm64

qemu-system-aarch64 -M virt -m 8G -cpu neoverse-n1 -nographic -smp cpus=8 \
-kernel Image -append "rootwait root=/dev/vda rw console=ttyAMA0" \
-netdev tap,id=net0,script=no,downscript=no,ifname=tap0 -device virtio-net-device,netdev=net0 \
-drive file=rootfs.ext4,if=none,format=raw,id=hd0 -device virtio-blk-device,drive=hd0 \
-fsdev local,id=fs0,path=/opt/oai-gnb,security_model=none,multidevs=remap \
-device virtio-9p-device,fsdev=fs0,mount_tag=oai_gnb \
 ${EXTRA_ARGS} "$@"

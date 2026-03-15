#!/bin/bash
mkdir -p /mnt/oai-gnb
mount -t 9p -o trans=virtio,version=9p2000.L oai_gnb /mnt/oai-gnb

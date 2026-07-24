#!/usr/bin/env bash
# 启动 UE (band78 SISO)。全部日志(stdout+stderr)实时保存到 .data/log/ 下带时间戳的文件（已 gitignore）。
log_dir=~/openairinterface5g/.data/log && mkdir -p "$log_dir"
log_file="$log_dir/ue-b78-$(date +%y%m%d-%H%M%S).log"
echo "日志保存到: $log_file"

sudo ~/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem \
    -O ~/6g/ue.conf \
    -r 106 --numerology 1 --band 78 -C 3619200000 \
    --ue-fo-compensation -E \
    --clock-source 2 --time-source 2 \
    --usrp-args "addr=192.168.40.2" \
    --log_config.global_log_options level,nocolor,wall_clock \
    2>&1 | tee "$log_file"

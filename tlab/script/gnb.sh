#!/usr/bin/env bash
# 启动 gNB (band78 106PRB X310)。全部日志(stdout+stderr)实时保存到 .data/log/ 下带时间戳的文件（已 gitignore）。
log_dir=~/openairinterface5g/.data/log && mkdir -p "$log_dir"
log_file="$log_dir/gnb-b78-$(date +%y%m%d-%H%M%S).log"
echo "日志保存到: $log_file"

sudo ~/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem \
    -O ~/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpx310.conf \
    --gNBs.[0].min_rxtxtime 6 \
    --usrp-tx-thread-config 1 \
    -E \
    --continuous-tx \
    --T_stdout 2 --T_nowait \
    --log_config.global_log_options level,nocolor,wall_clock \
    2>&1 | tee "$log_file"

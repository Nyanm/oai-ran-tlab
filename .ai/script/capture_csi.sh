#!/usr/bin/env bash
# 连接运行中的 gNB，把 GNB_MAC_CSI_REPORT 事件导出为带时间戳的 CSV。
#
# 前置条件：gNB 已用 `--T_stdout 2 --T_nowait` 启动（开放 T 端口、不阻塞等待 tracer）。
# 用法：  ./capture_csi.sh            # 连默认 127.0.0.1:2021，输出到 <repo>/.data/csi-rs-yymmdd-hhmmss.csv
#        CSI_IP=x CSI_PORT=y ./capture_csi.sh   # 覆盖远端地址/端口
# 停止：  Ctrl-C（csv 工具带 -f 实时 flush，已采集的数据不会丢）。
#
# 注意：csv 工具按 T_messages.txt 的事件顺序计算事件 ID，必须与 gNB 编译时所用的同一份 T_messages.txt 一致；
# 若改过 T_messages.txt 必须重编 gNB，否则 ID 错位会解析出乱码。

set -euo pipefail

trace_ip=${CSI_IP:-127.0.0.1}
trace_port=${CSI_PORT:-2021}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

csv_bin="$repo_root/cmake_targets/ran_build/build/common/utils/T/tracer/csv"
t_database="$repo_root/common/utils/T/T_messages.txt"
data_dir="$repo_root/.data"

if [[ ! -x "$csv_bin" ]]; then
  echo "错误：找不到 csv 工具 $csv_bin" >&2
  echo "请先编译：cd $repo_root/cmake_targets/ran_build/build && ninja csv" >&2
  exit 1
fi
if [[ ! -f "$t_database" ]]; then
  echo "错误：找不到 T 数据库 $t_database" >&2
  exit 1
fi

# 只读内核监听表做预检，绝不能真的发起连接：T tracer 是一次性 accept（get_connection 里
# accept 后即 close 监听 socket），任何试连都会吃掉唯一的连接名额，导致随后 csv 连接被拒。
if command -v ss >/dev/null 2>&1; then
  if ! ss -ltn 2>/dev/null | grep -qE "[:.]${trace_port}[[:space:]]"; then
    echo "警告：本机未监听 $trace_port 端口。确认 gNB 已用 --T_stdout 2 --T_nowait 启动。" >&2
    echo "      csv 将持续重试直到连上。" >&2
  fi
fi

mkdir -p "$data_dir"
out_file="$data_dir/csi-rs-$(date +%y%m%d-%H%M%S).csv"

# 字段顺序即 CSV 列顺序；timestamp 为伪字段，由 -t 指定、取 csv 工具的 e.sending_time（wall-clock，µs）
echo "采集中 -> $out_file   (Ctrl-C 停止)"
"$csv_bin" -d "$t_database" -f -t timestamp -ip "$trace_ip" -p "$trace_port" \
  GNB_MAC_CSI_REPORT \
  timestamp gNB_ID rnti frame slot csi_report_id report_quantity \
  cqi_table wb_cqi_1tb wb_cqi_2tb ri pmi_x1 pmi_x2 cri li \
  > "$out_file"

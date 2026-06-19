#!/usr/bin/env python3
"""
CSI 数据集工具：清洗 + 终端文字统计/可视化（无图形输出）。两个子命令：

  clean   [raw.csv...]    原始 CSI CSV(capture_csi.sh 产出) -> 训练用 tidy 数据集 *.clean.csv
  analyze [clean.csv...]  读 *.clean.csv，终端打印文字统计与 ASCII 可视化

clean 处理：
  - 时间戳：csv 工具只给当天时分秒，用文件名里的 YYMMDD 补全日期，并处理跨零点（时间回退即 +1 天）
  - 报告类型：report_quantity(int) -> 字符串标签，并据其把该类型不携带的字段置为 NaN
  - 派生：rank = ri + 1；frame 每 1024 回绕，解回绕后算单调递增的 abs_slot
  - 去重：同一 (rnti, abs_slot, csi_report_id) 只保留一条
省略输入时：clean 扫 .data/csi-rs-*.csv，analyze 扫 .data/*.clean.csv。依赖 pandas。
"""

import argparse
import glob
import os
import re

import pandas as pd

# ---- clean 相关 ----
DEFAULT_SLOTS_PER_FRAME = 20  # band78 30kHz SCS (numerology mu=1): 20 slots/frame
FRAMES_PER_CYCLE = 1024       # NR SFN 取值 0..1023 后回绕
WRAP_THRESHOLD = 512          # frame 相邻两报跌幅超过半个周期，判定为一次回绕

# report_quantity 枚举值 -> 标签（见 NR_CSI-ReportConfig.h，0-indexed）。本埋点只会出现含 CQI 的三种。
REPORT_TYPE = {2: "cri_RI_PMI_CQI", 5: "cri_RI_CQI", 8: "cri_RI_LI_PMI_CQI"}
PMI_VALID_TYPES = {2, 8}  # 只有这两种报告携带 PMI
LI_VALID_TYPES = {8}      # 只有 cri_RI_LI_PMI_CQI 携带 LI
SECOND_TB_MIN_RANK = 5    # NR 5~8 层才有第二码字，rank<5 时 wb_cqi_2tb 无意义

NULLABLE_INT_COLS = ["cqi2", "pmi_x1", "pmi_x2", "li"]  # 可能被置 NaN，用可空整型避免变 float
OUT_COLUMNS = ["datetime", "t_sec", "rnti", "frame", "slot", "abs_slot", "csi_report_id", "report_type",
               "cqi_table", "cqi1", "cqi2", "rank", "pmi_x1", "pmi_x2", "cri", "li"]

# ---- analyze 相关 ----
BAR_WIDTH = 40                  # 直方图条最大字符宽度
SPARK_WIDTH = 72                # 火花线列数
SPARK_CHARS = "▁▂▃▄▅▆▇█"        # 8 级高度字符
OPTIONAL_METRICS = ["cqi2", "pmi_x1", "pmi_x2", "cri", "li"]  # 仅在含非空值时才展开


def parse_file_date(path):  # 从文件名 csi-rs-YYMMDD-HHMM.csv 取出采集日期
    match = re.search(r"(\d{6})-(\d{4})", os.path.basename(path))
    if not match:
        raise ValueError(f"文件名不含 YYMMDD-HHMM：{os.path.basename(path)}")
    yymmdd = match.group(1)
    return pd.Timestamp(2000 + int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))


def clean(raw, base_date, slots_per_frame):
    df = raw.copy()

    # 时间：当天时分秒 -> 补日期；当天时分秒相对上一行回退即跨零点 +1 天
    tod = pd.to_timedelta(df["timestamp"])
    day_offset = (tod.diff() < pd.Timedelta(0)).cumsum()  # 首行 diff 为 NaT，比较得 False
    full_dt = base_date + pd.to_timedelta(day_offset, unit="D") + tod

    # frame 解回绕 -> 单调递增的绝对槽号
    frame_cycle = (df["frame"] < df["frame"].shift() - WRAP_THRESHOLD).cumsum()
    abs_slot = (frame_cycle * FRAMES_PER_CYCLE + df["frame"]) * slots_per_frame + df["slot"]

    report_type = df["report_quantity"].map(REPORT_TYPE)
    report_type = report_type.where(report_type.notna(), "unknown_" + df["report_quantity"].astype(str))
    rank = df["ri"] + 1

    out = pd.DataFrame({
        "datetime": full_dt.dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "t_sec": (full_dt - full_dt.iloc[0]).dt.total_seconds().round(6),
        "rnti": df["rnti"],
        "frame": df["frame"],
        "slot": df["slot"],
        "abs_slot": abs_slot,
        "csi_report_id": df["csi_report_id"],
        "report_type": report_type,
        "cqi_table": df["cqi_table"],
        "cqi1": df["wb_cqi_1tb"],
        "cqi2": df["wb_cqi_2tb"],
        "rank": rank,
        "pmi_x1": df["pmi_x1"],
        "pmi_x2": df["pmi_x2"],
        "cri": df["cri"],
        "li": df["li"],
    })

    # 据报告类型把不携带的字段置 NaN（先转可空整型，否则 NaN 会把整列变 float）
    out[NULLABLE_INT_COLS] = out[NULLABLE_INT_COLS].astype("Int64")
    out.loc[~df["report_quantity"].isin(PMI_VALID_TYPES), ["pmi_x1", "pmi_x2"]] = pd.NA
    out.loc[~df["report_quantity"].isin(LI_VALID_TYPES), "li"] = pd.NA
    out.loc[rank < SECOND_TB_MIN_RANK, "cqi2"] = pd.NA

    out = out.drop_duplicates(subset=["rnti", "abs_slot", "csi_report_id"], keep="first")
    return out[OUT_COLUMNS]  # 显式锁定输出列顺序，不依赖上面字典的插入顺序


def process_clean(path, slots_per_frame):
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raw = pd.DataFrame()
    if raw.empty:
        print(f"  [跳过] {os.path.basename(path)}：无数据行")
        return
    out = clean(raw, parse_file_date(path), slots_per_frame)
    out_path = re.sub(r"\.csv$", "", path) + ".clean.csv"
    out.to_csv(out_path, index=False)  # NaN/NA 默认写为空字段
    print(f"  {os.path.basename(path)}: {len(out)} 行, 时长 {out['t_sec'].iloc[-1]:.1f}s -> {out_path}")


def ascii_bar(count, max_count):
    if max_count <= 0 or count == 0:
        return ""
    return "█" * max(1, round(count / max_count * BAR_WIDTH))


def print_histogram(title, series):
    counts = series.dropna().value_counts().sort_index()  # 只展示观测到的取值
    total = int(counts.sum())
    if total == 0:
        print(f"\n  {title}: (无数据)")
        return
    max_c = int(counts.max())
    print(f"\n  {title}  (n={total})")
    for value, count in counts.items():
        pct = 100.0 * count / total
        print(f"    {str(value):>5} | {ascii_bar(count, max_c):<{BAR_WIDTH}} {count:>6} ({pct:4.1f}%)")


def describe_metric(name, series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    print(f"    {name:<7} n={len(s):<6} min={s.min():<4g} max={s.max():<4g} "
          f"mean={s.mean():6.2f} std={s.std():6.2f} p50={s.quantile(.5):<4g} p90={s.quantile(.9):<4g}")


def sparkline(values):
    s = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(s) == 0:
        return "(无数据)"
    # 按顺序下采样到 SPARK_WIDTH 个桶，每桶取均值
    n_buckets = min(SPARK_WIDTH, len(s))
    edges = [round(i * len(s) / n_buckets) for i in range(n_buckets + 1)]
    buckets = [s[edges[i]:edges[i + 1]].mean() for i in range(n_buckets)]
    lo, hi = min(buckets), max(buckets)
    span = hi - lo
    if span == 0:  # 全程恒定 -> 取中间高度
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(buckets)
    return "".join(SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((b - lo) / span * len(SPARK_CHARS)))] for b in buckets)


def analyze(path):
    df = pd.read_csv(path)
    print(f"\n===== {os.path.basename(path)} =====")
    if df.empty:
        print("  (无数据行)")
        return
    span = df["t_sec"].iloc[-1]
    rate = len(df) / span if span > 0 else float("nan")
    rntis = ", ".join(f"{int(r)}(0x{int(r):04x})" for r in sorted(df["rnti"].unique()))
    print(f"  行数 {len(df)} | 时长 {span:.1f}s | 平均 {rate:.2f} 报/s | RNTI: {rntis}")
    rep = ", ".join(f"{k}:{v}" for k, v in df["report_type"].value_counts().sort_index().items())
    print(f"  报告类型: {rep}")

    print("\n  数值摘要:")
    describe_metric("cqi1", df["cqi1"])
    describe_metric("rank", df["rank"])
    for col in OPTIONAL_METRICS:
        if col in df and df[col].notna().any():
            describe_metric(col, df[col])

    print_histogram("CQI1 分布", df["cqi1"])
    print_histogram("Rank 分布", df["rank"])

    print(f"\n  CQI1 时间序列 (左=早 右=晚, 低 {SPARK_CHARS[0]} ~ 高 {SPARK_CHARS[-1]}):")
    print(f"    {sparkline(df['cqi1'])}")
    if df["cqi1"].nunique() == 1 and df["rank"].nunique() == 1:
        print("\n  [警告] cqi1 与 rank 全程恒定，目标无方差，需在采集时制造信道变化才有训练价值")


def resolve_inputs(inputs, pattern):  # 省略输入时按 pattern 扫 .data/
    if inputs:
        return inputs
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    found = sorted(glob.glob(os.path.join(repo_root, ".data", pattern)))
    if pattern == "csi-rs-*.csv":  # clean 的输入不含已清洗结果
        found = [p for p in found if not p.endswith(".clean.csv")]
    return found


def main():
    parser = argparse.ArgumentParser(description="CSI 数据集：清洗 + 终端文字统计/可视化")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="原始 CSV -> tidy 数据集 *.clean.csv")
    p_clean.add_argument("inputs", nargs="*", help="原始 CSV；省略则处理 .data/csi-rs-*.csv")
    p_clean.add_argument("--slots-per-frame", type=int, default=DEFAULT_SLOTS_PER_FRAME,
                         help=f"每帧槽数，默认 {DEFAULT_SLOTS_PER_FRAME}（band78 30kHz）")

    p_an = sub.add_parser("analyze", help="读 *.clean.csv，终端文字统计 + ASCII 可视化")
    p_an.add_argument("inputs", nargs="*", help="*.clean.csv；省略则处理 .data/*.clean.csv")

    args = parser.parse_args()

    if args.cmd == "clean":
        inputs = resolve_inputs(args.inputs, "csi-rs-*.csv")
        if not inputs:
            print("没有找到输入文件（.data/csi-rs-*.csv）")
            return
        for path in inputs:
            process_clean(path, args.slots_per_frame)
    elif args.cmd == "analyze":
        inputs = resolve_inputs(args.inputs, "*.clean.csv")
        if not inputs:
            print("没有找到输入文件（.data/*.clean.csv）")
            return
        for path in inputs:
            analyze(path)


if __name__ == "__main__":
    main()

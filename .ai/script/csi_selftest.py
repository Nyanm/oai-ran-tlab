#!/usr/bin/env python3
"""
离线端到端回归自测：合成一份带 wall_clock 纪元秒 + CSI_REPORT 行的 gNB 日志，
跑 csi_dataset.py wash -> clean -> analyze，断言各阶段关键不变量。
纯软件、不依赖硬件、可重复运行；失败时退出码非 0。

覆盖：wash 抽行/hex rnti 转十进制/过滤非 CSI 行、去重、report_quantity->类型映射、
按类型 NaN 屏蔽(pmi/li)、rank<5 屏蔽 cqi2、rank=ri+1、frame SFN 解回绕(abs_slot 单调)、
纪元秒 -> t_sec 单调 & datetime 可解析、analyze 跑通且含关键段。
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSI_DATASET = os.path.join(SCRIPT_DIR, "csi_dataset.py")

# 合成 gNB 日志：含 2 条噪声行(应被过滤)、1 条完全重复、rq=2/5/8 各类型、rank>=5、SFN 1016->0 回绕。
# 时间戳为 wall_clock 纪元秒 SEC.US；rnti 十六进制 39b8 = 14776。
FIXTURE_LOG = """\
1753363200.005000 [PHY][I] this is a noise line without csi
1753363200.000000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=0 slot=7 report_id=0 rq=2 cqi_table=0 cqi1=15 cqi2=0 ri=0 pmi_x1=1 pmi_x2=2 cri=0 li=0
1753363200.000000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=0 slot=7 report_id=0 rq=2 cqi_table=0 cqi1=15 cqi2=0 ri=0 pmi_x1=1 pmi_x2=2 cri=0 li=0
1753363200.080000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=8 slot=7 report_id=0 rq=5 cqi_table=0 cqi1=9 cqi2=0 ri=1 pmi_x1=3 pmi_x2=3 cri=0 li=0
1753363200.020000 [MAC][W] another noise line
1753363200.160000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=16 slot=7 report_id=0 rq=8 cqi_table=0 cqi1=11 cqi2=7 ri=5 pmi_x1=2 pmi_x2=1 cri=0 li=3
1753363319.900000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=1016 slot=7 report_id=0 rq=2 cqi_table=0 cqi1=15 cqi2=0 ri=0 pmi_x1=0 pmi_x2=0 cri=0 li=0
1753363320.100000 [NR_MAC][I] CSI_REPORT rnti=39b8 frame=0 slot=7 report_id=0 rq=2 cqi_table=0 cqi1=14 cqi2=0 ri=0 pmi_x1=0 pmi_x2=0 cri=0 li=0
"""

results = []


def check(name, cond):
    results.append((name, bool(cond)))


def run_cmd(*cmd_args):
    return subprocess.run([sys.executable, CSI_DATASET, *cmd_args], capture_output=True, text=True)


def run_selftest():
    tmp = tempfile.mkdtemp(prefix="csi_selftest_")
    try:
        log_path = os.path.join(tmp, "gnb-b78-260724-120000.log")
        with open(log_path, "w") as fh:
            fh.write(FIXTURE_LOG)

        # --- wash ---
        w = run_cmd("wash", log_path)
        check("wash 退出码 0", w.returncode == 0)
        raw_path = os.path.splitext(log_path)[0] + ".csv"
        check("wash 生成原始 CSV", os.path.exists(raw_path))
        if not os.path.exists(raw_path):
            return
        raw = pd.read_csv(raw_path)
        check("wash 抽出 6 条(过滤噪声行)", len(raw) == 6)
        check("wash rnti 十六进制转十进制", (raw["rnti"] == 14776).all())
        check("wash 时间戳非空", raw["timestamp"].notna().all() and (raw["timestamp"] > 0).all())

        # --- clean ---
        c = run_cmd("clean", raw_path)
        check("clean 退出码 0", c.returncode == 0)
        clean_path = os.path.splitext(raw_path)[0] + ".clean.csv"
        check("clean 生成 .clean.csv", os.path.exists(clean_path))
        if not os.path.exists(clean_path):
            return
        df = pd.read_csv(clean_path)

        check("去重 6->5 行", len(df) == 5)
        check("report_type 映射正确",
              sorted(df["report_type"]) == ["cri_RI_CQI", "cri_RI_LI_PMI_CQI",
                                            "cri_RI_PMI_CQI", "cri_RI_PMI_CQI", "cri_RI_PMI_CQI"])
        check("rank = ri+1", list(df["rank"]) == [1, 2, 6, 1, 1])
        check("abs_slot 解回绕后单调", list(df["abs_slot"]) == [7, 167, 327, 20327, 20487])
        check("t_sec 单调递增且末值≈120.1", df["t_sec"].is_monotonic_increasing and abs(df["t_sec"].iloc[-1] - 120.1) < 1e-3)
        check("datetime 可解析", pd.to_datetime(df["datetime"]).notna().all())

        rq5 = df[df["report_type"] == "cri_RI_CQI"].iloc[0]
        check("cri_RI_CQI 屏蔽 pmi/li", pd.isna(rq5["pmi_x1"]) and pd.isna(rq5["pmi_x2"]) and pd.isna(rq5["li"]))
        rq8 = df[df["report_type"] == "cri_RI_LI_PMI_CQI"].iloc[0]
        check("cri_RI_LI_PMI_CQI 保留 pmi/li/cqi2", rq8["pmi_x1"] == 2 and rq8["li"] == 3 and rq8["cqi2"] == 7)
        check("rank<5 屏蔽 cqi2", df[df["rank"] < 5]["cqi2"].isna().all())

        # --- analyze ---
        a = run_cmd("analyze", clean_path)
        check("analyze 退出码 0", a.returncode == 0)
        check("analyze 含直方图与火花线", "CQI1 分布" in a.stdout and "时间序列" in a.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    run_selftest()
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

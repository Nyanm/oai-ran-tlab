#!/usr/bin/env python3
"""
离线端到端回归自测：合成一份覆盖各边界的原始 CSI 数据 -> csi_dataset.py clean -> analyze，
断言清洗结果的关键不变量。纯软件、不依赖硬件、可重复运行；失败时退出码非 0。

覆盖：去重、report_quantity->类型映射、按类型 NaN 屏蔽(pmi/li)、rank<5 屏蔽 cqi2、
rank=ri+1、frame SFN 解回绕(abs_slot 单调)、跨零点日期滚动、analyze 跑通且含关键段。
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSI_DATASET = os.path.join(SCRIPT_DIR, "csi_dataset.py")

# 6 行：含 1 条完全重复、rq=2/5/8 各类型、rank>=5、SFN 1016->0 回绕、23:59->次日 跨零点
FIXTURE = (
    "timestamp,gNB_ID,rnti,frame,slot,csi_report_id,report_quantity,cqi_table,"
    "wb_cqi_1tb,wb_cqi_2tb,ri,pmi_x1,pmi_x2,cri,li\n"
    "23:58:00.000000,0,14776,0,7,0,2,0,15,0,0,1,2,0,0\n"
    "23:58:00.000000,0,14776,0,7,0,2,0,15,0,0,1,2,0,0\n"
    "23:58:00.080000,0,14776,8,7,0,5,0,9,0,1,3,3,0,0\n"
    "23:58:00.160000,0,14776,16,7,0,8,0,11,7,5,2,1,0,3\n"
    "23:59:59.900000,0,14776,1016,7,0,2,0,15,0,0,0,0,0,0\n"
    "00:00:00.100000,0,14776,0,7,0,2,0,14,0,0,0,0,0,0\n"
)

results = []


def check(name, cond):
    results.append((name, bool(cond)))


def run_selftest():
    tmp = tempfile.mkdtemp(prefix="csi_selftest_")
    try:
        raw = os.path.join(tmp, "csi-rs-260619-2358.csv")
        with open(raw, "w") as fh:
            fh.write(FIXTURE)

        clean_run = subprocess.run([sys.executable, CSI_DATASET, "clean", raw], capture_output=True, text=True)
        check("clean 退出码 0", clean_run.returncode == 0)
        clean_path = raw[:-4] + ".clean.csv"
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

        rq5 = df[df["report_type"] == "cri_RI_CQI"].iloc[0]
        check("cri_RI_CQI 屏蔽 pmi/li", pd.isna(rq5["pmi_x1"]) and pd.isna(rq5["pmi_x2"]) and pd.isna(rq5["li"]))
        rq8 = df[df["report_type"] == "cri_RI_LI_PMI_CQI"].iloc[0]
        check("cri_RI_LI_PMI_CQI 保留 pmi/li/cqi2", rq8["pmi_x1"] == 2 and rq8["li"] == 3 and rq8["cqi2"] == 7)
        check("rank<5 屏蔽 cqi2", df[df["rank"] < 5]["cqi2"].isna().all())
        check("跨零点日期滚动到次日", str(df["datetime"].iloc[-1]).startswith("2026-06-20"))

        an_run = subprocess.run([sys.executable, CSI_DATASET, "analyze", clean_path], capture_output=True, text=True)
        check("analyze 退出码 0", an_run.returncode == 0)
        check("analyze 含直方图与火花线", "CQI1 分布" in an_run.stdout and "时间序列" in an_run.stdout)
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

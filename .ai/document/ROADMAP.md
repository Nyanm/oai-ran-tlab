# CSI 数据采集与离线分析工具链 — 路线图

## 背景与目标

为「无反馈网络」训练建立 CSI 数据集。需要把 gNB 收到的所有 UE 的 CSI 上报（CQI / PMI / RI，以及 CRI / LI）抓下来，
带上 RNTI、frame/slot 和时间戳，做离线整理、分析与简单可视化。采集量级约 10 分钟，应抓尽抓，后期再标记字段有效性。

**数据流（已确认）**：UE 在 PUCCH format 2/3/4 上以 UCI 形式上报 CSI → PHY `nr_decode_pucch2()`（`openair1/PHY/NR_TRANSPORT/pucch_rx.c`）
解码成 nFAPI 结构 → MAC `extract_pucch_csi_report()`（`openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_uci.c:730`）拆字段填入
`sched_ctrl->CSI_report.cri_ri_li_pmi_cqi_report`（`struct CRI_RI_LI_PMI_CQI`，定义于 `nr_mac_gNB.h:556`）。

**整体技术选型（已确认）**
| 环节 | 选型 | 未采用 / 理由 |
|---|---|---|
| 抓取 | 新增 T tracer 事件 `GNB_MAC_CSI_REPORT`，MAC 层单点埋发 | 不用 Wireshark：CSI 是 L1/UCI 控制信息，不走 MAC PDU 封装，无 dissector，抓不到 |
| 采集 | 复用 OAI 自带 `csv` tracer 工具 + 封装脚本 | 不自写采集器：`csv.c` 已能连 gNB 导出带时间戳 CSV |
| 时间戳 | `csv` 工具的 `e.sending_time`（wall-clock，µs） | 不在 C 内嵌 `gettimeofday`：精度够用，精度不足再说 |
| 落地格式 | CSV | 暂不用 Parquet：量小、便于肉眼核对，分析端用 pandas |
| 整理/分析/可视化 | Python（pandas + matplotlib） | — |

**环境（已核实）**
- `T_TRACER` 默认 `ON`（`CMakeLists.txt:348`），当前编译命令未禁用，已具备埋点能力。
- `csv` 工具随 `nr-softmodem` 自动编译（`tracer/CMakeLists.txt:101` 的 `add_dependencies(nr-softmodem T_tools)`）。
- 编译命令：`~/openairinterface5g/cmake_targets/build_oai -w USRP --ninja --nrUE --gNB --build-lib "nrscope" -C`
- T 数据库文件：`common/utils/T/T_messages.txt`

---

## 第一步：C 侧埋点（新增 T 事件 + 采集点）

### 目标
- [x] 在 `common/utils/T/T_messages.txt` 新增事件 `GNB_MAC_CSI_REPORT`（仿 `GNB_MAC_RSRP_MEASUREMENT`）
- [x] 在 `extract_pucch_csi_report()` 中以「单点埋发」方式 T() 上报一条 CSI 事件
- [x] 重新编译 gNB，确认无编译错误、`csv` 工具编出

### 事件定义（拟）
```
ID = GNB_MAC_CSI_REPORT
    DESC = NR MAC CSI report (CRI/RI/LI/PMI/CQI) for an UE
    GROUP = ALL:MAC:GNB:CSV
    FORMAT = int,gNB_ID : int,rnti : int,frame : int,slot : int,csi_report_id : int,report_quantity
           : int,cqi_table : int,wb_cqi_1tb : int,wb_cqi_2tb : int,ri : int,pmi_x1 : int,pmi_x2 : int,cri : int,li
```
（最终字段顺序以实现为准；`report_quantity` 携带 reportQuantity 枚举值，供后期标记字段有效性。）

### 技术选型
- **采用**：单点埋发——在 `extract_pucch_csi_report()` 内 `switch` 之后、period 匹配 `if` 之内、`for` 循环之内，用一个局部
  `bool is_cqi_report` 在三个 `cri_RI_*_CQI` 分支置位，循环尾部统一 `T(...)` 一次。
- **未采用**：在三个 case 内各埋一次（重复 3 处，违反小而坚固原则）；在 `evaluate_*` 函数内分别埋（字段分散，无法一条记录聚齐）。

### 决策
- **RI 语义**：事件中直接写结构体里的原始 `ri`（0-indexed，rank-1 对应 0），不在 C 侧 `+1`。语义换算（rank = ri+1）放到 Python 端，保持埋点「只搬运、不解释」。
- **字段有效性标记**：因当前仅 1x1 MIMO，PMI 等字段无实际意义但仍要抓。通过额外字段 `report_quantity` 记录本条上报类型，
  Python 端据此判断哪些字段有效（应抓尽抓 + 后期标记）。
- **gNB_ID 来源**：从 `nrmac` 取；单 gNB 场景恒为 0，保留字段以兼容模板与未来多 gNB。

### 困难与坑
- **陈旧字段（stale fields）**：`CRI_RI_LI_PMI_CQI` 是 `sched_ctrl` 里的持久结构体。纯 `cri_RI_CQI` 上报不写 PMI，
  结构体会残留上一次的 PMI 值。**不**在 C 侧清零（会动到调度器共享状态），改为靠 `report_quantity` 在 Python 端置空。
- 埋点位置必须在 period 匹配 `if` 之内，否则非本帧上报的 report 也会误发空数据。

### 实现记录（已完成）
- 事件 ID：`T_GNB_MAC_CSI_REPORT = T_ID(58)`（`T_IDs.h` 由 `T_messages.txt` 自动生成）。
- 埋点：`gNB_scheduler_uci.c` 内，局部 `bool is_cqi_report` 在三个 CQI 分支置位，`switch` 后单点 `T()` 发一条。
- **坑1（已确认）**：`csv` 可执行文件在 `cmake_targets/ran_build/build/common/utils/T/tracer/csv`，**不在** build 根目录。
- **坑2**：改 `T_messages.txt` 会触发包含 `T.h` 的文件大面积重编，但本工程仅 566 个目标，增量编译约 1–2 分钟，无需 `-C` 全清。
- 编译验证：`ninja nr-softmodem csv` 退出码 0，`gNB_scheduler_uci.c.o`/`T_IDs.h`/`nr-softmodem` 均刷新，无相关告警。
- **待运行时验证**（属第二步）：需 USRP + UE 实跑，确认事件真正触发并导出非空 CSV。

---

## 第二步：采集到 CSV（复用 csv 工具 + 封装脚本）

### 目标
- [ ] 确认 `csv` 工具路径（`cmake_targets/ran_build/build/csv`）
- [ ] 编写 `.ai/script/capture_csi.sh`：连 gNB 把 `GNB_MAC_CSI_REPORT` 导出为带时间戳 CSV
- [ ] 冒烟测试：gNB（`--T_stdout 2 --T_nowait`）+ 1 个 UE，跑出非空 CSV

### 采集命令（拟）
```bash
# gNB 启动需加：--T_stdout 2 --T_nowait   （开 2021 端口且不阻塞等待 tracer）
csv -d common/utils/T/T_messages.txt -t time \
    GNB_MAC_CSI_REPORT time rnti frame slot report_quantity cqi_table \
    wb_cqi_1tb wb_cqi_2tb ri pmi_x1 pmi_x2 cri li > csi_capture.csv
```

### 技术选型
- **采用**：live 直连 `csv` 工具导 CSV（`-t time` 用 `sending_time` 打时间戳）。
- **未采用**：`record` 录二进制再离线 `replay` 转换——适合需反复重放/存档，本场景一次性采集，从简。

### 决策 / 坑
- gNB 默认 `T_stdout=1` **不开端口**（`T.c:209`），必须 `--T_stdout 0|2` 才监听；`--T_nowait` 避免启动阻塞等 tracer。
- `--T_stdout 0`（纯二进制、阻塞等 tracer）可保证不丢启动期事件；`--T_stdout 2` 保留正常日志便于调试。二选一按需。

---

## 第三步：整理成数据集（Python）

### 目标
- [ ] `.ai/script/build_dataset.py`：读原始 CSV → 清洗 → 输出训练用 tidy CSV
- [ ] 处理：类型规整、去重、`ri+1` 换算 rank、`report_quantity` 置空无效字段、frame/slot 解 wrap（frame 0–1023）

### 决策
- 一行一条 CSI report；列含 `timestamp, rnti, frame, slot, abs_slot, rank, cqi1, cqi2, pmi_x1, pmi_x2, cri, li, cqi_table, report_quantity`。
- frame wrap：结合 wall-clock 时间戳排序 + monotonic 累加 `abs_slot`，避免 1024 帧回绕造成时序错乱。

---

## 第四步：分析 + 简单可视化（Python）

### 目标
- [ ] `.ai/script/analyze_csi.py`：基础统计 + 出图
- [ ] 图：CQI/RI 分布直方图、CQI 时间序列、（多天线时）PMI 分布

---

## 第五步：测试与评估

### 目标
- [ ] 端到端回归脚本：采集 → 整理 → 分析 一键跑通
- [ ] `/simplify` 审查 C 埋点与 Python 工具链，修正不良设计

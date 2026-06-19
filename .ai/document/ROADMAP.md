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
| 整理/分析/可视化 | Python（pandas + matplotlib） | 系统 Python 无 pip，pandas 经 `apt install python3-pandas` 安装 |

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
- [x] 确认 `csv` 工具路径（`cmake_targets/ran_build/build/common/utils/T/tracer/csv`）
- [x] 编写 `.ai/script/capture_csi.sh`：连 gNB 把 `GNB_MAC_CSI_REPORT` 导出为带时间戳 CSV
- [x] 离线冒烟：无 gNB 时校验通过、进入连接重试（事件+14 字段被 csv 正确识别）
- [x] **运行时冒烟**：gNB（`--T_stdout 2 --T_nowait`）+ 1 UE，跑出 396 行真实数据，字段对齐、取值合理

### 运行时验证记录（2026-06-19）
- 实采样例：`rnti=14776, frame +8/报（80ms 周期）, report_quantity=2 (cri_RI_PMI_CQI), cqi=15, ri=0(rank1), pmi=0(1x1)`。
- enum 确认：`reportQuantity_PR` 0-indexed → 0 NOTHING / 1 none / 2 cri_RI_PMI_CQI / 5 cri_RI_CQI / 6 cri_RSRP / 7 ssb_Index_RSRP / 8 cri_RI_LI_PMI_CQI。
- **数据集隐患**：静态近距离下 CQI 恒 15、RI 恒 rank1，目标无方差→需在采集时制造信道变化（移动/衰减/功率），属实验设计。

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
- 采集方式已定：`--T_stdout 2 --T_nowait`，**先起 gNB 再接 UE，再跑采集脚本**。
- 输出落地：项目根 `.data/`（隐藏目录，已 gitignore），命名 `csi-rs-yymmdd-hhmm.csv`（脚本自动建目录、按启动时刻命名）。
- **坑（已记入脚本头）**：csv 工具按 `T_messages.txt` 事件顺序算 ID，必须与 gNB 编译时同一份；改过 `T_messages.txt` 必须重编 gNB，否则 ID 错位会解析出乱码。
- **CSV 列**：`timestamp`（伪字段，`-t` 取 `sending_time`）+ `gNB_ID, rnti, frame, slot, csi_report_id, report_quantity, cqi_table, wb_cqi_1tb, wb_cqi_2tb, ri, pmi_x1, pmi_x2, cri, li`。原始 CSV 全量保留（含恒为 0 的 gNB_ID），常量列留给第三步 Python 清洗。
- **坑（严重，已修）**：T tracer 是**一次性 accept** 模型——`local_tracer.c:get_connection()` 里 `accept` 到一个连接后立刻 `close` 监听 socket。脚本里任何「真发起连接」的端口预探测（如 `/dev/tcp`）都会吃掉这唯一名额，导致随后 csv 连接被拒。已把预检改为只读内核监听表的 `ss -ltn`（不建立连接）。
- **配套坑**：`do_SRS` 等配置项从整数改成字符串枚举（`none/periodic/aperiodic`）。旧 conf 的 `do_SRS = 1` 需改为 `"periodic"`。这是合并带来的配置 schema 变更，非本功能引入。

---

## 第三步：整理成数据集（Python）

### 目标
- [x] `.ai/script/csi_dataset.py`：读原始 CSV → 清洗 → 输出训练用 tidy CSV（pandas 实现）
- [x] 处理：类型规整、去重、`ri+1` 换算 rank、`report_quantity` 置空无效字段、frame/slot 解 wrap（frame 0–1023）
- [x] 夹具冒烟：覆盖去重/类型映射/NaN 屏蔽/rank/SFN 解回绕/跨零点，全部通过（stdlib 版）
- [x] pandas 版重跑夹具冒烟：输出与 stdlib 版一致，Int64 可空整型保证 NaN 不污染为 float（pandas 2.3.3）

### 决策
- 一行一条 CSI report；输出列 `datetime, t_sec, rnti, frame, slot, abs_slot, csi_report_id, report_type, cqi_table, cqi1, cqi2, rank, pmi_x1, pmi_x2, cri, li`。
- frame wrap：相邻两报 frame 跌幅 > 512 判定回绕，`frame_cycle` 累加 → `abs_slot = (cycle*1024+frame)*slots_per_frame + slot`（slots_per_frame 默认 20，CLI 可改）。
- 时间戳：csv 工具只给当天时分秒，用文件名 YYMMDD 补日期；当天时分秒相对上一行回退即 +1 天（跨零点）。
- 无效字段（按 report_type）：rq=5 屏蔽 pmi/li；rq=2 屏蔽 li；rank<5 屏蔽 cqi2。写为空（NaN）。
- **技术选型**：用户要求引入 **pandas**（后续分析迟早要用）。系统 Python（3.14，apt 管理）无 pip，经 `sudo apt install python3-pandas` 安装。
- 输出命名 `<输入名>.clean.csv`；自动 glob `.data/csi-rs-*.csv` 时排除 `*.clean.csv` 避免自处理。
- **重命名**：原 `build_dataset.py` 撞 `.gitignore` 的 `build*` 规则（以 build 开头即忽略），改名 `csi_dataset.py`。

### 困难与坑
- **`data/` 被清空**：用户配置 `.gitignore` 提交时（疑似 `git clean -fdx`）连同未跟踪的 `data/` 一起删除，首份真实采集 CSV 丢失。`data/` 已 gitignore，重采即可。本步改用真实样本行 + 人造边界行的夹具验证逻辑。

---

## 第四步：分析 + CLI 可视化（Python，无图形输出）

### 目标
- [x] 分析功能**整合进 `csi_dataset.py`**，用子命令 `analyze` 调用（不再单独脚本）
- [x] 内容：概览（行数/时长/速率/RNTI/报告类型）、数值摘要（n/min/max/mean/std/p50/p90）、ASCII 直方图（cqi1/rank）、CQI 时间序列火花线
- [x] 夹具+合成数据冒烟：火花线还原 CQI 起伏，直方图/摘要正常；恒定数据触发无方差警告

### 决策
- **不做图形输出**（用户要求）：纯文字统计 + 终端 ASCII 可视化（直方图条 `█` + sparkline `▁▂▃▄▅▆▇█`），不依赖 matplotlib。
- **单脚本双子命令**（用户要求整合）：`csi_dataset.py clean`（原始→tidy）/ `analyze`（tidy→统计）。子命令 required；省略输入时分别扫 `.data/csi-rs-*.csv`、`.data/*.clean.csv`。
- clean 改为只打一行反馈（行数/时长/输出路径），分布明细统一归 analyze，避免逻辑重复。
- 可选指标（cqi2/pmi/li/cri）仅在含非空值时才展开，避免 1x1/单 TB 下刷无意义的全 0/全空行。
- analyze_csi.py 已删除（功能并入）。

---

## 第五步：测试与评估

### 目标
- [ ] 端到端回归脚本：采集 → 整理 → 分析 一键跑通
- [ ] `/simplify` 审查 C 埋点与 Python 工具链，修正不良设计

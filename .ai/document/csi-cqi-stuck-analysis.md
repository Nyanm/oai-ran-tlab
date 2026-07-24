# CQI 恒为 15 问题分析：链路无问题 · SINR 过于乐观 · CQI 被物理制约

> 日期：2026-07-24　环境：双 USRP X310（各 2×UBX160，2T2R），band40 2x2，两端均跑本仓库 OAI（gNB/nrUE），桌面 OTA 近距离。

## 一、现象

UE 上报的 CQI **恒为 15**（最高档），无论移动天线、增加遮挡都几乎不变（CQI/PMI/RI 都不动）。但 SINR 会随物理信道变化。信号再差一点就直接**接不进来 / 掉线**，而不是"降 CQI 保传输"。

## 二、结论（TL;DR）

1. **链路与配置链没有问题**：CSI-RS + CSI-IM 从 gNB 生成、下发、到 UE 测量、喂给 CQI，整条链路都正常工作（CSI-IM 确实在跑）。
2. **SINR 估计过于乐观**：UE 算出的 SINR 高达 **60–72 dB**（真实解调 SINR 上限约 30–40 dB）。它是「NZP-CSI-RS 的 RE 功率 / CSI-IM 静默 RE 功率」的裸比值，单小区无邻区干扰下分母极小 → 比值虚高。
3. **CQI 被物理/标定制约钉死**：CQI 查表在 SINR > 19 dB 即返回 15。实测 SINR 60–72 dB 远在其上，且离 19 dB 有 ~40 dB 余量，动一动到不了 CQI 敏感区 → CQI 永远 15。

即：**不是 bug，是标定错位 + 工作点太高。** 靠加衰减无法在链路存活的前提下把 CQI 打下来。

## 三、CSI → CQI 的代码路径（UE 侧，`openair1/PHY/NR_UE_TRANSPORT/csi_rx.c`）

- CQI 查表：`nr_csi_rs_cqi_estimation`（`csi_rx.c:666`），`precoded_sinr > 19 → cqi = 15`。**非硬编码**，是 SINR→CQI 表。
- SINR 计算：`nr_csi_rs_pmi_estimation`（`:550`），`sinr = signal_power / ipn`，`ipn` 来源在 `:918`：
  ```c
  csi_info->csi_im_meas_computed ? csi_info->interference_plus_noise_power  // CSI-IM 实测
                                 : noise_power;                             // 信道估计插值残差(回退)
  ```
- CSI-IM 测量：`nr_csi_im_power_estimation`（`:701`），对静默 RE 求方差 `power = E[x²]-E[x]²`（`:770`）；`nr_ue_csi_im_procedures`（`:777`）在 `:793` 置 `csi_im_meas_computed = true`。
- 始终可见的结果打印：`csi_rx.c:934`（measurement_bitmap==26 分支），打印 `RI / i1 / i2 / SINR / CQI`。

调度与配置（均正常）：
- UE PHY 逐 slot：`phy_procedures_nr_ue.c:1171` 调 `nr_ue_csi_im_procedures`（门槛 `csiim_vars.active==1`，`:1163`），排在 CSI-RS（`:1184`）之前。
- UE MAC：`nr_ue_scheduler.c:1248 nr_schedule_csi_for_im` 与 `:1195 nr_schedule_csirs_reception` 在 `:1318/:1319` 背靠背被调用，用同一 `mac->sc_info.csi_MeasConfig`、同一周期/偏移判断。
- gNB 生成：`nr_radio_config.c:583 config_csiim`（`do_CSIRS` 为真即生成 CSI-IM）、`:2111 config_csi_meas_report`（`:2143` 把 CQI 报告的干扰资源绑到 CSI-IM）、编排在 `:3489-3547`。CSI-IM 与 CSI-RS 同由 `do_CSIRS` 门控、周期/偏移同值（`:614` + `set_csiim_offset`）。
- gNB 发射端静默 CSI-IM 的 RE：`phy_procedures_nr_gNB.c:169`，`csi_type==2 (ZP-CSI) → return`（不发）。
- gNB→UE 配置传递对称（`config_ue.c:2446 modify_csi_measconfig` 等），`csi_IM` 与 `nzp_CSI_RS` 同存同传，无选择性丢弃。

## 四、为什么 SINR 会虚高（机理详解）

SINR = 分子(信号功率) / 分母(干扰+噪声功率)，而这两者在 OAI 里用**不同的量、不同的处理**算出，导致比值系统性偏高约 30–40 dB。

**分子——去噪后的信道估计功率（带处理增益）**
- SISO：`signal_power = |ĥ|²`（`csi_rx.c:582-583`），`ĥ` 是 NZP-CSI-RS 上做完 LS + 插值/滤波的**信道估计值**。信道估计对导频跨 RE 平滑/平均，相当于对噪声做了**相干平均（处理增益）**，把估计里的噪声压下去了。
- 2 端口：把各 RE / 各接收天线上的预编码信道**相干累加**后再取功率（`csi_rx.c:609-644`），同样带相干合并增益。
- 即：分子是"**噪声已被压低后的**信道能量"。

**分母——空 RE 上的原始逐 RE 噪声方差（无处理增益）**
- `nr_csi_im_power_estimation`：对 CSI-IM 静默 RE 上的**原始接收样点** `rxdataF` 求方差 `E[x²]-E[x]²`（`csi_rx.c:767-770`），既没做信道估计平滑，也没相干合并。
- 即：分母是"**未经任何处理的**逐 RE 噪声底"。

**四个叠加原因：**
1. **处理增益不对称**：分子被信道估计平滑压过噪、分母是裸噪声 → 比值高估真实"逐 RE SINR"一个"信道估计处理增益"的量（取决于滤波长度/参与平均的 RE 数，可达十几到几十 dB）。二者本就不在同一标度。
2. **分母只含热噪/量化噪，不含真实损伤**：CSI-IM 静默 RE 只反映接收机噪声底，**不包含**信道估计误差、发射 EVM（PA 非线性、IQ 失衡、相位噪声）、残余 ICI/ISI 等真正限制解调性能的因素。而 3GPP CQI 表（TS 38.214，按 0.1 BLER 标定）期望的是把这些都算进去的**有效 SINR**。用"信道能量 / 纯噪声底"查表自然过冲到 15。
3. **单小区无干扰**：CSI-IM 的本职是量**邻区干扰**；你只有一个小区，分母 ≈ 噪声底。近距离强信号下"信号/噪声底"本就可达 60–70 dB 量级——真实多小区网络里 CSI-IM 会测到邻区干扰、给出现实 SINR，这里"虚高"很大程度上就是"你的环境里确实没有干扰"。
4. **无任何标定校正**：OAI 不对该比值做偏置/上限修正；`dB_fixed` 直接饱和到 90 dB（`PHY/TOOLS/dB_routines.c`）。`ipn` 仅在为 0 时钳到 1（`csi_rx.c:561`，注释是给 ZMQ 零噪声仿真用的）——真实 RF 下 `ipn`=25–263 不触发钳位，但这么小的分母仍给出 60–72 dB。

**量化对照（实测）**：`csiim_ipn`（分母）=25–263，分子（相干合并后的信道能量）约 1e7–1e8 → 比值 1e6–1e7 → `dB_fixed` ≈ 60–72 dB，全程 ≫ 19 dB → CQI 顶格。

**一句话**：OAI 把「去噪信道能量 / 空 RE 裸噪声底」当 SINR，喂给按「有效 SINR」标定的 CQI 表，二者差了「信道估计处理增益 + 未建模损伤 + 无邻区干扰」之和，约 30–40 dB——这就是 SINR 虚高、CQI 打不下来的机理。

## 五、排查过程与两次纠错（教训）

1. **误判一**：起初以为"CSI-IM 未配置，UE 回退到插值残差"。实际 gNB 在 `do_CSIRS=1` 下**确实**生成并下发了 CSI-IM。
2. **误判二（关键坑）**：以"UE 日志里没有 `interference_plus_noise_power based on CSI-IM`"为据，判定 CSI-IM 没跑。**该日志在 `csi_rx.c:772-774` 被 `#ifdef NR_CSIIM_DEBUG` 编译掉，默认不输出**——其缺失完全不能作为"CSI-IM 未运行"的证据。
3. **定论手段**：在始终可见的 `csi_rx.c:934` 打印里加诊断字段，直接暴露干扰来源与原始功率值（见第六节）。

## 六、实测证据

UE 日志（band40 2x2，加诊断后）：
```
RI = 2 i1 = 0.0.0, i2 = 0, SINR = 72 dB, CQI = 15 [ipn_src=CSI-IM csiim_ipn=25  chest_noise=6778]
RI = 2 i1 = 0.0.0, i2 = 1, SINR = 61 dB, CQI = 15 [ipn_src=CSI-IM csiim_ipn=210 chest_noise=188339]
RI = 2 i1 = 0.0.0, i2 = 1, SINR = 63 dB, CQI = 15 [ipn_src=CSI-IM csiim_ipn=146 chest_noise=92431]
```
- `ipn_src=CSI-IM`：CSI-IM 在跑，SINR 用的是实测干扰（非残差）。
- `csiim_ipn = 25~263`（线性数字域功率）：极小 → SINR = 信号/它 ≈ 60~72 dB。
- SINR、csiim_ipn、RI(=2)、i2 都在随信道波动，唯独 **CQI 恒 15**（因全程 SINR ≫ 19 dB）。
- gNB 侧对照：262 条 cri_RI_PMI_CQI 报告，`do_CSIRS=1`、2 端口，证实 gNB 侧配置生效。

另：用户观察到"会变的那个 SINR" 是另一条独立路径 `ue->measurements.ssb_sinr_dB`（`nr_ue_measurements.c:200-202`，基于 SSB 与真实噪声底 `n0_power_avg`），**从不进入** CSI-RS 的 CQI 计算。

## 七、为什么"再差就掉线"而非降 CQI

掉线是**解码失败/超时**驱动，与 CQI 无关：
- gNB 侧：连续 PUSCH DTX ≥ `pusch_FailureThres`（默认 **10**，`gNB_scheduler_ulsch.c:990`）→ 释放 UE。
- UE 侧：连续 SSB 解码失败 ≥ N310 → T310 超时 → RLF/重建（`rrc_timers_and_constants.c`）。
- 由于 OAI 的 SINR 比真实值虚高约 40 dB：当真实链路已濒临失败时，OAI 的 SINR 才刚从 70 掉到 30、离 19 还远 → CQI 还没动，链路先死。故存在一段"CQI 恒 15 → 直接掉线"的悬崖。

## 八、加入的诊断

`csi_rx.c:934`（measurement_bitmap==26）打印追加：
`[ipn_src=%s csiim_ipn=%u chest_noise=%u]`，显示 SINR 的干扰来源（`CSI-IM`/`chest-residual`）及两者原始功率。已注明"确认后可移除"。仅 UE 侧改动，需重编 `nr-uesoftmodem`。

## 九、对数据集的影响与后续方向

- **CQI 在本环境物理死锁、无方差，不宜作为「无反馈网络」的训练目标。**
- **有信息量的会变信号在 UE 侧**：`SINR`、`RI`、`PMI(i2)`、`csiim_ipn`、`chest_noise`。若目标是"从其它观测推断信道质量"，应把采集/训练目标转向这些原始测量。
- 若确需 CQI 有方差：
  - **B**：改同轴线 + 步进衰减器，把真实工作点拉进 CQI 敏感区（OTA 桌面近距离难稳定）；
  - **代码标定**：给 `precoded_sinr` 加偏置或改查表阈值，把 60–70 → 15–25，使 CQI 在本环境响应（为造数据的人为标定，非"修复"）。
  - **A（延后掉线）**：调大 `pusch_FailureThres`/`pucch_FailureThres`、先接入再衰减，可采到更低 SINR 段（但 CQI 仍不动）。

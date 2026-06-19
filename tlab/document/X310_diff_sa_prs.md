
# X310 SA 配置 与 PRS 配置 对照说明

对比文件（均在 `~/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/`）：

- **SA 版本**：`gnb.sa.band78.fr1.106PRB.usrpx310.conf`（AI 生成，接真实 X310，本机 5GC 接入）
- **PRS 版本**：`gnb0.prs.band78.fr1.106PRB.usrpx310.conf`（仓库自带，用于下行定位实验）

---

## 一、相同点

两者本质都是 **band78 / FR1 / 106 PRB / X310** 的 **SA（Standalone）gNB** 配置，主体结构一致：

- `gNBs` 段：gNB_ID、PLMN、TAC、`servingCellConfigCommon`（载波、SSB、PRACH 等）基本相同
- NGAP / AMF 接口结构相同（指向本机核心网 192.168.70.x）
- 物理层带宽、子载波间隔、TDD 配置一致
- 均为 SA 模式，UE 可直接驻留接入

---

## 二、不同点

### 1. 功能性差异（影响行为）

| 项目 | PRS 配置 | SA 配置（AI） | 含义 |
|---|---|---|---|
| `prs_config` 段 | **有** | **无（已删除）** | PRS 配置才做定位；SA 配置是纯接入 |
| `do_CSIRS` | 无（默认关） | `= 1`（开启） | SA 配置启用 CSI-RS，便于 CQI/链路自适应 |
| 时钟/时间源 | `clock_src = "internal"` | `sdr_addrs = "addr=192.168.40.2,clock_source=gpsdo,time_source=gpsdo"` | **核心区别**：SA 配置接真实 X310 并用 GPSDO 同步 |
| `att_tx` / `att_rx` | `12` / `12` | `0` / `0` | SA 配置不做发/收衰减 |
| `max_rxgain` | `114` | `75` | SA 配置降低最大接收增益 |

### 2. 网络 / 语法差异

| 项目 | PRS 配置 | SA 配置（AI） |
|---|---|---|
| AMF/NGU 子网掩码 | `192.168.70.129/24` | `192.168.70.129/26` |
| `rfsimulator` 段写法 | 新式 `rfsimulator : { … };` | 旧式 `rfsimulator = ( { … } );` |
| 末尾仿真段 | `telnetsrv` + `@include "channelmod_rfsimu.conf"` | `vrtsim`（虚拟信道模型）段 |

---

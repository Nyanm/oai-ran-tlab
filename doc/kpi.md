# Key Performance Indicators for the OpenAirInterface Code Base

This document summarizes the main throughput KPIs 

## 1. `nr-softmodem` Performance in `oai-gNB` and `oai-gNB-du` Modes for FR1 bands

### Test Profile

The following results apply to the TDD configuration below:

|Parameter          |Value                |
|-------------------|---------------------|
|Band               |n78/n77              |
|SCS                |30 kHz               |
|DL test TDD Pattern|`DDDSU`, 2.5ms 10D2U |
|UL test TDD Pattern|`DDSUU`, 2.5ms, 64D4U|

#### KPI

Configuration: `DDDSU`, mixed slot `6D4U`

|Bandwidth MHz/PRB|Layers|DL Throughput (Mbps)|UL Throughput (Mbps)|
|-----------------|------|--------------------|--------------------|
|20(51)           |1     |72                  |39                  |
|                 |2     |143                 |65                  |
|                 |4     |258                 |X                   |
|40(106)          |1     |152                 |81                  |
|                 |2     |304                 |154                 |
|                 |4     |550                 |X                   |
|60(162)          |1     |233                 |123                 |
|                 |2     |466                 |175                 |
|                 |4     |730                 |X                   |
|80(217)          |1     |310                 |80                  |
|                 |2     |622                 |140                 |
|                 |4     |x                   |X                   |
|100(273)         |1     |X                   |X                   |
|                 |2     |X                   |X                   |
|                 |4     |1400                |X                   |


## 2. `nr-softmodem` Performance in `oai-gNB` and `oai-gNB-du` Modes for FR2 bands

### Test Profile

The following results apply to the TDD configuration below:

| Parameter | Value                         |
|-----------|-------------------------------|
|Band       | 257                           |
|SCS        | 120 kHz                       |
|DL test TDD Pattern|`DDDSU`, 0.625ms, 10D2U|
|UL test TDD Pattern|`DDDSU`, 0.625ms, 64D4U|

#### KPI

Radio Unit: MicroAmp

|Bandwidth MHz/PRB|Layers|DL Throughput (Mbps)|UL Throughput (Mbps)|
|-----------------|------|--------------------|--------------------|
|200(132)         |1     |500                 |86                  |
|                 |2     |890                 |x                   |

Round trip time (measured using ping): 4.526 ms

With `ulsch_max_frame_inactivity= 0;`

## 3. Performance Metrics for OAI Block Tests

### 3.1 Physical Simulators (`physims`)

For execution details, see [physical-simulators.md](./physical-simulators.md).

Results:


### 3.2 `phytest`

Needs to update

## 4. `nr-uesoftmodem`

### Test Profile

The following results apply to the TDD configuration below:

| Parameter | Value         |
|-----------|---------------|
| Band      | n78/n77       |
| SCS       | 30 kHz        |
| QAM       | 64            |

The UE was in SISO mode.

Testbed Architecture: 

UE <--> Over the Air 1.5m to 2m distance <--> USRP/RU <--> gNB/DU server

| Platform    | UE-Radio  | Bandwidth | DL Throughput | UL Throughput |
| ----------- | --------- | --------- | ------------- | ------------- |
| Jetson Orin | B210      | 10 MHz    | 12 Mbps       | 7.5 Mbps      |
| Jetson Orin | B210      | 20 MHz    | 20 Mbps       | 9.5 Mbps      |
| Jetson Orin | B210      | 30 MHz    | 61 Mbps       | 33 Mbps       |
| Jetson Orin | B210      | 40 MHz    | 69 Mbps       | 46 Mbps       |
| DGX Spark   | B210      | 40 MHz    | 86 Mbps       | 46 Mbps       |
| DGX Spark   | N310/x410 | 100 MHz   | 231 Mbps      | 118 Mbps      |

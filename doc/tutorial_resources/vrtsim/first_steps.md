# About

In this step the basics of compiling and running `vrtsim` will be explained.

# About vrtsim

vrtsim is a realtime/fixed timescale split8 device emulator. The main difference between
rfsim and vrtsim is the timing constraints vrtsim forces on the applications that use it.

# Compiling

Start by making a new directory under `cmake_targets`

```bash
mkdir -p cmake_targets/build/
```

Compile vrtsim + RAN elements

```bash
cd /cmake_targets/build
cmake ../../ -GNinja
cmake --build . --target vrtsim nr-softmodem nr-uesoftmodem ldpc params_libconfig params_yaml
```

# Running

Run a basic testcase to verify connectivity, no core network present

```bash
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --device.name vrtsim --vrtsim.role server
```

```bash
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim
```

# Verification

Goal is to have the UE connect for now (i.e. enter RRC Connected mode). The success depends on your
CPU speed.

## Example NR UE output when UE is connected

```
[PHY]   Initial sync: pbch decoded sucessfully, ssb index 0
[PHY]   pbch rx ok. rsrp:51 dB/RE, adjust_rxgain:-1 dB
[NR_PHY]   Cell Detected with GSCN: 0, SSB SC offset: 516, SSB Ref: 0.000000, PSS Corr peak: 99 dB, PSS Corr Average: 61
[PHY]   [UE0] In synch, rx_offset 444704 samples
[PHY]   [UE 0] Measured Carrier Frequency offset 6 Hz
[PHY]   Initial sync successful, PCI: 0
[PHY]   HW: Configuring channel 0 (rf_chain 0): setting tx_freq 3619200006 Hz, rx_freq 3619200006 Hz, tune_offset 0
[PHY]   Got synch: hw_slot_offset 29, carrier off 6 Hz, rxgain 110.000000 (DL 3619200006.000000 Hz, UL 3619200006.000000 Hz)
[PHY]   UE synchronized! decoded_frame_rx=404 UE->init_sync_frame=0 trashed_frames=14
[PHY]   Resynchronizing RX by 444704 samples
[HW]   received write reorder clear context
[NR_RRC]   SIB1 decoded
[NR_MAC]   TDD period index = 6, based on the sum of dl_UL_TransmissionPeriodicity from Pattern1 (5.000000 ms) and Pattern2 (0.000000 ms): Total = 5.000000 ms
[NR_MAC]   Set TDD configuration period to: 8 DL slots, 3 UL slots, 10 slots per period (NR_TDD_UL_DL_Pattern is 7 DL slots, 2 UL slots, 6 DL symbols, 4 UL symbols)
[NR_MAC]   Configured 1 TDD patterns (total slots: pattern1 = 10, pattern2 = 0)
[PHY]   N_TA_offset changed from 0 to 800
[MAC]   Initialization of 4-Step CBRA procedure
[NR_MAC]   PRACH scheduler: Selected RO Frame 421, Slot 19, Symbol 0, Fdm 0
[PHY]   PRACH [UE 0] in frame.slot 421.19, placing PRACH in position 2828, Msg1/MsgA-Preamble frequency start 0 (k1 0), preamble_offset 6, first_nonzero_root_idx 0, preambleIndex = 27
[PHY]   [UE 0] RAR-Msg2 decoded
[NR_MAC]   [UE 0][RAPROC][RA-RNTI 010b] Got BI RAR subPDU 5 ms
[NR_MAC]   [UE 0][RAPROC][RA-RNTI 010b] Got RAPID RAR subPDU
[NR_MAC]   [UE 0][RAPROC][422.10] Found RAR with the intended RAPID 27
[MAC]   received TA command 31
[NR_MAC]   [RAPROC][422.19] RA-Msg3 transmitted
[MAC]   [UE 0][423.10][RAPROC] 4-Step RA procedure succeeded. CBRA: Contention Resolution is successful.
[NR_RRC]   [UE0][RAPROC] Logical Channel DL-CCCH (SRB0), Received NR_RRCSetup
[RLC]   Added srb 1 to UE 0
[NR_RRC]   State = NR_RRC_CONNECTED
```

## Example gNB output when connected

```
[NR_PHY]   [RAPROC] 421.19 Initiating RA procedure with preamble 27, energy 31.7 dB (I0 0, thres 120), delay 0 start symbol 0 freq index 0
[NR_MAC]   421.19 UE RA-RNTI 010b TC-RNTI 2c32: initiating RA procedure
[NR_MAC]   UE 2c32: Msg3 scheduled at 422.19 (422.10 TDA 0) start 0 RBs 8
[NR_MAC]   UE 2c32: 422.10 Generating RA-Msg2 DCI, RA RNTI 0x10b, state 1, preamble_index(RAPID) 27, timing_offset = 0 (estimated distance 0.0 [m])
[NR_MAC]   422.10 Send RAR to RA-RNTI 010b
[NR_MAC]    422.19 PUSCH with TC_RNTI 0x2c32 received correctly
[MAC]   [RAPROC] Received SDU for CCCH length 6 for UE 2c32
[RLC]   Activated srb0 for UE 11314
[RLC]   Added srb 1 to UE 11314
[NR_MAC]   Activating scheduling Msg4 for TC_RNTI 0x2c32 (state WAIT_Msg3)
[NR_RRC]   Decoding CCCH: RNTI 2c32, payload_size 6
[NR_RRC]   [--] (cellID 0, UE ID 1 RNTI 2c32) Create UE context: CU UE ID 1 DU UE ID 11314 (rnti: 2c32, random ue id 1c14de2d3f000000)
[RRC]   activate SRB 1 of UE 1
[NR_RRC]   [DL] (cellID bc614e, UE ID 1 RNTI 2c32) Send RRC Setup
[NR_MAC]   UE 2c32 Generate Msg4: feedback at  423.17, payload 225 bytes, next state nrRA_WAIT_Msg4_MsgB_ACK
[NR_MAC]    423.17 UE 2c32: Received Ack of Msg4. CBRA procedure succeeded (UE Connected)
[NR_MAC]   Adding new UE context with RNTI 0x2c32
[NR_RRC]   [UL] (cellID bc614e, UE ID 1 RNTI 2c32) Received RRCSetupComplete (RRC_CONNECTED reached)
```

## Troubleshooting:
In case the UE does not connect:

Interrupt the process using `CTRL + C` and analyze `vrtsim` stdout at the end. You should see the histogram
print from `vrtsim`

Example from a working configuration:
```
[HW]   VRTSIM: Realtime issues: TX 0.00%, RX 0.00%
[HW]   VRTSIM: Read/write too early (suspected radio implementaton error) TX: 0, RX: 0
[HW]   VRTSIM: Average TX budget 913.111 uS (more is better)
[HW]   VRTSIM: TX budget histogram: 6468 samples
[HW]   Bin 0    [0.0 - 100.0uS]:                0
[HW]   Bin 1    [100.0 - 200.0uS]:              0
[HW]   Bin 2    [200.0 - 300.0uS]:              0
[HW]   Bin 3    [300.0 - 400.0uS]:              0
[HW]   Bin 4    [400.0 - 500.0uS]:              1
[HW]   Bin 5    [500.0 - 600.0uS]:              0
[HW]   Bin 6    [600.0 - 700.0uS]:              0
[HW]   Bin 7    [700.0 - 800.0uS]:              1
[HW]   Bin 8    [800.0 - 900.0uS]:              125
[HW]   Bin 9    [900.0 - 1000.0uS]:             6242
[HW]   Bin 10   [1000.0 - 1100.0uS]:            0
[HW]   Bin 11   [1100.0 - 1200.0uS]:            0
```

The histogram above indicates if the application encountered any timing related issues. If the average TX budget
is low (near ~0) or bin 0 is being filled that could mean that your CPU is too slow.

Possible fixes: 
 - Try disabling idle states
 ```bash
 sudo cpupower idle-set -D 0
 ```
 - Change cpu frequency scaling governor to performance
 ```bash
 sudo cpupower frequency-set --governor performance
 ```
 - Modify this line in the code of the UE and rebuild:
```diff
-#define NR_UE_CAPABILITY_SLOT_RX_TO_TX (3)
+#define NR_UE_CAPABILITY_SLOT_RX_TO_TX (6)
```

If that doesnt work, adjust `--vrtsim.timescale <timescale>` argument on `nr-softmodem`.
Start lowering the timescale (e.g. to 0.2) to slow down time until the UE connects.

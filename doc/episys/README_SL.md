
<p align="center">
  <a href="http://www.openairinterface.org/">
    <img src="./EpiSci_logo.png" alt="OAI Logo" height="90"/>
  </a>
</p>
<h1 align="center">
5G Sidelink (SL) Mode 2 Implementation in OpenAirInterface (OAI) Overview
</h>

## 1. Overview of 5G SL Features
This implementation adds support for **5G sidelink (SL) Mode 2** features within the **OpenAirInterface (OAI)** codebase, enabling device-to-device (D2D) communication without the need for a network infrastructure. Currently, the following features are implemented:

#### &emsp; ✅ **Key Features**
&emsp;&emsp; ◉ **SL Synchronization:**<br>&emsp;&emsp;&emsp;&emsp; Devices can autonomously synchronize over the PC5 interface, Sidelink, using predefined synchronization resources.<br>
&emsp;&emsp; ◉ **SL Configuration:**<br>&emsp;&emsp;&emsp;&emsp; Pre-configured SL resource pools are supported to enable flexible networking scenarios.<br>
&emsp;&emsp; ◉ **Data Transmission and Reception:**<br>&emsp;&emsp;&emsp;&emsp; End-to-end transmission and reception of SL data packets, including SL-SCH and SL-PSCCH channel handling.<br>
&emsp;&emsp; ◉ **CSI Reporting:**<br>&emsp;&emsp;&emsp;&emsp; Support for basic Channel State Information (CSI) reporting mechanisms for better link adaptation.<br>
&emsp;&emsp; ◉ **Basic Scheduling:**<br>&emsp;&emsp;&emsp;&emsp; A basic SL MAC scheduler has been implemented to enable time resource allocations in Mode 2.<br>
&emsp;&emsp; ◉ **Resource Pool Scheme:**<br>&emsp;&emsp;&emsp;&emsp; Static and pre-configured resource pool definitions are supported to enable Mode 2 communication.<br>
&emsp;&emsp; ◉ **Data Feedback:**<br>&emsp;&emsp;&emsp;&emsp; Provided a data feedback for 5G SL communication to share the reception status with the transmitter.<br>
&emsp;&emsp; ◉ **Hybrid Automatic Repeat reQuest:**<br>&emsp;&emsp;&emsp;&emsp; Enhances reliability and throughput by combining error detection with retransmission and error correction.<br>
&emsp;&emsp; ◉ **UE-to-Network (U2N) Relay**<br>&emsp;&emsp;&emsp;&emsp; U2N Relay capabilities are supported to facilitate the communication between Remote UE and gNB via Relay UE.<br>

## 2. Added Features

&emsp; The following features have been implemented and integrated into the OAI codebase to support 5G SL Mode 2:

&emsp;&emsp; ◉ Full PHY and MAC channel support <br>
&emsp;&emsp;&emsp;&emsp;🔹 **PHY:** PSBCH, PSSCH, PSCCH, PSFCH <br>
&emsp;&emsp;&emsp;&emsp;🔹 **PHY ⇄ MAC:** SL-SCH, SL-BCH<br>
&emsp;&emsp;&emsp;&emsp;🔹 **MAC ⇄ RLC:** SBCCH, SCCH, STCH<br>
&emsp;&emsp;◉ TX/RX data path support for SL Mode 2<br>
&emsp;&emsp;&emsp;&emsp;🔹 CSI Reporting (basic support)<br>
&emsp;&emsp;&emsp;&emsp;🔹 CSI Reference Signals (CSI-RS)<br>
&emsp;&emsp;&emsp;&emsp;🔹 SINR Estimation<br>
&emsp;&emsp;◉ Basic MAC scheduling for Mode 2 operation<br>
&emsp;&emsp;◉ Resource pool configuration (pre-configured/static)<br>
&emsp;&emsp;◉ Dynamic MCS support (currently up to MCS 9)<br>
&emsp;&emsp;◉ HARQ retransmission handling (basic)<br>
&emsp;&emsp;◉ SL pre-configuration support (static configuration via .conf files)<br>
&emsp;&emsp;◉ SL IP Traffic support (updates to PDCP, RLC, and SDAP layers)<br>
&emsp;&emsp;◉ 5G Sidelink Relay Adaptation Protocol (SRAP) based U2N relay support

## 3. Missing Features or Features Needing Updates
&emsp; The following features are either missing or require further updates and debugging:

&emsp;&emsp;❌ MCS Dynamic Range Limitation:<br>
&emsp;&emsp;&emsp;&emsp;Current implementation limits MCS values to a maximum of 9. This may be due to the lack of support for multiple PDUs in a single transmission time interval.<br>
&emsp;&emsp;❌ Multiple PDU Support:<br>
&emsp;&emsp;&emsp;&emsp;Not yet implemented; needed for higher MCS values and throughput.<br>
&emsp;&emsp;❌ Multiple Subchannel Support:<br>
&emsp;&emsp;&emsp;&emsp;Currently limited to single subchannel operation; lacks logical channel prioritization.<br>
&emsp;&emsp;❌ Sensing Algorithm:<br>
&emsp;&emsp;&emsp;&emsp;No support for channel sensing (needed for advanced Mode 2 and resource allocation decisions).<br>
&emsp;&emsp;❌ N310 Hardware Support:<br>
&emsp;&emsp;&emsp;&emsp;SL communication does not work on N310; only tested successfully on B210.<br>
&emsp;&emsp;❌ HARQ PID Detection:<br>
&emsp;&emsp;&emsp;&emsp;Lacks dynamic handling of HARQ process IDs.<br>
&emsp;&emsp;❌ Advanced MAC Scheduling:<br>
&emsp;&emsp;&emsp;&emsp;Only a basic round-robin/time-domain scheduler is in place. No support for advanced algorithms (e.g., sensing-based or QoS-aware).<br>
&emsp;&emsp;❌ Logical Channel Prioritization:<br>
&emsp;&emsp;&emsp;&emsp;Not currently implemented; needed for multiple logical channel management.<br>
&emsp;&emsp;❌ 5G SL Relay Validation on USRP:<br>
&emsp;&emsp;&emsp;&emsp; Not currently supported; only validated in RFSim.<br>
&emsp;&emsp;❌ Control Plane for SRAP:<br>
&emsp;&emsp;&emsp;&emsp; Not currently supported; only user plane of UE-to-Network mode is developed and validated.<br>

## 4. Test Features
&emsp; The current implementation has been tested with the following configuration:

&emsp;&emsp;✅ Working Setup:<br>
&emsp;&emsp;&emsp;&emsp; ◉ Two UE devices communicating over sidelink Mode 2 using Ettus B210 SDRs<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Basic SL transmission and reception are confirmed functional in this setup<br>
&emsp;&emsp;&emsp;&emsp; ◉ Three node Relay scenario (remote UE, relay UE and gNB) is working only on RFSIM<br>
&emsp;&emsp;❌ Unsupported or Non-Functional Setup:<br>
&emsp;&emsp;&emsp;&emsp; ◉ Ettus N310 devices: SL Mode 2 does not currently work. Debugging in work.<br>

## 5. EpiSci's 5G Sidelink Mode 2
### 5.1 **Build OAI:**
&emsp;Follow these steps to build OpenAirInterface (OAI) with support for 5G Sidelink and related features:
```
$ git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
$ cd ~/openairinterface5g
$ git fetch
$ git clean -fdX
$ git checkout sl-release-1.0
$ source oaienv
$ cd cmake_targets
$ ./build_oai -C -I --install-optional-packages   # Only necessary on fresh installs
$ ./build_oai --nrUE --gNB -w USRP -w SIMU
```

#### 5.1.1 **For Active Development and Faster Build Times:**

&emsp;If you are actively developing and want to speed up the build process, you can directly build only the executables:
```
$ cd ~/openairinterface5g/cmake_targets/ran_build/build
$ make nr-uesoftmodem rfsimulator nr_psbchsim nr_psschsim -j128
```

#### 5.1.2 **Ubuntu 24 + Address Sanitizer Workaround:**
&emsp;If you encounter a DEADLYSIGNAL error from AddressSanitizer (ASan) during OAI compilation on Ubuntu 24, apply the following workaround:
```
$ sudo sysctl vm.mmap_rnd_bits=28
```

#### 5.1.3 **Ubuntu 24.04 Build Support:**

&emsp;This branch includes Ubuntu 24.04 support with necessary build system modifications. Simply follow the standard build instructions:

```bash
$ cd ~/openairinterface5g/cmake_targets
$ ./build_oai -C -I --install-optional-packages  # Installs all dependencies
$ ./build_oai --nrUE --gNB -w USRP -w SIMU
```

&emsp;&emsp;**Note:** This branch includes a fix for ASN.1 compiler installation. The build system will automatically install **ASN.1 compiler v0.9.29** (commit 998e7ea2) instead of the newer vlm_master branch. This is required because the codebase is currently compatible with v0.9.29 (June 2024), while newer versions (v1.2+, v1.4+) generate incompatible pointer structures instead of structs, which would require approximately 1500 code changes throughout the codebase.

&emsp;&emsp;**Verification (after running `./build_oai -I`):**
```bash
$ /opt/asn1c/bin/asn1c -version
# Should output: ASN.1 Compiler, v0.9.29
```

### 5.2 **Running on RF Simulator:**

&emsp; RFSim in OAI codebase is a radio frequency (RF) simulation module that enables end-to-end testing without physical RF hardware by simulating the wireless channel and signal propagation. It facilitates testing of 4G/5G network components entirely in software, making it ideal for CI/CD, development, and validation environments.

#### 5.2.1 **Test Environment: (SL Mode 2)**

&emsp; To test IP traffic using `ping` in 5G SL Mode 2, SyncRef and the Nearby UE should run on separate machines. Currently, IP traffic is not supported when both processes are running on the same machine. Following commands allow testing of 5G SL Mode 2 with two UEs in RF simulator.

#### 5.2.1.1 **Commands:**

&emsp; ***SyncRef UE on Machine 1:***
```
$ cd $HOME/openairinterface5g/cmake_targets/ran_build/build
$ sudo LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH -E \
./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf \
--sa --sl-mode 2 --sync-ref --rfsim --thread-pool -1,-1,-1,-1 \
--rfsimulator.serveraddrsl server --rfsimulator.serverportsl 4048
```
&emsp; ***Nearby UE on Machine 2:***
```
$ cd $HOME/openairinterface5g/cmake_targets/ran_build/build
$ sudo LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH -E \
./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_ue1.conf \
--sa --sl-mode 2 --rfsim --thread-pool -1,-1,-1,-1 \
--rfsimulator.serveraddrsl <MACHINE 1 IP Address> --rfsimulator.serverportsl 4048
```
Run `ping` command on a terminal in Machine 2. Note, oaitun_ue2 is the interface name of the Nearby UE. 10.0.0.1 is the IP address of the SyncRef UE.

```
ping -I oaitun_ue2 10.0.0.1
```

**🔔Note:** Following errors can be seen when PSFCH is enabled (sl_PSFCH_period = 1, 2, 3) in the configurations files; we are working to fix this issue.
```
[NR_PHY]   [UE] SLSCH 0 in error: Setting NAK for SFN/SF 254/19 (pid 5, ndi 0, status 0, round 0, RV 0, prb_start 0, subchannel_size 50, TBS 656) r 0
[PDCP]   discard NR PDU rcvd_count=9, entity->rx_deliv 10,sdu_in_list 0
```

To perform full system testing (including CSI Reporting and PSFCH feedback), the commands remain unchanged - you only need to update the UE configuration files as outlined below.

#### 5.2.1.2 **Changing Configurations (CSI Reporting and PSFCH Period):**
&emsp; To change CSI Reporting and PSFCH configurations for sidelink testing, modify the following configuration files:

SyncRef UE Configuration File:
```
$HOME/openairinterface5g/targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf
```
Nearby UE Configuration File:
```
$HOME/openairinterface5g/targets/PROJECTS/NR-SIDELINK/CONF/sl_ue1.conf
```
In each file, update the following variables values provided in given Table 1:

&emsp; ◉ sl_CSI_Acquisition<br>
&emsp; ◉ sl_TxResPools → sl_PSFCH_period<br>
&emsp; ◉ sl_RxResPools → sl_PSFCH_period<br>

<h>Table 1: Configuration Table</h3>
<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>sl_CSI_Acquisition</th>
      <th>sl_PSFCH_period (Tx/Rx Pools)</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td>CSI Disabled 0</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">0/0</td>
  </tr>
  <tr>
    <td>CSI Disabled 1</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">1/1</td>
  </tr>
  <tr>
    <td>CSI Disabled 2</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">2/2</td>
  </tr>
    <tr>
    <td>CSI Disabled 4</td>
    <td style="text-align: center;">1</td>
    <td style="text-align: center;">3/3</td>
  </tr>
    <tr>
    <td>CSI Enabled 0</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">0/0</td>
  </tr>
  <tr>
    <td>CSI Enabled 1</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">1/1</td>
  </tr>
  <tr>
    <td>CSI Enabled 2</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">2/2</td>
  </tr>
    <tr>
    <td>CSI Enabled 4</td>
    <td style="text-align: center;">0</td>
    <td style="text-align: center;">3/3</td>
  </tr>
  <tbody>
</table>

**🔔Note:** Ensure the **sl_CSI_Acquisition** and **sl_PSFCH_period** values are set consistently across both UEs for valid test.

### 5.3&emsp;**5G SL Relay**

&emsp; To enable relay scenario support in our system, we have implemented the Sidelink Relay Adaptation Protocol (SRAP). The SRAP supports two types of relaying modes:<br>
&emsp;&emsp; ◉ UE-to-Network (U2N)<br>
&emsp;&emsp; ◉ UE-to-UE (U2U)<br>
Currently, only the U2N mode is implemented, which enables a Relay UE to forward traffic from a Remote UE to the gNB. The code of SRAP implementation is available under `openair2/LAYER2/nr_srap`, which provides following support:<br>
&emsp;&emsp; ◉ structures and functions to define the SRAP entity.<br>
&emsp;&emsp; ◉ addition and removal of SRAP headers.<br>
&emsp;&emsp; ◉ processing of the received pdu.<br>
&emsp;&emsp; ◉ forwarding of the received messages.<br>
&emsp;&emsp; ◉ passing of the PDU to lower layers.<br>
&emsp;&emsp; ◉ passing of the SDU to upper layers.<br>

### 5.4&emsp;**Over-the-air (OTA) USRP Testing:**
The OTA USRP testing was conducted using two B210s. The following commands are OTA USRP commands used for testing Sidelink Mode 2 on B210s. The following UHD commands will verify the USRP is ready for deployment and provide you with the necessary information, such as the serial and addr values.

```
$ uhd_find_devices # This will find all USRPs
$ uhd_usrp_probe # This will probe the USRP and will ensure the status is ready
```

The USRPs can be connected through either cable or over-the-air medium. In a case of cable connectivity, an attenuator can be used in lab environment to simulate real-world signal loss conditions.

#### 5.4.1 **Set the Attenuation Value**

In order to set the attenuation for each channel, run the following command.
```
curl http://169.254.10.10/:CHAN:<channel number>:SETATT:<attenuation in dB>
```
An example command is shown below, where we are setting the attenuation of channels 1 to 4 to 30 dB.
```
curl http://169.254.10.10/:CHAN:1:2:3:4:SETATT:30
```
Additionally, if you want to view the current attenuation values for each channel, run the folloiwng command.
```
curl http://169.254.10.10/:ATT?
```

#### 5.4.2 **Running of SL Mode 2 on B210s**

SSH to Machine 1. Note, the serial field may need to be changed to match the USRPs:
```
$ cd ~/openairinterface5g/cmake_targets/ran_build/build
$ sudo LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH -E \
./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_ue1.conf \
--sa --sl-mode 2 --ue-txgain 10 --ue-rxgain 100 --usrp-args "serial=3150361,type=b200" \
--thread-pool -1,-1 -E
```

SSH to Machine 2. Note, the serial field may need to be changed to match the USRPs:

```
$ cd ~/openairinterface5g/cmake_targets/ran_build/build
$ sudo LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH -E \
./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf \
--sa --sl-mode 2 --sync-ref --ue-txgain 10 --ue-rxgain 100 --usrp-args "serial=3150384,type=b200" \
--thread-pool -1,-1 -E
```
Run `ping` command on a terminal in Machine 2. Note, oaitun_ue1 is the interface name of the SyncRef UE. 10.0.0.2 is the IP address of the Nearby UE.
```
ping -I oaitun_ue1 10.0.0.2
```



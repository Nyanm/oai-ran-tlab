# OpenAirInterface documentation overview

This is the general overview page of the OpenAirInterface documentation.
This page groups links to general information, tutorials, design documents, radio integration, and special-purpose libraries.

**IMPORTANT NOTE:**
Before reading this documentation, we strongly advise you to keep your own repository rebased on `develop`
or at least to checkout the documentation on the version of the repository you are using.
Then the documentation will better reflect the features available in your repository so that you may avoid some errors.
Beware if you previously pulled the `develop` branch that your repository may be now behind `develop`.

[[_TOC_]]

## General

- [Contributing](../CONTRIBUTING.md): how to contribute to the project
- [Changelog](../CHANGELOG.md): release notes and version history
- [Notice](../NOTICE.md): third-party software credits and licenses
- [License](../LICENSE): software license and terms of use
- [Mailing Lists](https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/MailingList): subscribe for updates and get support from the community.
- [OAI Developer Meetings](https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/OpenAirDevMeetings): weekly OAI Developer Calls open to all OAI community members
- [System Requirements](./system_requirements.md): system requirements for running the OAI 4G/5G stack
- [Feature Set](./FEATURE_SET.md): lists supported features
- [Get Sources](./GET_SOURCES.md): how to download the sources
- [Build](./BUILD.md): how to build the sources
- [Code Style & Contribution](./code-style-contrib.md): overall working practices, code style, and review process
- [Cross Compile](./cross-compile.md): how to cross-compile OAI for ARM
- [Clang Format](./clang-format.md): how to format the code
- [Sanitizers](./dev_tools/sanitizers.md): how to run with ASan/UBSan/MemSAN/TSan
- [Environment Variables](./environment-variables.md): the environment variables used by OAI
- [Tuning and Security](./tuning_and_security.md): performance and security considerations

There is some general information in the [OpenAirInterface Gitlab Wiki](https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/home)

## Tutorials

- [OAI 5GC](./NR_SA_Tutorial_OAI_CN5G.md): Step-by-step tutorial to set up 5G Core Network
- [OAI gNB with COTS UE](./NR_SA_Tutorial_COTS_UE.md): Step-by-step tutorial for gNB with COTS-UE
- [OAI NR-UE](./NR_SA_Tutorial_OAI_nrUE.md): Step-by-step tutorial for NR-UE
- [Multiple OAI NR-UE with RFsimulator](./NR_SA_Tutorial_OAI_multi_UE.md): Setup for multiple UEs
- [RUNMODEM](./RUNMODEM.md): Generic information on how to run simulators, run
  with hardware, specific OAI modes (phy-test, do-ra, noS1), (5G) using SDAP and
  custom DRBs, IF setups and arbitrary frequencies, and MIMO.
- [How to run OAI with O-RAN 7.2 FHI](./ORAN_FHI7.2_Tutorial.md): Use OAI with O-RAN 7.2 fronthaul interface
- [How to run a 5G-NSA setup](./TESTING_OAI_NSA_COTS_UE.md): Guide to set up and test 5G Non-Standalone (NSA)
- [How to run a 4G setup using L1 simulator](./L1SIM.md): _Note: we recommend the RFsimulator_
- [How to use the L2 simulator](./L2NFAPI.md): L2 nFAPI Simulator Usage
- [How to use the OAI channel simulator](../openair1/SIMULATION/TOOLS/DOC/channel_simulation.md): Guide to configure and run OAI channel simulation
- [How to use GPU-accelerated channel simulation](../openair1/SIMULATION/TOOLS/DOC/gpu_acceleration.md): CUDA GPU acceleration for channel simulation
- [How to run OAI-VNF and OAI-PNF](./nfapi.md): how to run the FAPI/nFAPI split,
  including some general remarks on FAPI/nFAPI.
- [How to use the positioning reference signal (PRS)](./RUN_NR_PRS.md): How to run NR PRS with OAI gNB and nrUE
- [How to use device-to-device communication (D2D, 4G)](./d2d_emulator_setup.md): Setup and test D2D communication in 4G
- [How to run with E2 agent](../openair2/E2AP/README.md): Guide to deploy 5G OAI RAN with E2 Agent and FlexRIC nearRT-RIC
- [How to run the physical simulators](./physical-simulators.md): Guide for using physims
- [How to setup OAI with Nvidia Aerial and Foxconn](./Aerial_FAPI_Split_Tutorial.md): Guide to setup OAI with Nvidia Aerial FAPI split and Foxconn Radio Units
- [How to setup OAI with LDPC accelerators (Xilinx T2/Intel ACCs)](./LDPC_OFFLOAD_SETUP.md): Guide to setup OAI LDPC offload
- [How to setup OAI with the XDMA FPGA LDPC accelerator](./LDPC_XDMA_offload_setup.md): Guide to setup LDPC offload with XDMA driver
- [How to do a handover](./handover-tutorial.md): Guide to perform F1 and N2 handovers
- [How to setup gNB frequency](./gNB_frequency_setup.md): Instructions to configure 5G gNB frequency
- [How to use the RT data recording app](./data_recording.md): Synchronized real-time 5G data recording
- [How to use packages](./packages.md): Build OAI Deb/RPM packages using CMake
- [NR LDPC AAL](../openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/README.md): implementation for O-RAN LDPC BBDEV

### Legacy unmaintained files

- [L2NFAPI_NOS1](./L2NFAPI_NOS1.md): old L2simulator, not valid anymore
- [L2NFAPI_S1](./L2NFAPI_S1.md): old L2simulator, not valid anymore

## Designs

- [Software Architecture](./SW_archi.md): General software architecture notes
- [Information on E1](./E1AP/E1-design.md): E1 interface between CU-CP and CU-UP
- [E1AP Procedures](./E1AP/e1ap_procedures.md): Information on E1 interface procedures
- [F1AP Messages Encoding & Decoding Library](./F1AP/F1AP-lib.md): Library for encoding, decoding, and testing F1AP messages
- [Information on F1](./F1AP/F1-design.md): F1 interface design between CU and DU
- [Information on how NR nFAPI works](./NR_NFAPI_archi.md): SmallCellForum 5G (n)FAPI split and message handling
- [Flow graph of the L1 in gNB](SW-archi-graph.md): L1 threading and scheduler flow in gNB
- [L1 threads in NR-UE](./nr-ue-design.md): L1 PHY-MAC flow and UE thread processing
- [Information on gNB MAC](./MAC/mac-usage.md): 5G MAC scheduler and configuration
- [Information on gNB RRC](./RRC/rrc-usage.md): 5G RRC layer functioning and configuration
- [5G RRC Layer](./RRC/rrc-dev.md): 5G RRC procedures and UE handling
- [Information on analog beamforming implementation](./analog_beamforming.md): Implementation of analog beamforming in OAI codebase.
- [Information on the UE 5G NAS implementation](./5Gnas.md): 5GS NAS protocol and its implementation in OAI
- [Information on UL-MIMO](./UL_MIMO.md): UL-MIMO specific notes
- [gNB RACH Processing](./rach_processing_in_gNB.md): gNB RACH handling from sample collection to response scheduling

## Building and running from images

- [How to build images](../docker/README.md): Build and Use OAI with Docker/Podman
- [How to run 5G with the RFsimulator from images](../ci-scripts/yaml_files/5g_rfsimulator/README.md): Run 5G NR using RF simulator with Docker containers
- [How to run 4G with the RFsimulator from images](../ci-scripts/yaml_files/4g_rfsimulator_fdd_05MHz/README.md): Run 4G-LTE using RF simulator with Docker containers
- [How to run physical simulators in OpenShift](../openshift/README.md): OpenShift Build and Usage Procedures

## Libraries

### General

- [T tracer](../common/utils/T/DOC/T.md): a generic tracing tool (VCD, Wireshark, GUI, to save for later, ...)
- [OPT](../openair2/UTIL/OPT/README.txt): how to trace to wireshark
- [Threadpool](../common/utils/threadPool/thread-pool.md): used in L1
- [LDPC Implementation](../openair1/PHY/CODING/DOC/LDPCImplementation.md): It is a shared library
- [Time Management](time_management.md): manages time for OAI components
- [LTE RRC Configuration (ASN.1 Based)](../openair2/RRC/LTE/MESSAGES/README.md): generates LTE RRC data structures based on the ASN.1

#### Configuration Module

- [OAI Configuration Module](../common/config/DOC/config.md): Manages and validates OAI parameters
- [Config Runtime Usage](../common/config/DOC/config/rtusage.md): overview of config runtime usage
- [Config Architecture](../common/config/DOC/config/arch.md): config module source files and components
- [Config Dev Usage API](../common/config/DOC/config/devusage/api.md): information on configuration module API
- [Config Dev Usage Add A Param](../common/config/DOC/config/devusage/addaparam.md): add parameters in an existing section
- [Config Dev Usage Add Param Set](../common/config/DOC/config/devusage/addparamset.md): add a parameter set in a new section
- [Config Dev Usage Struct](../common/config/DOC/config/devusage/struct.md): configuration module public structures
- [Config Dev Usage](../common/config/DOC/config/devusage.md): overview of config dev usage

#### Loader

- [Shared Object Loader](../common/utils/DOC/loader.md): manage modular shared libraries
- [Loader Runtime Usage](../common/utils/DOC/loader/rtusage.md): information on loader runtime usage
- [Loader Architecture](../common/utils/DOC/loader/arch.md): details on loader source files
- [Loader Dev Usage Loading](../common/utils/DOC/loader/devusage/loading.md): implementation of a shared library dynamic load using the oai loader
- [Loader Dev Usage API](../common/utils/DOC/loader/devusage/api.md): information on loader API
- [Loader Dev Usage Struct](../common/utils/DOC/loader/devusage/struct.md): `loader_shlibfunc_t` structure
- [Loader Dev Usage](../common/utils/DOC/loader/devusage.md): overview of loader dev usage


#### Logging

- [Logging Module](../common/utils/LOG/DOC/log.md): OAI console logging facility
- [LOG Add Console Trace](../common/utils/LOG/DOC/addconsoletrace.md): Adding console traces in oai code
- [LOG LTTng Logs](../common/utils/LOG/DOC/lttng_logs.md): OAI gNB LTTng Tracing Setup Guide
- [LOG Configure Log](../common/utils/LOG/DOC/configurelog.md): Initializing and configuring the logging facility
- [LOG Runtime Usage](../common/utils/LOG/DOC/rtusage.md): details on LOG Runtime Usage
- [LOG Architecture](../common/utils/LOG/DOC/arch.md): logging facility source files
- [LOG Dev Usage](../common/utils/LOG/DOC/devusage.md): logging facility developer usage

### Radios

Some directories under `radio` contain READMEs:

- [RFsimulator](../radio/rfsimulator/README.md): Guide to using the RF Simulator
- [USRP](../radio/USRP/README.md): USRP Radio Setup and Usage Guide.
- [BladeRF](../radio/BLADERF/README.md): BladeRF 2.0 Micro Setup instructions
- [IQPlayer](./iqrecordplayer_usage.md): general documentation for the IQ player
- [IQ Record Playback](../radio/iqplayer/DOC/iqrecordplayer_usage.md): specific usage documentation in the `radio` directory
- [FHI 7.2](../radio/fhi_72/README.md): XRAN Driver (FHI 7.2) Setup
- [vrtsim](../radio/vrtsim/README.md): VRTSim Setup and Usage Guide
- [RF Emulator](../radio/emulator/README.md): RF Emulator Library
- [Ethernet Drivers](../radio/ETHERNET/ethernet.md): Ethernet-based drivers for fronthaul

The other SDRs (AW2S, LimeSDR, ...) have no READMEs.

### Special-purpose libraries

- [OAI Scopes](../openair1/PHY/TOOLS/readme.md): OAI has two scopes (one based on Xforms and one based on imgui)

#### Telnet Server

- [Telnet Server Help](../common/utils/telnetsrv/DOC/telnethelp.md): OAI comes with an integrated telnet server to monitor and control
- [Telnet Server](../common/utils/telnetsrv/DOC/telnetsrv.md): OAI embedded telnet server
- [Telnet Add Command](../common/utils/telnetsrv/DOC/telnetaddcmd.md): example of adding a command to the telnet server
- [Telnet Log](../common/utils/telnetsrv/DOC/telnetlog.md): telnet log command
- [Telnet Measurement](../common/utils/telnetsrv/DOC/telnetmeasur.md): information on telnet `measur` command
- [Telnet Usage](../common/utils/telnetsrv/DOC/telnetusage.md): Using the telnet server
- [Telnet Loop](../common/utils/telnetsrv/DOC/telnetloop.md): information on telnet `loop` command
- [Telnet Loader](../common/utils/telnetsrv/DOC/telnetloader.md): information on telnet `loader` command
- [Telnet History](../common/utils/telnetsrv/DOC/telnethist.md): implementation of simple history system
- [Telnet Get/Set](../common/utils/telnetsrv/DOC/telnetgetset.md): information on telnet `getall` command
- [Telnet Architecture](../common/utils/telnetsrv/DOC/telnetarch.md): overview of telnet server architecture
- [Telnet O1](../common/utils/telnetsrv/DOC/telneto1.md): telnet module to perform some O1-related actions

#### Web Server

- [Web Server](../common/utils/websrv/DOC/websrv.md): OAI comes with an integrated web server
- [Web Server Architecture](../common/utils/websrv/DOC/websrvarch.md): web server interface implementation
- [Web Server Usage](../common/utils/websrv/DOC/websrvuse.md): building the web server
- [Web Server Development](../common/utils/websrv/DOC/websrvdev.md): enhancing the web server
- [Web Server Frontend](../common/utils/websrv/frontend/README.md): information on web server frontend

## Testing

- [Unit Tests](./UnitTests.md): explains the unit testing setup
- [CU-UP Tester](../tests/nr-cuup/README.md): Component tests are under `tests/` (currently, a simple CU-UP tester)
- [Test Benches](./TESTBenches.md): Lists the CI setup and links to pipelines
- [CI Framework](../ci-scripts/README.md): The CI setup uses a custom framework to run end-to-end tests.

## Developer tools

- [Formatting](../tools/formatting/README.md): a clang-format error detection tool
- [IWYU](../tools/iwyu/README.md): a tool to detect `#include` errors
- [Docker Dev Environment](../tools/docker-dev-env/README.md): Ubuntu 24 Docker development environment
- [Plotting tools](../tools/plots/README.md): tools to visualize 5G NR simulation results
- [Documentation Best Practices](./doc_best_practices.md): overall best practices for writing documentations
- [Documentations using MkDocs](../tools/mkdocs/README.md): generate and serve documentation with MkDocs

## CI

- [Configuration Files](../ci-scripts/conf_files/README.md): naming style guide for config files
- [OAI O-RAN 7.2 Front-haul VVDN RU](../ci-scripts/yaml_files/sa_fhi_7.2_vvdn_gnb/README.md): OAI O-RAN 7.2 Front-haul Docker Compose with VVDN RU
- [OAI O-RAN 7.2 Front-haul Metanoia RU](../ci-scripts/yaml_files/sa_fhi_7.2_metanoia_2x2_gnb/README.md): OAI O-RAN 7.2 Front-haul Docker Compose with Metanoia RU
- [OAI O-RAN 7.2 Front-haul Benetel RU](../ci-scripts/yaml_files/sa_fhi_7.2_benetel550_gnb/README.md): OAI O-RAN 7.2 Front-haul Docker Compose with Benetel RU
- [CI test for 5G F1+E1 splits with RFsimulator](../ci-scripts/yaml_files/5g_rfsimulator_e1/README.md): CI test for 5G F1+E1 splits with RFsimulator
- [CI tests](../ci-scripts/tests/README.md): CI code test cases and helper scripts
- [Colosseum Testing](../ci-scripts/colosseum_scripts/README.md): automated testing of OpenAirInterface (OAI) gNB and softUE on the Colosseum

## nFAPI

- [Open-nFAPI](../nfapi/open-nFAPI/README.md): implementation of the Small Cell Forum's Network Functional API (nFAPI)
- [Open-nFAPI License](../nfapi/open-nFAPI/LICENSE.md): licensing information for Open-nFAPI
- [Open-nFAPI Examples](../nfapi/open-nFAPI/utils/examples.md): Open-nFAPI usage examples
- [Open-nFAPI Changelog](../nfapi/open-nFAPI/CHANGELOG.md): release notes for Open-nFAPI
- [nFAPI Changes](../nfapi/CHANGES.md): information on nFAPI changes

## Common Utilities


- [OCP ITTI](../common/utils/ocp_itti/itti.md): interthread interface (ITTI)
- [T Tracer Wireshark](../common/utils/T/DOC/T/wireshark.md): MAC PDUs and wireshark
- [T Tracer CSV](../common/utils/T/DOC/T/csv.md): tracer to dump information of a single trace to a CSV.
- [T Tracer Record](../common/utils/T/DOC/T/record.md): how to use the `record` tracer
- [T Tracer ENB](../common/utils/T/DOC/T/enb.md): information on T Tracer eNB
- [T Tracer Replay](../common/utils/T/DOC/T/replay.md): how to use the `replay` tracer
- [T Tracer Basic](../common/utils/T/DOC/T/basic.md): Basic usage of the T tracer
- [T Tracer How To New Trace](../common/utils/T/DOC/T/howto_new_trace.md): simple tutorial on T tracer
- [T Tracer Multi](../common/utils/T/DOC/T/multi.md): Multiple tracers
- [T Tracer To VCD](../common/utils/T/DOC/T/to_vcd.md): tracer to dump a VCD trace of the softmodem.
- [Actor README](../common/utils/actor/README.md): simple actor model implementation

## L2 Emulator (deprecated)

- [LTE Mode L2 Emulator](./episys/lte_mode_l2_emulator/README.md): Multi-UE Proxy for UEs to communicate with eNB (LTE mode)
- [NSA Mode L2 Emulator](./episys/nsa_mode_l2_emulator/README.md): Multi-UE Proxy for UEs to communicate with gNB (NSA mode)

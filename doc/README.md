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

- [Contributing](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)
- [Notice](../NOTICE.md)
- [License](../LICENSE)
- [System Requirements](./system_requirements.md)
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

- [OAI 5GC](./NR_SA_Tutorial_OAI_CN5G.md): Step-by-step tutorial to set up 5G Core
- [OAI gNB with COTS UE](./NR_SA_Tutorial_COTS_UE.md): Step-by-step tutorial for gNB with COTS-UE
- [OAI NR-UE](./NR_SA_Tutorial_OAI_nrUE.md): Step-by-step tutorial for NR-UE
- [Multiple OAI NR-UE with RFsimulator](./NR_SA_Tutorial_OAI_multi_UE.md): Setup for multiple UEs
- [RUNMODEM](./RUNMODEM.md): Generic information on how to run simulators, run
  with hardware, specific OAI modes (phy-test, do-ra, noS1), (5G) using SDAP and
  custom DRBs, IF setups and arbitrary frequencies, and MIMO.
- [How to run OAI with O-RAN 7.2 FHI](./ORAN_FHI7.2_Tutorial.md)
- [How to run a 5G-NSA setup](./TESTING_OAI_NSA_COTS_UE.md)
- [How to run a 4G setup using L1 simulator](./L1SIM.md) _Note: we recommend the RFsimulator_
- [How to use the L2 simulator](./L2NFAPI.md)
- [How to use the OAI channel simulator](../openair1/SIMULATION/TOOLS/DOC/channel_simulation.md)
- [How to use GPU-accelerated channel simulation](../openair1/SIMULATION/TOOLS/DOC/gpu_acceleration.md)
- [How to run OAI-VNF and OAI-PNF](./nfapi.md): how to run the FAPI/nFAPI split,
  including some general remarks on FAPI/nFAPI.
- [How to use the positioning reference signal (PRS)](./RUN_NR_PRS.md)
- [How to use device-to-device communication (D2D, 4G)](./d2d_emulator_setup.md)
- [How to run with E2 agent](../openair2/E2AP/README.md)
- [How to run the physical simulators](./physical-simulators.md)
- [How to setup OAI with Nvidia Aerial and Foxconn](./Aerial_FAPI_Split_Tutorial.md)
- [How to setup OAI with LDPC accelerators (Xilinx T2/Intel ACCs)](./LDPC_OFFLOAD_SETUP.md)
- [How to setup OAI with the XDMA FPGA LDPC accelerator](./LDPC_XDMA_offload_setup.md)
- [How to do a handover](./handover-tutorial.md)
- [How to setup gNB frequency](./gNB_frequency_setup.md)
- [How to use the RT data recording app](./data_recording.md)
- [How to use packages](./packages.md)

### Legacy unmaintained files

- [L2NFAPI_NOS1](./L2NFAPI_NOS1.md): old L2simulator, not valid anymore
- [L2NFAPI_S1](./L2NFAPI_S1.md): old L2simulator, not valid anymore

## Designs

- [Software Architecture](./SW_archi.md): General software architecture notes
- [Information on E1](./E1AP/E1-design.md)
- [E1AP Procedures](./E1AP/e1ap_procedures.md)
- [F1AP Messages Encoding & Decoding Library](./F1AP/F1AP-lib.md)
- [Information on F1](./F1AP/F1-design.md)
- [Information on how NR nFAPI works](./NR_NFAPI_archi.md)
- [Flow graph of the L1 in gNB](SW-archi-graph.md)
- [L1 threads in NR-UE](./nr-ue-design.md)
- [Information on gNB MAC](./MAC/mac-usage.md)
- [Information on gNB RRC](./RRC/rrc-usage.md)
- [5G RRC Layer](./RRC/rrc-dev.md)
- [Information on analog beamforming implementation](./analog_beamforming.md)
- [Information on the UE 5G NAS implementation](./5Gnas.md)
- [Information on UL-MIMO](./UL_MIMO.md): UL-MIMO specific notes
- [gNB RACH Processing](./rach_processing_in_gNB.md)

## Building and running from images

- [How to build images](../docker/README.md)
- [How to run 5G with the RFsimulator from images](../ci-scripts/yaml_files/5g_rfsimulator/README.md)
- [How to run 4G with the RFsimulator from images](../ci-scripts/yaml_files/4g_rfsimulator_fdd_05MHz/README.md)
- [How to run physical simulators in OpenShift](../openshift/README.md)

## Libraries

### General

- [T tracer](../common/utils/T/DOC/T.md): a generic tracing tool (VCD, Wireshark, GUI, to save for later, ...)
- [OPT](../openair2/UTIL/OPT/README.txt): how to trace to wireshark
- [Configuration Module](../common/config/DOC/config.md)
- [Logging Module](../common/utils/LOG/DOC/log.md)
- [Shared Object Loader](../common/utils/DOC/loader.md)
- [Threadpool](../common/utils/threadPool/thread-pool.md): used in L1
- [LDPC Implementation](../openair1/PHY/CODING/DOC/LDPCImplementation.md): It is a shared library
- [NR LDPC AAL](../openair1/PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/README.md)
- [Time Management](time_management.md)
- [LTE RRC Configuration (ASN.1 Based)](../openair2/RRC/LTE/MESSAGES/README.md)

### Radios

Some directories under `radio` contain READMEs:

- [RFsimulator](../radio/rfsimulator/README.md)
- [USRP](../radio/USRP/README.md)
- [BladeRF](../radio/BLADERF/README.md)
- [IQPlayer](./iqrecordplayer_usage.md): general documentation for the IQ player
- [IQ Record Playback](../radio/iqplayer/DOC/iqrecordplayer_usage.md): specific usage documentation in the `radio` directory
- [FHI 7.2](../radio/fhi_72/README.md)
- [vrtsim](../radio/vrtsim/README.md)
- [RF Emulator](../radio/emulator/README.md)
- [Ethernet Drivers](../radio/ETHERNET/ethernet.md)

The other SDRs (AW2S, LimeSDR, ...) have no READMEs.

### Special-purpose libraries

- [OAI Scopes](../openair1/PHY/TOOLS/readme.md): OAI has two scopes (one based on Xforms and one based on imgui)
- [Telnet Server](../common/utils/telnetsrv/DOC/telnethelp.md): OAI comes with an integrated telnet server to monitor and control
- [Web Server](../common/utils/websrv/DOC/websrv.md): OAI comes with an integrated web server

## Testing

- [Unit Tests](./UnitTests.md): explains the unit testing setup
- [CU-UP Tester](../tests/nr-cuup/README.md): Component tests are under `tests/` (currently, a simple CU-UP tester)
- [Test Benches](./TESTBenches.md): Lists the CI setup and links to pipelines
- [CI Framework](../ci-scripts/README.md): The CI setup uses a custom framework to run end-to-end tests.

## Developer tools

- [Formatting](../tools/formatting/README.md): a clang-format error detection tool
- [IWYU](../tools/iwyu/README.md): a tool to detect `#include` errors
- [Docker Dev Environment](../tools/docker-dev-env/README.md): a ubuntu24 docker development environment
- [Plotting tools](../tools/plots/README.md)
- [Documentation Best Practices](./doc_best_practices.md): overall best practices for writing documentations
- [Documentations using MkDocs](../tools/mkdocs/README.md)

## CI

- [Configuration Files](../ci-scripts/conf_files/README.md)
- [OAI O-RAN 7.2 Front-haul VVDN RU](../ci-scripts/yaml_files/sa_fhi_7.2_vvdn_gnb/README.md)
- [OAI O-RAN 7.2 Front-haul Metanoia RU](../ci-scripts/yaml_files/sa_fhi_7.2_metanoia_2x2_gnb/README.md)
- [OAI O-RAN 7.2 Front-haul Benetel RU](../ci-scripts/yaml_files/sa_fhi_7.2_benetel550_gnb/README.md)
- [CI test for 5G F1+E1 splits with RFsimulator](../ci-scripts/yaml_files/5g_rfsimulator_e1/README.md)
- [CI tests](../ci-scripts/tests/README.md)
- [Colosseum Testing](../ci-scripts/colosseum_scripts/README.md)

## nFAPI

- [Open-nFAPI](../nfapi/open-nFAPI/README.md)
- [Open-nFAPI License](../nfapi/open-nFAPI/LICENSE.md)
- [Open-nFAPI Examples](../nfapi/open-nFAPI/utils/examples.md)
- [Open-nFAPI Changelog](../nfapi/open-nFAPI/CHANGELOG.md)
- [nFAPI Changes](../nfapi/CHANGES.md)

## Common Utilities

- [Config Runtime Usage](../common/config/DOC/config/rtusage.md)
- [Config Architecture](../common/config/DOC/config/arch.md)
- [Config Dev Usage API](../common/config/DOC/config/devusage/api.md)
- [Config Dev Usage Add A Param](../common/config/DOC/config/devusage/addaparam.md)
- [Config Dev Usage Add Param Set](../common/config/DOC/config/devusage/addparamset.md)
- [Config Dev Usage Struct](../common/config/DOC/config/devusage/struct.md)
- [Config Dev Usage](../common/config/DOC/config/devusage.md)
- [OCP ITTI](../common/utils/ocp_itti/itti.md)
- [LOG Add Console Trace](../common/utils/LOG/DOC/addconsoletrace.md)
- [LOG LTTng Logs](../common/utils/LOG/DOC/lttng_logs.md)
- [LOG Configure Log](../common/utils/LOG/DOC/configurelog.md)
- [LOG Runtime Usage](../common/utils/LOG/DOC/rtusage.md)
- [LOG Architecture](../common/utils/LOG/DOC/arch.md)
- [LOG Dev Usage](../common/utils/LOG/DOC/devusage.md)
- [T Tracer Wireshark](../common/utils/T/DOC/T/wireshark.md)
- [T Tracer CSV](../common/utils/T/DOC/T/csv.md)
- [T Tracer Record](../common/utils/T/DOC/T/record.md)
- [T Tracer ENB](../common/utils/T/DOC/T/enb.md)
- [T Tracer Replay](../common/utils/T/DOC/T/replay.md)
- [T Tracer Basic](../common/utils/T/DOC/T/basic.md)
- [T Tracer How To New Trace](../common/utils/T/DOC/T/howto_new_trace.md)
- [T Tracer Multi](../common/utils/T/DOC/T/multi.md)
- [T Tracer To VCD](../common/utils/T/DOC/T/to_vcd.md)
- [Loader Runtime Usage](../common/utils/DOC/loader/rtusage.md)
- [Loader Architecture](../common/utils/DOC/loader/arch.md)
- [Loader Dev Usage Loading](../common/utils/DOC/loader/devusage/loading.md)
- [Loader Dev Usage API](../common/utils/DOC/loader/devusage/api.md)
- [Loader Dev Usage Struct](../common/utils/DOC/loader/devusage/struct.md)
- [Loader Dev Usage](../common/utils/DOC/loader/devusage.md)
- [Actor README](../common/utils/actor/README.md)
- [Telnet Add Command](../common/utils/telnetsrv/DOC/telnetaddcmd.md)
- [Telnet Log](../common/utils/telnetsrv/DOC/telnetlog.md)
- [Telnet Measurement](../common/utils/telnetsrv/DOC/telnetmeasur.md)
- [Telnet Usage](../common/utils/telnetsrv/DOC/telnetusage.md)
- [Telnet Loop](../common/utils/telnetsrv/DOC/telnetloop.md)
- [Telnet Loader](../common/utils/telnetsrv/DOC/telnetloader.md)
- [Telnet History](../common/utils/telnetsrv/DOC/telnethist.md)
- [Telnet Get/Set](../common/utils/telnetsrv/DOC/telnetgetset.md)
- [Telnet Architecture](../common/utils/telnetsrv/DOC/telnetarch.md)
- [Telnet O1](../common/utils/telnetsrv/DOC/telneto1.md)
- [Telnet Server](../common/utils/telnetsrv/DOC/telnetsrv.md)
- [Web Server Architecture](../common/utils/websrv/DOC/websrvarch.md)
- [Web Server Usage](../common/utils/websrv/DOC/websrvuse.md)
- [Web Server Development](../common/utils/websrv/DOC/websrvdev.md)
- [Web Server Frontend README](../common/utils/websrv/frontend/README.md)

## EpiSci

- [LTE Mode L2 Emulator](./episys/lte_mode_l2_emulator/README.md)
- [NSA Mode L2 Emulator](./episys/nsa_mode_l2_emulator/README.md)

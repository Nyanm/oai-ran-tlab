# OpenAirInterface documentation overview

This is the general overview page of the OpenAirInterface documentation.
This page groups links to general information, tutorials, design documents, radio integration, and special-purpose libraries.

**IMPORTANT NOTE:**
Before reading this documentation, we strongly advise you to keep your own repository rebased on `develop`
or at least to checkout the documentation on the version of the repository you are using.
Then the documentation will better reflect the features available in your repository so that you may avoid some errors.
Beware if you previously pulled the `develop` branch that your repository may be now behind `develop`.

> There is some general information in the [OpenAirInterface Gitlab Wiki](https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/home)

[[_TOC_]]

## General

- [Feature Set](./FEATURE_SET.md): lists supported features
- [Get Sources](./GET_SOURCES.md): how to download the sources
- [Build](./BUILD.md): how to build the sources
- [Code Style & Contribution](./code-style-contrib.md): overall working practices, code style, and review process
- [Cross Compile](./cross-compile.md): how to cross-compile OAI for ARM
- [Clang Format](./clang-format.md): how to format the code
- [Sanitizers](./dev_tools/sanitizers.md): how to run with ASan/UBSan/MemSAN/TSan
- [Environment Variables](./environment-variables.md): the environment variables used by OAI
- [Tuning and Security](./tuning_and_security.md): performance and security considerations

## Tutorials

- [OAI 5GC](./NR_SA_Tutorial_OAI_CN5G.md): Step-by-step tutorial to set up 5G Core
- [OAI gNB with COTS UE](./NR_SA_Tutorial_COTS_UE.md): Step-by-step tutorial for gNB with COTS-UE
- [OAI NR-UE](./NR_SA_Tutorial_OAI_nrUE.md): Step-by-step tutorial for NR-UE
- [Multiple OAI NR-UE with RFsimulator](./NR_SA_Tutorial_OAI_multi_UE.md): Setup for multiple UEs
- [RUNMODEM](./RUNMODEM.md): Generic information on how to run simulators, run
  with hardware, specific OAI modes (phy-test, do-ra, noS1), (5G) using SDAP and
  custom DRBs, IF setups and arbitrary frequencies, and MIMO.
- [O-RAN 7.2 FHI](./ORAN_FHI7.2_Tutorial.md): How to run OAI with O-RAN 7.2 FHI
- [5G-NSA setup](./TESTING_OAI_NSA_COTS_UE.md): How to run a 5G-NSA setup
- [4G L1 Simulator](./L1SIM.md): How to run a 4G setup using L1 simulator (Note: we recommend the RFsimulator_)
- [L2 simulator](./L2NFAPI.md): How to use the L2 simulator
- [OAI channel simulator](../openair1/SIMULATION/TOOLS/DOC/channel_simulation.md): How to use the OAI channel simulator
- [Multiple BWPs](./RUN_NR_multiple_BWPs.md): How to use multiple BWPs
- [OAI-VNF and OAI-PNF](./nfapi.md): how to run the FAPI/nFAPI split, including some general remarks on FAPI/nFAPI.
- [Positioning Reference Signal](./RUN_NR_PRS.md): How to use the positioning reference signal (PRS)
- [D2D Communication](./d2d_emulator_setup.md): How to use device-to-device communication (D2D, 4G)
- [How to run with E2 agent](../openair2/E2AP/README.md): How to run with E2 agent
- [Physical Simulators](./physical-simulators.md): How to run the physical simulators
- [Nvidia Aerial and Foxconn](./Aerial_FAPI_Split_Tutorial.md): How to setup OAI with Nvidia Aerial and Foxconn
- [LDPC accelerators](./LDPC_OFFLOAD_SETUP.md): How to setup OAI with LDPC accelerators (Xilinx T2/Intel ACCs)
- [XDMA FPGA LDPC accelerator](./LDPC_XDMA_offload_setup.md): How to setup OAI with the XDMA FPGA LDPC accelerator
- [Handover Tutorial](./handover-tutorial.md): How to perform handovers
- [gNB frequency Setup](./gNB_frequency_setup.md): How to setup gNB frequency
- [RT Data Recording](./data_recording.md): How to use the RT data recording app
- [Packages](./packages.md): How to build Deb/RPM packages with OAI's build system

### Legacy unmaintained files

- [L2NFAPI_NOS1.md](./L2NFAPI_NOS1.md): old L2simulator, not valid anymore
- [L2NFAPI_S1.md](./L2NFAPI_S1.md): old L2simulator, not valid anymore

## Designs

- [SW Architecture](./SW_archi.md): General software architecture notes
- [E1 Design](./E1AP/E1-design.md): General design and information on E1
- [F1 Design](./F1AP/F1-design.md): General design and information on F1
- [NR nFAPI](./NR_NFAPI_archi.md): Information on how NR nFAPI works
- [L1 Flow Graph](SW-archi-graph.md): Flow graph of the L1 in gNB
- [NR-UE L1 Threads](./nr-ue-design.md): L1 threads in NR-UE
- [gNB MAC](./MAC/mac-usage.md): Information on gNB MAC
- [gNB RRC](./RRC/rrc-usage.md): Information on gNB RRC
- [Analog Beamforming](./analog_beamforming.md): Information on analog beamforming implementation
- [UE 5G NAS](./5Gnas.md): Information on the UE 5G NAS implementation
- [UL-MIMO](./UL_MIMO.md): UL-MIMO specific notes

## Building and running from images

- [Build Images](../docker/README.md): How to build images
- [5G RFsimulator Images](../ci-scripts/yaml_files/5g_rfsimulator/README.md): How to run 5G with the RFsimulator from images
- [4G RFsimulator Images](../ci-scripts/yaml_files/4g_rfsimulator_fdd_05MHz/README.md): How to run 4G with the RFsimulator from images
- [Physical Simulators in OpenShift](../openshift/README.md): How to run physical simulators in OpenShift

## Libraries

### General

- The [T tracer](../common/utils/T/DOC/T.md): a generic tracing tool (VCD, Wireshark, GUI, to save for later, ...)
- [OPT](../openair2/UTIL/OPT/README.txt): how to trace to wireshark
- [configuration module](../common/config/DOC/config.md)
- [logging module](../common/utils/LOG/DOC/log.md)
- [shared object loader](../common/utils/DOC/loader.md)
- [threadpool](../common/utils/threadPool/thread-pool.md): used in L1
- [LDPC implementation](../openair1/PHY/CODING/DOC/LDPCImplementation.md): It is a shared library
- [time management](time_management.md)

### Radios

> Some directories under `radio` contain READMEs. The other SDRs (AW2S, LimeSDR, ...) have no READMEs.

- [RFsimulator](../radio/rfsimulator/README.md): documentation for the RF simulator
- [USRP](../radio/USRP/README.md): documentation for USRP
- [BladeRF](../radio/BLADERF/README): documentation for BladeRF
- [IQPlayer](./iqrecordplayer_usage.md): general documentation for the IQ player
- [IQ Record Playback](../radio/iqplayer/DOC/iqrecordplayer_usage.md): specific usage documentation in the `radio` directory
- [FHI 7.2](../radio/fhi_72/README.md): documentation for the FHI 7.2 radio
- [VRT Simulator](../radio/vrtsim/README.md): documentation for the VRT simulator
- [RF Emulator](../radio/emulator/README.md): documentation for the RF emulator

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

- [formatting](../tools/formatting/README.md): a clang-format error detection tool
- [iwyu](../tools/iwyu/README.md): a tool to detect `#include` errors
- [docker-dev-env](../tools/docker-dev-env/README.md): a ubuntu24 docker development environment
- [Documentation Best Practices](./doc_best_practices.md): overall best practices for writing documentations

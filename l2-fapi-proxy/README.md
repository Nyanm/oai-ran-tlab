# Description #

This directory contains the L2 FAPI Proxy server, which allows multiple UEs to communicate with an OAI gNB at Layer 2 of the stack.
This makes it possible to bypass the L1 processing of the gNB/UE.
The code is based off the original Multi-UE Proxy from EpiSci (located at <https://gitlab.eurecom.fr/oai/openairinterface5g>). 
The L2 Proxy and UE L2 Simulation (i.e., emulate-l1) Mode code has been updated based on the wk25 release of the OAI gNB and UE. 
Note that only NR Standalone (SA) Mode is currently supported with a single gNB.
The updated feature is provided by Samsung Research America (SRA). 

The high-level architecture and networking of the L2 Proxy and UE in L2 Simulation Mode is shown below. 

![L2 Proxy Architecture](l2_proxy_arch_diag.png)

## Included Features ##

The features included from the former EpiSci multi-UE proxy are:

- Socket communciation from/to nrUE(s)
- Socket communication from/to gNB
- Uplink/downlink packet queueing
- nFAPI compatibility
- Logging mechanism

Modifications made in the updated L2 FAPI Proxy/L2 Simulation Mode:

- Updated nFAPI messages and methods to the latest API version used by OAI (as of wk25)
- Support for NR Standalone (SA) operation
- Tested to support up to 50 UEs running on a single multi-core server in real-time
- Downlink (PDSCH) PHY abstraction and error model based on CSI traces

** Note: Only NR Standalone (SA) Mode is currently supported with a single gNB. **

## L2 Simulation and PHY Abstraction ##

![L2 Simulation Mode](l2_simulation_diag.png)

## Contact ##

For questions on the updated L2 FAPI Proxy, please contact:
- Russell Ford <russelldford@gmail.com>
- Daoud Burghal <d.burghal@samsung.com>
- Pranav Madadi <p.madadi@samsung.com>

## Prerequsites ##

TODO
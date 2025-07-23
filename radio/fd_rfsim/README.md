# FD RFsimulator Adapter
## Overview 

This library provides a frequency domain adaptation layer for the OAI RFsimulator,
enabling integration with OAI O-DU configuration files. It facilitates testing and
development of 5G NR O-DU and UE components in a simulated radio environment without
real time requirements.

## Usage

### Build

```
cmake --build . --target fd_rfsim
```

Example config provided in `gnb.sa.band77.106prb.fhi72.4x4-fd-rfsim.conf`

### Run

Run gNB
```
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band77.106prb.fhi72.4x4-fd-rfsim.conf --rfsimulator.serveraddr server
```

Run UE
```
sudo ./nr-uesoftmodem -C 4049760000 -r 106 --numerology 1 --ssb 516  --band 77 --rfsim
```

# L2 FAPI Proxy

## Overview

The L2 FAPI Proxy is a module that enables Layer 2 (MAC-to-MAC) communication between OAI gNB and UE instances by proxying FAPI/nFAPI messages. This allows testing L2+ protocols without physical layer processing or radio hardware.

In standard OAI nFAPI split architecture, the Virtual Network Function (VNF) running L2+ stack communicates with the Physical Network Function (PNF) running L1 over the FAPI interface. When this communication occurs over network sockets, the nFAPI protocol is used.

In L2 Proxy mode, the gNB VNF communicates with the proxy server instead of its own L1. The proxy forwards FAPI/nFAPI messages between gNB L2 and UE L2, applying PHY abstraction and optional channel simulation in place of actual physical layer processing.

```
┌─────────────┐    nFAPI (UDP)    ┌──────────────┐    nFAPI (UDP)   ┌─────────────┐
│   gNB VNF   │◄─────────────────►│  L2 Proxy    │◄────────────────►│   UE PNF    │
│   (L2+)     │                   │   Server     │                  │   (L2+)     │
└─────────────┘                   └──────────────┘                  └─────────────┘
                                        │
                                        ├─ PHY Abstraction
                                        ├─ Channel Simulation (optional)
                                        └─ CSI Trace Injection (optional)
```

The proxy implements nFAPI P5 (configuration) and P7 (slot-based messaging) interfaces, supports multiple UE connections, and replaces waveform-level PHY processing with link-level abstraction models.

## Architecture

This is a separate module with independent build system that does not modify core OAI gNB or UE source code. It communicates via standard nFAPI sockets

## Directory Structure

```
l2-fapi-proxy/
├── open-nFAPI/          # nFAPI library (P4/P5/P7 interfaces)
│   ├── nfapi/           # Message definitions and encode/decode
│   ├── fapi/            # NR FAPI primitives  
│   ├── pnf/             # PNF-side implementation
│   └── common/          # Utilities and debug functions
├── src/                 # Proxy implementation
│   ├── proxy.cc         # Main entry point
│   ├── nr_proxy.cc      # NR proxy logic
│   ├── lte_proxy.cc     # LTE proxy support
│   ├── nfapi_pnf.c      # PNF interface handling
│   ├── nfapiutils.c     # Utility functions
│   ├── queue.c          # Message queue
│   └── run_l2_proxy.sh  # Deployment script for Kubernetes
├── common/              # Common utilities
├── build/               # Build output
│   └── proxy            # Compiled executable
├── Makefile             # Standalone build system (not CMake)
└── README.md
```

## Building

The L2 Proxy has its own Makefile and is built independently from the OAI CMake build system.

```bash
cd l2-fapi-proxy
make
```

Output: `build/proxy`

To clean:
```bash
make clean
```

## Usage

### Automated Script (Kubernetes)

For Kubernetes deployments, the script automatically discovers pod IP addresses:

```bash
./src/run_l2_proxy.sh <NUM_GNB> <NUM_UE> [CH_TRACE_FILE]
```

Example (1 gNB, 2 UEs):
```bash
./src/run_l2_proxy.sh 1 2
```

### Manual Execution

Specify IP addresses explicitly:

```bash
./build/proxy --gnb <GNB_IP> --proxy <PROXY_IP> --ue <UE0_IP> <UE1_IP> ... [--ch <CH_TRACE_FILE>]
```

Example:
```bash
./build/proxy --gnb 10.42.0.220 --proxy 10.42.0.221 --ue 10.42.0.222 10.42.0.223
```

## Deployment (Kubernetes)

### 1. Deploy Core Network

```bash
cd <oai-cn5g-fed>/charts
helm install basic oai-5g-core/oai-5g-basic/ -n oai
```

Verify:
```bash
kubectl get pods -n oai
```

### 2. Start gNB in VNF Mode

Get gNB pod name:
```bash
kubectl get pods -n oai | grep oai-gnb
```

Enter container:
```bash
kubectl exec -it <GNB_POD_NAME> -n oai -- /bin/bash
```

Run gNB softmodem:
```bash
cd /opt/oai-ran/cmake_targets/ran_build/build
./nr-softmodem -O /tmp/gnb.conf --nfapi VNF
```

Expected logs:
```
[GTPU]   Configuring GTPu address : 10.42.0.220, port : 2152
[NGAP]   Received NGSetupResponse from AMF
[VNF]    Sent NFAPI_VNF_CONFIG_REQ num_tlv:131
[NR_MAC] Frame.Slot 128.0
```

### 3. Start L2 Proxy

In the gNB container (new terminal):
```bash
cd /opt/oai-ran/l2-fapi-proxy
./src/run_l2_proxy.sh 1 2
```

Expected logs:
```
RUNNING pack_param_response
[PNF] vnf p7 0.1.4.0:50611
```

### 4. Attach UE Instances

From host OS:
```bash
cd <oai-cn5g-fed>/scripts/kube
./run_ue_oai <START_UE_IDX> <END_UE_IDX>
```

Example (UEs 0-9):
```bash
./run_ue_oai 0 9
```

**Manual UE execution** (inside UE container):
```bash
/opt/oai-ran/cmake_targets/ran_build/build/nr-uesoftmodem \
  -O /tmp/nr-ue.conf \
  --nfapi STANDALONE_PNF \
  --node-number 2 \
  --emulate-l1 \
  -r 106 \
  --numerology 1 \
  -C 3619200000
```

Note: `--node-number` should be `2 + UE_ID` (e.g., 2 for UE 0, 3 for UE 1)

Expected UE logs:
```
[MAC]   [UE 0][RAPROC] 4-Step RA procedure succeeded. CBRA: Contention Resolution is successful.
[NAS]   Received Registration Accept with result 3GPP
[NAS]   Received PDU Session Establishment Accept, UE IPv4: 12.1.1.100
```

After attach, the `oaitun_ue1` interface should have IP `12.1.1.x` and can ping UPF at `12.1.1.1`

## Implementation Notes

- Derived from the EpiSci multi-UE proxy project
- The `open-nFAPI` directory contains modified copies of FAPI interface files from OAI's top-level `nfapi` directory
- Uses standalone Makefile, not integrated with OAI CMake build system
- Typically runs inside the gNB container in Kubernetes deployments
- gNB VNF and L2 Proxy can be started in any order
- Default nFAPI ports: P5 (50600-50601), P7 (50610-50611)

## Contact

For questions on the L2 FAPI Proxy:

- Russell Ford: russelldford@gmail.com
- Daoud Burghal: d.burghal@samsung.com
- Pranav Madadi: p.madadi@samsung.com
- Salim El Ghalbzouri: s.elghalbzo@partner.samsung.com / salim.elghalb@gmail.com
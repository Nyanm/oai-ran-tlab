# OAI L2 FAPI Proxy - Kubernetes Deployment

Complete Helm charts for deploying OAI 5G with L2 FAPI Proxy simulation on Kubernetes.

## Overview

This deployment enables **hardware-free** 5G RAN testing using the L2 FAPI Proxy, which implements the nFAPI protocol to connect gNB and UE components without physical radio:

- **gNB (VNF mode)**: Runs as Virtual Network Function, communicates via nFAPI
- **UE (PNF mode)**: Runs as Physical Network Function (emulated), connects to L2 Proxy  
- **Emulated L1**: PHY layer simulation enables protocol testing without actual radio hardware

## Chart Structure

```
helm-charts/
├── oai-5g-basic/          # 5G Core Network (umbrella chart)
├── mysql/                 # Database for core network
├── oai-nrf/              # Network Repository Function
├── oai-amf/              # Access and Mobility Management
├── oai-smf/              # Session Management Function
├── oai-upf/              # User Plane Function
├── oai-udm/              # Unified Data Management
├── oai-udr/              # Unified Data Repository
├── oai-ausf/             # Authentication Server Function
├── oai-lmf/              # Location Management Function
├── oai-traffic-server/   # Traffic generation server
├── oai-gnb-dev/          # gNB for L2 simulation (VNF mode)
└── oai-nr-ue-multi/      # UE for L2 simulation (PNF mode, multi-instance)
```

**13 charts total** - Complete standalone deployment.

---

## Prerequisites

### System Requirements
- **Kubernetes**: v1.24+ (K3s, minikube, or any distribution)
- **Helm**: 3.8+
- **Docker**: 20.10+
- **Resources**: Minimum 8GB RAM, 4 CPUs

### Software Requirements
- OAI RAN codebase with L2 FAPI Proxy built
- `kubectl` configured for your cluster
- Cluster with privileged pod support (for network namespace operations)

---

## Deployment Guide

### Phase 1: Environment Setup

#### 1.1 Create Kubernetes Namespace

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: oai
  labels:
    pod-security.kubernetes.io/enforce: privileged
    pod-security.kubernetes.io/warn: privileged
EOF
```

#### 1.2 Build OAI Container Images

```bash
# Clone OAI RAN repository (if not already done)
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g

# Build gNB development image (multi-stage build)
docker build --target ran-base-dev \
  --tag ran-base-dev:latest \
  --file docker/Dockerfile.base.dev.ubuntu22 .

docker build --target ran-build-dev \
  --tag ran-build-dev:latest \
  --file docker/Dockerfile.build.dev.ubuntu22 .

docker build --target oai-gnb-dev \
  --tag oai-gnb-dev:latest \
  --file docker/Dockerfile.gNB.dev.ubuntu22 .
```

#### 1.3 Import Images to Kubernetes Registry

```bash
# For K3s
docker save oai-gnb-dev:latest | sudo k3s ctr images import -

# For minikube
minikube image load oai-gnb-dev:latest

# For kind
kind load docker-image oai-gnb-dev:latest

# Verify import
kubectl run test --image=oai-gnb-dev:latest --command -- sleep 3600 -n oai
kubectl delete pod test -n oai
```

---

### Phase 2: Core Network Deployment

#### 2.1 Build Chart Dependencies

```bash
cd l2-fapi-proxy/helm-charts/

# Build dependencies for core network chart
helm dependency build oai-5g-basic/
```

This packages all 10 sub-charts (mysql + 9 core network functions) into `oai-5g-basic/charts/`.

#### 2.2 Deploy Core Network

```bash
helm install core5g oai-5g-basic/ --namespace oai
```

#### 2.3 Verify Core Network

```bash
# Check all pods are Running (may take 2-3 minutes)
kubectl get pods -n oai

# Expected output:
# NAME                                READY   STATUS    RESTARTS   AGE
# core5g-mysql-0                      1/1     Running   0          2m
# core5g-oai-amf-...                  1/1     Running   0          2m
# core5g-oai-smf-...                  1/1     Running   0          2m
# core5g-oai-upf-...                  1/1     Running   0          2m
# ... (additional core network pods)

# Check Helm release
helm list -n oai
```

---

### Phase 3: RAN Component Configuration

**CRITICAL**: Before deploying, you **MUST** configure the path to your OAI RAN code on the host machine.

#### 3.1 Configure gNB Chart

Edit `oai-gnb-dev/values.yaml` and update line 87:

```yaml
config:
  # Change from placeholder to your actual path:
  oaiCodePath: /home/YOUR_USERNAME/openairinterface5g  # <<< CHANGE THIS
```

**Example**: If you cloned OAI to `/home/salim/openairinterface5g`:
```yaml
config:
  oaiCodePath: /home/salim/openairinterface5g
```

**Note**: 
- This is the **HOST machine path** where your OAI code exists
- It will be automatically mounted to `/opt/oai-ran` inside the container
- The deployment template already has the volume mount configured

#### 3.2 Configure UE Chart

Edit `oai-nr-ue-multi/values.yaml` and update line 42:

```yaml
config:
  # Use the SAME path as gNB:
  oaiCodePath: /home/YOUR_USERNAME/openairinterface5g  # <<< CHANGE THIS
```

**Optional**: To deploy multiple UE instances, edit the `ueIds` array (line 3):
```yaml
ueIds: ["00", "01"]  # Deploy 2 UEs (UE-00 and UE-01)
```

---

### Phase 4: RAN Deployment

#### 4.1 Deploy gNB

```bash
helm install gnb oai-gnb-dev/ --namespace oai

# Verify deployment
kubectl get pods -n oai | grep gnb
```

**Note**: `start.gnb: false` in values.yaml means the pod runs in **sleep mode** for manual execution.

#### 4.2 Deploy UE(s)

```bash
helm install ue oai-nr-ue-multi/ --namespace oai

# Verify deployment (should see number matching replicaCount)
kubectl get pods -n oai | grep ue
```

---

### Phase 5: L2 FAPI Proxy Execution

#### 5.1 Build L2 Proxy (First Time Only)

```bash
# Get gNB pod name
GNB_POD=$(kubectl get pods -n oai -l app.kubernetes.io/name=oai-gnb-dev -o jsonpath='{.items[0].metadata.name}')

# Enter gNB container
kubectl exec -it $GNB_POD -n oai -- /bin/bash

# Inside container: Build proxy
cd /opt/oai-ran/l2-fapi-proxy
make

# Executable will be at: /opt/oai-ran/l2-fapi-proxy/build/proxy
```

#### 5.2 Build gNB Softmodem (First Time Only)

```bash
# Still inside gNB container
cd /opt/oai-ran
source oaienv

cd cmake_targets
./build_oai -I --install-optional-packages

./build_oai -c --gNB --nrUE \
  --build-lib "telnetsrv" \
  -t Ethernet \
  --ninja
```

**Subsequent builds** (after code changes):
```bash
cd /opt/oai-ran/cmake_targets/ran_build/build
cmake --build . --target nr-softmodem nr-uesoftmodem -- -j$(nproc)
```

#### 5.3 Start L2 Proxy Server

Open a new terminal and keep it running:

```bash
kubectl exec -it $GNB_POD -n oai -- /bin/bash

cd /opt/oai-ran/l2-fapi-proxy

# Use the helper script (automatically discovers pod IPs)
./src/run_l2_proxy.sh 1 2  # 1 gNB, 2 UEs
```

**Expected output**:
```
[VNF] Binding to 0.0.0.0:50001
[VNF] Waiting for connections...
```

#### 5.4 Start gNB Softmodem

Open another terminal:

```bash
kubectl exec -it $GNB_POD -n oai -- /bin/bash

cd /opt/oai-ran/cmake_targets/ran_build/build
./nr-softmodem -O /tmp/gnb.conf --sa --nfapi VNF --emulate-l1
```

**Expected output**:
```
[GTPU] Configuring GTPu address : 10.X.X.X, port : 2152
[NGAP] Received NGSetupResponse from AMF
[VNF] Sent NFAPI_VNF_CONFIG_REQ
```

#### 5.5 Start UE Softmodem(s)

For each UE pod:

```bash
# Get UE pod names
UE_PODS=($(kubectl get pods -n oai -l app=oai-nr-ue-multi --no-headers -o custom-columns=":metadata.name"))

# Start first UE (UE-00, node-number 2)
kubectl exec -it ${UE_PODS[0]} -n oai -- /bin/bash
cd /opt/oai-ran/cmake_targets/ran_build/build
./nr-uesoftmodem -O /tmp/nr-ue.conf --nfapi STANDALONE_PNF --emulate-l1 \
  --node-number 2 -r 106 --numerology 1 -C 3619200000

# In another terminal, start second UE (UE-01, node-number 3)
kubectl exec -it ${UE_PODS[1]} -n oai -- /bin/bash
cd /opt/oai-ran/cmake_targets/ran_build/build
./nr-uesoftmodem -O /tmp/nr-ue.conf --nfapi STANDALONE_PNF --emulate-l1 \
  --node-number 3 -r 106 --numerology 1 -C 3619200000
```

**Important**: `--node-number` must be `2 + UE_ID` (UE-00 uses 2, UE-01 uses 3, etc.)

**Expected output**:
```
[PNF] Connecting to VNF...
[PNF] Connected to L2 Proxy
[NR_MAC] UE synchronized to gNB
```

---

## Verification

### Check UE Attachment

```bash
# In gNB logs, look for:
[NR_MAC] UE RNTI 1234 Context Setup
[NGAP] INITIAL_CONTEXT_SETUP_REQUEST

# In UE logs:
[NAS] Registration Accept received
[NAS] PDU Session Establishment Accept
```

### Traffic Test

```bash
# Inside UE pod
kubectl exec -it $UE_POD -n oai -- /bin/bash

# Check interface
ip addr show oaitun_ue1

# Ping data network
ping -I oaitun_ue1 8.8.8.8
```

---

## Configuration Reference

### gNB Key Parameters (`oai-gnb-dev/values.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config.l2Fapi` | `true` | Enable L2 FAPI mode |
| `config.l2FapiAdditionalOpts` | `--sa --nfapi VNF --emulate-l1` | nFAPI VNF options |
| `config.oaiCodePath` | `/opt/oai-ran/` | Container mount path |
| `config.amfhost` | `oai-amf` | AMF service name |
| `start.gnb` | `false` | Auto-start (keep false for L2 sim) |

### UE Key Parameters (`oai-nr-ue-multi/values.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config.l2Fapi` | `true` | Enable L2 FAPI mode |
| `config.l2FapiAdditionalOpts` | `--nfapi STANDALONE_PNF --emulate-l1` | nFAPI PNF options |
| `replicaCount` | `1` | Number of UE instances |
| `config.oaiCodePath` | `/opt/oai-ran/` | Container mount path |
| `start.nrue` | `false` | Auto-start (keep false for L2 sim) |

---

## Troubleshooting

### Pods Not Starting

```bash
# Check events
kubectl describe pod <pod-name> -n oai

# Common issues:
# - Image pull errors: Verify image imported to cluster registry
# - Volume mount errors: Check hostPath exists and has correct permissions
# - Resource constraints: Ensure cluster has sufficient CPU/memory
```

### L2 Proxy Connection Failures

```bash
# Verify pod IPs
kubectl get pods -n oai -o wide

# Check connectivity between pods
kubectl exec -it $GNB_POD -n oai -- ping <UE_POD_IP>

# Verify proxy is listening
kubectl exec -it $GNB_POD -n oai -- netstat -tuln | grep 50001
```

### UE Not Attaching

```bash
# Check gNB logs
kubectl logs $GNB_POD -n oai --tail=100

# Check AMF connectivity
kubectl exec -it $GNB_POD -n oai -- ping oai-amf

# Verify configuration alignment (MCC/MNC/TAC/SST/SD)
# between gNB, UE, and core network
```

### Build Failures

```bash
# Clean build
cd /opt/oai-ran/cmake_targets
rm -rf ran_build/
./build_oai -c --gNB --nrUE -t Ethernet --ninja
```

---

## Clean Up

```bash
# Uninstall Helm releases
helm uninstall ue -n oai
helm uninstall gnb -n oai  
helm uninstall core5g -n oai

# Delete namespace (removes all resources)
kubectl delete namespace oai
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │           5G Core Network (oai-5g-basic)        │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │   │
│  │  │ AMF  │ │ SMF  │ │ UPF  │ │ NRF  │ │ UDM  │  │   │
│  │  └───┬──┘ └───┬──┘ └───┬──┘ └──────┘ └──────┘  │   │
│  └──────│────────│────────│──────────────────────┘   │
│         │ N2     │ N4     │ N3                         │
│  ┌──────▼────────▼────────▼──────────────────────┐   │
│  │         gNB Pod (oai-gnb-dev)                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐  │   │
│  │  │  nr-softmodem   │    │   L2 Proxy      │  │   │
│  │  │   (VNF mode)    │◄───┤   nFAPI Server  │  │   │
│  │  └─────────────────┘    └────────┬────────┘  │   │
│  │         /opt/oai-ran (hostPath mount)        │   │
│  └──────────────────────────────────│───────────┘   │
│                                      │ nFAPI          │
│  ┌───────────────────────────────── │──────────┐   │
│  │      UE Pods (oai-nr-ue-multi)   │          │   │
│  │  ┌──────────────────┐    ┌───────▼────────┐ │   │
│  │  │ nr-uesoftmodem   │    │ nr-uesoftmodem │ │   │
│  │  │  (PNF mode)      │    │  (PNF mode)    │ │   │
│  │  │  UE Instance 1   │    │  UE Instance 2 │ │   │
│  │  └──────────────────┘    └────────────────┘ │   │
│  │         /opt/oai-ran (hostPath mount)        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Additional Resources

- **Main L2 Proxy README**: [../README.md](../README.md)
- **OAI Documentation**: https://gitlab.eurecom.fr/oai/openairinterface5g/-/wikis/home
- **nFAPI Specification**: https://scf.io/en/documents/082_-_nFAPI_and_FAPI_specifications.php
- **Helm Documentation**: https://helm.sh/docs/

---

## License

Licensed under OAI Public License v1.1 - see main repository LICENSE file.

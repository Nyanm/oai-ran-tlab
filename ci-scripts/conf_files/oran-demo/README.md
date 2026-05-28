# O-RAN 7.2 Split Demo

Setup for running RU+UE on Peafowl and DU on Broadbill.

---

## Step 1: Environment and Repository

```bash
cd ~
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git oru-demo
cd oru-demo
git checkout oru-demo-branch
```

Save as `~/oru-demo/testenv`:

```bash
export TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DPDK_INST=$TEST_DIR/dpdk-stable-20.11.9
export C_INCLUDE_PATH="$DPDK_INST/include"
```

---

## Step 2: Build DPDK

```bash
cd ~
wget http://fast.dpdk.org/rel/dpdk-20.11.9.tar.xz
tar xf dpdk-20.11.9.tar.xz
mv dpdk-stable-20.11.9 ~/oru-demo/dpdk-stable-20.11.9
rm dpdk-20.11.9.tar.xz
cd ~/oru-demo/dpdk-stable-20.11.9
meson setup --prefix=$(pwd) build
sudo ninja -C build install
```

---

## Step 3: Build xRAN

```bash
git clone https://gerrit.o-ran-sc.org/r/o-du/phy.git ~/oru-demo/phy-f-1.0
cd ~/oru-demo/phy-f-1.0
git checkout oran_f_release_v1.0
git apply ~/oru-demo/cmake_targets/tools/oran_fhi_integration_patches/F/oru.patch
git apply ~/oru-demo/cmake_targets/tools/oran_fhi_integration_patches/F/oaioran_F.patch
cd ~/oru-demo/phy-f-1.0/fhi_lib/lib
TARGET=x86 WIRELESS_SDK_TOOLCHAIN=gcc \
  RTE_SDK=~/oru-demo/dpdk-stable-20.11.9 \
  XRAN_DIR=~/oru-demo/phy-f-1.0/fhi_lib \
  make XRAN_LIB_SO=1
```

---

## Step 4: Build OAI

```bash
source ~/oru-demo/testenv
mkdir -p ~/oru-demo/build && cd ~/oru-demo/build

PKG_CONFIG_PATH=$DPDK_INST/lib/x86_64-linux-gnu/pkgconfig/ \
  cmake .. -GNinja \
  -DOAI_FHI72=ON \
  -Dxran_LOCATION=$TEST_DIR/phy-f-1.0/fhi_lib/lib \
  -DENABLE_WEBSRV=OFF -DENABLE_NRSCOPE=OFF -DENABLE_ENBSCOPE=OFF \
  -DENABLE_UESCOPE=OFF -DENABLE_IMSCOPE=OFF \
  -DENABLE_LDPC_CUDA=OFF -DENABLE_LDPC_AAL=OFF -DENABLE_LDPC_XDMA=OFF

PKG_CONFIG_PATH=$DPDK_INST/lib/x86_64-linux-gnu/pkgconfig/ \
  cmake --build . --target oran_fhlib_5g

PKG_CONFIG_PATH=$DPDK_INST/lib/x86_64-linux-gnu/pkgconfig/ \
  cmake --build . --target params_libconfig

PKG_CONFIG_PATH=$DPDK_INST/lib/x86_64-linux-gnu/pkgconfig/ \
  cmake --build . --target vrtsim oairu oran_fhlib_5g nr-softmodem nr-oru nr-uesoftmodem
```

---

## Step 5: SR-IOV Setup

Run after every reboot. Scripts are in this folder.

Peafowl:
```bash
bash sriov-peafowl.sh
```

Broadbill:
```bash
bash sriov-broadbill.sh
```

---

## Step 6: Run

Broadbill (DU):
```bash
source ~/oru-demo/testenv
cd ~/oru-demo/build
sudo taskset -c 6-24 env LD_LIBRARY_PATH=.:$DPDK_INST/lib/x86_64-linux-gnu/ \
  ./nr-softmodem -O ../../ci-scripts/conf_files/oran-demo/du-demo.conf \
  --thread-pool 16,17,18,19,20,21,22,23,24 \
  --gNBs.[0].min_rxtxtime 6
```

Peafowl (RU):
```bash
source ~/oru-demo/testenv
cd ~/oru-demo/build
sudo taskset -c 6-24 env LD_LIBRARY_PATH=.:$DPDK_INST/lib/x86_64-linux-gnu/ \
  ./nr-oru -O ../../ci-scripts/conf_files/oran-demo/ru-demo.conf --vrtsim.role server
```

Peafowl (UE):
```bash
source ~/oru-demo/testenv
cd ~/oru-demo/build
sudo taskset -c 25-31 env LD_LIBRARY_PATH=.:$DPDK_INST/lib/x86_64-linux-gnu/ \
  ./nr-uesoftmodem -C 4049760000 -r 106 --numerology 1 --band 77 --ssb 516 \
  --device.name vrtsim --vrtsim.role client --ue-nb-ant-tx 1 --ue-nb-ant-rx 1
```

Config file for RU: `ci-scripts/conf_files/oran-demo/ru-demo.conf`
Config file for DU: `ci-scripts/conf_files/oran-demo/du-demo.conf`

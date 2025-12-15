## Option 1: Work via Emitter

### About

Precomputed 3GPP 38.901 TDL profiles (A to E) with statistical MIMO channels, plus a Python emitter that publishes taps to VRTSIM over a nanomsg PUB socket using FlatBuffers. Supports multiple antenna shapes (up to 64x64), speed presets (walking about 1.5 m/s, car about 30 m/s), spatial correlation, mutual coupling, LOS steering for D and E, and time-varying lognormal shadowing.

**Key Features:**
* Precomputed 3GPP 38.901 TDL taps (A to E) stored in a CIR database, published by a Python emitter over a PUB socket using FlatBuffers.
* Avoids on-the-fly channel synthesis during runs, which reduces CPU load and improves timing stability.
* Database contains profiles, delay spreads, antenna shapes, and speed presets, so scenarios are reproducible and easy to share.
* VRTSim consumes taps from the socket and applies them per slot, which decouples tap generation from application.
* Channel model is editable in the external repo, so new profiles or motion models can be added without touching VRTSim.

### Clone Channel Emulator
```bash
git clone https://gitlab.eurecom.fr/oai/raytracing-channel-emulator.git
```

### Setup Schema
```bash
python3 -m pip install --user flatbuffers nanomsg numpy
cd ~/raytracing-channel-emulator/server/api
flatc --python -o . taps.fbs    # generates ./Phy/Taps.py
```

### Generate CIR DB
```bash
cd ~/raytracing-channel-emulator/server/external_taps
python3 CIR_generator.py --out ./cir_db.bin --demo
```

### Run CIR Emitter
```bash
python3 emit_from_db.py --bind tcp://127.0.0.1:5555 --model TDL-A --ds-ns 10 --nrx 1 --ntx 1 --interval 0.5
```

### Clone OAI
```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g
git fetch origin vrtsim_cirdb_read:vrtsim_cirdb_read
git checkout vrtsim_cirdb_read
```

### Build
```bash
cd openairinterface5g/cmake_targets
cd build 
cmake ../.. -GNinja   -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DOAI_VRTSIM_TAPS_CLIENT=ON   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build . --target taps_client vrtsim rfsimulator nr-softmodem nr-uesoftmodem ldpc params_libconfig -j"$(nproc)"
```

### Run gNB
```bash
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --device.name vrtsim --vrtsim.role server --vrtsim.taps-socket tcp://127.0.0.1:5555 --gNBs.[0].min_rxtxtime 6
```

### Run UE
```bash
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim --vrtsim.role client
```

---

## Option 2: Read DB Directly (In-process CIRDB)

### About

**Key Features:**
* Adds an in-process CIR database path in VRTSim that applies taps internally, so no external emitter is required and no socket exchange occurs.
* The dataset is produced offline using the CIR DB generator in the channel-emulator repo, then consumed by VRTSim for deterministic replay.
* Avoids on-the-fly channel synthesis during runs, which reduces CPU load and improves timing stability.
* Reduces transport jitter and context switches compared to a socket publisher, which helps at higher PRB, MIMO, and UE counts.

### Run gNB
```bash
sudo ./nr-softmodem \
  -O ./../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
  --device.name vrtsim \
  --vrtsim.role server \
  --gNBs.[0].min_rxtxtime 3 \
  --vrtsim.cirdb 1 \
  --vrtsim.cirdb_yaml /openairinterface5g/radio/vrtsim/cir_db.yaml \
  --vrtsim.cirdb_file /openairinterface5g/radio/vrtsim/cir_db.bin \
  --vrtsim.cirdb_model_id 0 \
  --vrtsim.cirdb_ds_ns 10 \
  --vrtsim.cirdb_speed_mps 1.5
```

### Run UE
```bash
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim --vrtsim.role client
```

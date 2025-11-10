# About

Precomputed 3GPP 38.901 TDL profiles (A to E) with statistical MIMO channels, plus a Python emitter that publishes taps to VRTSIM over a nanomsg PUB socket using FlatBuffers. Supports multiple antenna shapes (up to 64x64), speed presets (walking about 1.5 m/s, car about 30 m/s), spatial correlation, mutual coupling, LOS steering for D and E, and time-varying lognormal shadowing. 

---

## 1) Setup

    python3 -m pip install --user flatbuffers nanomsg numpy pyyaml
    cd ~/raytracing-channel-emulator/server/api
    flatc --python -o . taps.fbs    # generates ./Phy/Taps.py
    export PYTHONPATH="$(pwd):$PYTHONPATH"

    # switch to external_taps for the tools below
    cd ~/raytracing-channel-emulator/server/external_taps

---

## 2) Generate the CIR file

Default profiles, delay spreads, antenna shapes, and speed presets are embedded.  
Shadowing is enabled by default, lognormal per link, time varying across snapshots via AR(1).

This command writes:
- binary taps file: ./cir_db.bin
- YAML sidecar with per-entry metadata: ./cir_db.yaml (used by the emitter to resolve selections and offsets)

    python3 CIR_generator.py --out ./cir_db.bin

To change shadowing, pass --shadow-sigma-db and optionally --shadow-tau-c-s for the correlation time. Set sigma to 0 to disable.

---

## 3) Run the emitter

Sequential replay, one snapshot every 0.5 s:

    python3 emit_from_db.py --bind tcp://127.0.0.1:5555 --db ./cir_db.bin --model TDL-A --ds-ns 30 --nrx 1 --ntx 1 --interval 0.5

List what is in the database:

    python3 emit_from_db.py --db ./cir_db.bin --describe-db

Optional, print the selected entry before streaming:

    python3 emit_from_db.py --db ./cir_db.bin --print-selected


---

## 4) Run OAI with VRTSIM

gNB (server role):

    sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --device.name vrtsim --vrtsim.role server -vrtsim.taps-socket tcp://127.0.0.1:5555 --gNBs.[0].min_rxtxtime 3

UE (client role):

    sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim --vrtsim.role client

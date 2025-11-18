# About

This tutorial will showcase the usage fo channel emulation server to provide realistic ray-traced channel to the softmodems

# Compiling

Reuse the previous step binaries

```
cmake ../../ -DOAI_VRTSIM_TAPS_CLIENT=ON
```

# Running

```
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --device.name vrtsim --vrtsim.role server --vrtsim.taps-socket ipc:///tmp/ru_socket_0
```

```
source .venv/bin/activate
python main.py example_config.yaml 
```

```
sudo ./nr-uesoftmodem  -C 3619200000 -r 106 --numerology 1 --ssb 516 --device.name vrtsim
```

```
./move_ue.sh 
```

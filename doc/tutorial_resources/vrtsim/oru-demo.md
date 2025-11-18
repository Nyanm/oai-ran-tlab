# About

Just a demonstration of what we need vrtsim for

# Running

```
sudo -E LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu/:. -E ASAN_OPTIONS=detect_odr_violation=0  ./oainr_ru -O ~/ru.band77.106prb.fhi72.2x2.conf  --vrtsim.role server
```

```
sudo -E LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu/:. -E ASAN_OPTIONS=detect_odr_violation=0 ./nr-softmodem -O ~/gnb.2x2.conf  --gNBs.[0].min_rxtxtime 6
```

```
sudo -E LD_LIBRARY_PATH=. ./nr-uesoftmodem -C 4049760000 -r 106 --numerology 1 --ssb 516 --band 77  --device.name vrtsim
```

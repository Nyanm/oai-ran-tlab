This directory contains example configuration files for running O-RU + O-DU + UE

For O-DU:

```
sudo ./nr-softmodem -O du.2x2.conf  --gNBs.[0].min_rxtxtime 6
```

For O-RU:
```
sudo ./oainr_ru -O ru.2x2.conf  --vrtsim.role server
```

For UE:

```
sudo ./nr-uesoftmodem -C 4049760000 -r 106 --numerology 1 --ssb 516 --band 77  --device.name vrtsim 
```

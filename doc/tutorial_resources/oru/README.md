# Reproduction steps to run OAI O-RU with NR UE on broadbill with gNB on peafowl

1. Reserve both machines using the `auto_resrve` script

2. Compile the code on both machines. Example configuration:

```
cmake ../../ -GNinja -DOAI_FHI72=ON -Dxran_LOCATION=/home/bpodrygajlo/phy/fhi_lib/lib
cmake --build . --target vrtsim oainr_ru oran_fhlib_5g nr-softmodem ldpc params_libconfig nr-uesoftmodem
```

3. On broadbill, run `setup_ru_ifs_broadbill.sh`

4. On peafowl, run `setup_ifs_peafowl.sh`

5. On broadbill, run the O-RU:

```
sudo ./oainr_ru -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/ru.band77.106prb.fhi72.4x4.conf --vrtsim.role server
```

and UE 

```
sudo taskset -c 50-63 ./nr-uesoftmodem -C 4049760000 -r 106 --numerology 1 --band 77 --ssb 516 --device.name vrtsim
```

6. On peafowl, run the gNB:

```
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band77.106prb.fhi72.4x4-oairu.conf
```

7. Observe UE detecting SSB and decoding SIB1

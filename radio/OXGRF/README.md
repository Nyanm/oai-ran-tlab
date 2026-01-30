[[_TOC_]]

# OXGRF

## Build drivers

There are two methods to build OXGRF, as detailed below:

- using `build_oai` script

```bash
sudo ./build_oai -I -w oxgrf
```

- Manual Installation

```bash
# install drivers
git clone https://github.com/openxg-prog/oxgrf_driver /tmp/oxgrf_driver
cd /tmp/oxgrf_driver/XDMA/linux-kernel/xdma
make && sudo make install

# install liboxgrf
git clone -b 2505 https://github.com/openxg-prog/liboxgrf /tmp/liboxgrf
cd /tmp/liboxgrf
sudo cp oxgrf_api.h /usr/local/include/
sudo cp liboxgrf.so /usr/local/lib/
sudo ldconfig
```

## Configuration

Here is an example of the RUs configuration for a setup with OXGRF. The value **1488** in `sdr_addrs = "dev=pciex:0,auxdac1=1488"` represents the inherent frequency offset of the OXGRF board, which can be obtained directly from the physical board. Ensure you replace **1488** with the actual frequency offset value of your specific OXGRF board.

```bash
RUs = (
{
  local_rf       = "yes"
  nb_tx          = 1;
  nb_rx          = 1;
  att_tx         = 0;
  att_rx         = 0;
  bands          = [78];
  max_pdschReferenceSignalPower = -27;
  max_rxgain                    = 60;
  eNB_instances  = [0];
  sl_ahead       = 2;
  # clock_src = "internal";
  sdr_addrs = "dev=pciex:0,auxdac1=1488";
}
);
```

Example files can be found in the `targets/PROJECTS/GENERIC-NR-5GC/CONF/` directory with a`oxgrf` in the name, for instance[`gnb.sa.band78.fr1.273PRB.oxgrf.conf`](../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.273PRB.oxgrf.conf).

## Run

### gNB

To launch gNB with OXGRF, use the following command:

```bash
sudo ./ran_build/build/nr-softmodem -O ../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.273PRB.oxgrf.conf
```

### UE

Use the following command to start the UE with OXGRF:

```bash
sudo ./ran_build/build/nr-uesoftmodem -C 3649380000 -r 273 --numerology 1 --ssb 1472 --oxgrf-args "dev=pciex:0,auxdac1=1620" -O ../targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf
```

> When running the UE with OXGRF, the board's frequency offset value must be explicitly specified in the UE launch command via the `--oxgrf-args` parameter.

## Uninstall

```bash
cd oxgrf_driver/XDMA/linux-kernel/xdma
make uninstall
```

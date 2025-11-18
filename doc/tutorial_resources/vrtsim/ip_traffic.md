# About

In this step using `iperf` to test the modems performance is explained.

# Running

## Option 1: --noS1 mode

Select this option on slower machines. This doesn't require core network.

1. Run gNB

```bash
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --device.name vrtsim --vrtsim.role server --noS1 --do-ra
```

2. Create UE namespace

```bash
sudo ip netns add ue_ns
```

2. Run UE in the new namespace

```bash
sudo ip netns exec ue_ns bash
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim --noS1 --do-ra
```

3. Run iperf server in the UE namespace

```bash
sudo ip netns exec ue_ns bash
iperf -s -B 10.0.1.2
```

4. Run iperf client in host namespace

```bash
iperf -c 10.0.1.2 -B 10.0.1.1
```

## Option 2: with core network

1. Start `cn` in `doc/tutorial_resources/oai-cn5g` with `docker compose up`

2. Run gNB

```bash
sudo ./nr-softmodem -O ../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --device.name vrtsim --vrtsim.role server
```

3. Verify connection to core network in gNB stdout

```
[NGAP]   Send NGSetupRequest to AMF
[NGAP]   3584 -> 0000e000
[NGAP]   Served GUAMIs for AMF OAI-AMF (assoc_id=3):
[NGAP]    GUAMI:
[NGAP]      PLMN: MCC=001, MNC=01
[NGAP]      AMF Region ID: 1
[NGAP]      AMF Set ID: 1
[NGAP]      AMF Pointer: 1
[NGAP]   Supported PLMN 0: MCC=001 MNC=01
[NGAP]   Supported slice (PLMN 0): SST=0x01 SD=000
[NGAP]   Received NGSetupResponse from AMF
```

4. Connect the UE. 

```bash
sudo ./nr-uesoftmodem -C 3619200000 -r 106 --numerology 1 --ssb 516 --band 78 --device.name vrtsim  --uicc0.imsi 001010000000001
```

5. Verify interface is created

```
[OIP]   Interface oaitun_ue1 successfully configured, IPv4 10.0.0.2, IPv6 (null)
```

```bash
ifconfig | grep tun   
```

6. Run iperf in oai-ext-dn container

```bash
docker exec -it oai-ext-dn bash
iperf -c 10.0.0.2 -B 192.168.70.135
```

## Troubleshooting

Use the same steps as from `first_steps.md`. This test adds additional requirements
on the CPU which might cause the UE to fail.

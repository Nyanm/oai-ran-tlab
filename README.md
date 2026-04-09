# OAI-SL-MultiUE-Broker



## RFSim Broker based Multiple UEs Test 

This part test is based on ‘sl-eurecom4’ branch on openairinterface5g.
(https://gitlab.eurecom.fr/oai/openairinterface5g.git)


## To start

First go to the correct repository :
```
cd openairinterface5g/cmake_targets
```
To install zeroMQ : 
```
apt-get install libzmq3-dev
```
Then start compilation : 
```
sudo ./build_oai --nrUE -w SIMU --cmake-opt -DENABLE_T_TRACER=OFF
```
Or, to start a clean start compilation, do the clean up first then compile : 
```
sudo ./build_oai -c -C  

sudo ./build_oai -I --cmake-opt -DENABLE_T_TRACER=OFF
```
 Now we need to create three UEs through namespaces: 
```
cd openairinterface5g/cmake_targets
sudo ./multi-ue.sh -c1 -c2 -c3
```
* It is likely the new system would face the ASN.1 error, which is due to the wrong version of ASN.1. To fix it, please do the following: 

* Remove the current wrong version :
```
sudo rm -rf /opt/asn1c 
sudo rm -rf /tmp/asn1c
```
* Clone and checkout the exact same commit:
```
cd /tmp 
git clone https://github.com/mouse07410/asn1c.git 
cd asn1c 
git checkout 657f5790
```
* Build and install the correct version : 
```
git submodule update --init --recursive 
autoreconf -iv 
./configure --prefix=/opt/asn1c 
make -j$(nproc) 
sudo make install
```
* Then recompile. Do remember to delete the wrong /ran_build files : 
```
cd ~/OAI-SL-Broker/openairinterface5g/cmake_targets 
rm -rf ran_build 
sudo ./build_oai --nrUE -w SIMU --cmake-opt -DENABLE_T_TRACER=OFF
```
 


 
## Start RFSim for Multiple UEs through Broker


### Launch Broker (outside namespace):
```
cd ran_build/build/
./broker
```

### Launch SYNC-REF (namespace 1):
```
sudo ./multi-ue.sh -o1
cd ran_build/build/
sudo RFSIMULATOR=server ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf --sl-mode 2 --sa --sync-ref --brokerip 10.201.1.100
```

### Launch UE2 (namespace 2):
```
sudo ./multi-ue.sh -o2
cd ran_build/build/
sudo ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue1.conf --sl-mode 2 --sa --brokerip 10.202.1.100 --device_id 1 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log 
```

### Launch UE3 (namespace 3):
```
sudo ./multi-ue.sh -o3
cd ran_build/build/
 sudo ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue2.conf --sl-mode 2 --sa --brokerip 10.203.1.100 --device_id 2 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log
```

Optional : You can now quickly check the connection to sync-ref from two UEs : 
```
ping -I oaitun_ue2 10.0.0.1
ping -I oaitun_ue3 10.0.0.1
```

Note: when restart the simulation, you might need to kill everything first by the following command : 
``` 
sudo pkill -f nr-uesoftmodem 
sudo pkill -f broker
sudo pkill -f rfsimulator
```

##  Monitor route with Babel and Wireshark
You might need to first install Babel onto the local machine

```
sudo apt update
sudo apt install babeld
```

### Launch SYNC-REF (namespace 1):
First go into the correct namespace : 
```
sudo ./multi-ue.sh -o1
```

In order to have a readable address at the logside, we'll hardcode the IP address through following command, so sync-ref will be translate as 'fe80::1/64'
```
sudo ip netns exec ue1 ip -6 addr add fe80::1/64 dev oaitun_ue1 nodad  
```
Then launch Babel on sync-ref : 
```
ip netns exec ue1 bash -lc '
set -e
cat > /tmp/babeld-ue1.conf <<EOF
interface oaitun_ue1
redistribute local deny
pid-file /tmp/babeld-ue1.pid
EOF
babeld -d 1 -c /tmp/babeld-ue1.conf
'
```
For now, as no other UE is connected, sync-ref will simply print its local id.

### Launch UE2 (namespace 2):
First go into the correct namespace : 
```
sudo ./multi-ue.sh -o2
```

To make it easier to observe, UE2 will be allocated address as 'fe80::2/64' by following command : 
```
sudo ip netns exec ue2 ip -6 addr add fe80::2/64 dev oaitun_ue2 nodad
```
Then launch Babel on UE2 : 
```
ip netns exec ue2 bash -lc '
set -e
cat > /tmp/babeld-ue2.conf <<EOF
interface oaitun_ue2
redistribute local deny
pid-file /tmp/babeld-ue2.pid
EOF
babeld -d 1 -c /tmp/babeld-ue2.conf
'
```

### Launch UE3 (namespace 3):
First go into the correct namespace : 
```
sudo ./multi-ue.sh -o3
```

UE3 will be allocated address as 'fe80::3/64'
```
sudo ip netns exec ue3 ip -6 addr add fe80::3/64 dev oaitun_ue3 nodad

```
Then launch Babel on UE3 : 
```
ip netns exec ue3 bash -lc '
set -e
cat > /tmp/babeld-ue3.conf <<EOF
interface oaitun_ue3
redistribute local deny
pid-file /tmp/babeld-ue3.pid
EOF
babeld -d 1 -c /tmp/babeld-ue3.conf
'

```

Note : On babel, it will show neighbour connection with 'Neighbour ... rxcost 96 txcost 65535...', when it's 96, it means perfect connection, 65535 means a bad connection, if the connection is bad when generating the UE3, try to kill and restart again.

### Wireshark monitor
Within the each namespace, you can start wireshark with filter on each node to check the Babel messages (example as for sync-ref, please update the oaitun_ueX name for each UE): 
```
sudo wireshark -i oaitun_ue1 -k &
```
You should be able to see 'Babel Hello' message, and 'Babel Hello ihu'. ihu representing 'I hear you'.
When there are two UEs connecting to sync-ref at the same time, you should be able to see 'Babel Hello ihu ihu'


## Enable TAP in RFSim for Multiple UEs
This part is to switch from default TUN to TAP. 
The baseline is, by default, system is running through TUN. In order to enable TAP, when you run each node simply add a flag at the begining: " OAI_TUNTAP_MODE=tap".

Optional : To disable the TAP (in case of an uncleaned restart...), do the following : 
```
unset OAI_TUNTAP_MODE
```
### Launch Broker (outside namespace):
```
cd ran_build/build/
./broker
```
### Launch SYNC-REF (namespace 1):
```
sudo ./multi-ue.sh -o1
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap RFSIMULATOR=server ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf --sl-mode 2 --sa --sync-ref --brokerip 10.201.1.100
```

### Launch UE2 (namespace 2):
```
sudo ./multi-ue.sh -o2
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue1.conf --sl-mode 2 --sa --brokerip 10.202.1.100 --device_id 1 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log
```

### Launch UE3 (namespace 3):
```
sudo ./multi-ue.sh -o3
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue2.conf --sl-mode 2 --sa --brokerip 10.203.1.100 --device_id 2 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log
```

## Check connection with Batman

### Start Batman on sync-ref (namespace 1):
```
sudo ./multi-ue.sh -o1
ip link set oaitun_ue1 address 02:00:00:00:00:01
sudo batctl if add oaitun_ue1
```

### Start Batman on UE2 (namespace 2):
```
sudo ./multi-ue.sh -o2
ip link set oaitun_ue2 address 02:00:00:00:00:02
sudo batctl if add oaitun_ue2
```


### Start Batman on UE3 (namespace 3):
```
sudo ./multi-ue.sh -o3
ip link set oaitun_ue3 address 02:00:00:00:00:03
sudo batctl if add oaitun_ue3
```
### Batman command
To check the neighbour : 
```
batctl n
```
To check the routing table : 
```
batctl o
```

### Start wireshark with filter
You can check the current packets exchange with wireshark, please update the name 'oaitun_ueX' accordingly. 
```
wireshark -k -i oaitun_ueX -Y "eth.type == 0x4305"
```

## HW with TAP 
Similarly, we can start the HW with TAP with the following command 
 
### Launch SYNC-REF (machine 1):
```
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf --sa -E --sl-mode 2 --sync-ref  --usrp-args "type=b200,master_clock_rate=46.08e6,enable_gps=true" --ue-txgain 10 --ue-rxgain 100 --thread-pool -1,-1,-1,-1 --clock-source 2 --time-source 2
```

### Launch UE2 (machine 2):
```
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue1.conf --sa -E --sl-mode 2 --usrp-args "type=b200,master_clock_rate=46.08e6,enable_gps=true" --ue-txgain 10 --ue-rxgain 100 --thread-pool -1,-1,-1,-1 --clock-source 2 --time-source 2
```

### Launch UE3 (machine 3):
```
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue2.conf --sa -E --sl-mode 2 --usrp-args "type=b200,master_clock_rate=46.08e6,enable_gps=true" --ue-txgain 10 --ue-rxgain 100 --thread-pool -1,-1,-1,-1 --clock-source 2 --time-source 2
```

## More UEs generated with TAP 
The newest update enables to generate more than 2 UEs along with one sync-ref. The following is an example of 3 UEs.
One would, first, need to create a new .conf file in openairinterface5g/targets/PROJECTS/NR-SIDELINK/, second, in file /openairinterface5g/openair2/LAYER2/NR_MAC_UE/mac_defs.h, the parameter of "#define CUR_SL_UE_CONNECTIONS" need to be updated to match the max number of UEs(Exclusive of sync-ref, this value is limited to 6 though)
### Launch Broker (outside namespace):
```
cd ran_build/build/
./broker
```
### Launch SYNC-REF (namespace 1):
```
sudo ./multi-ue.sh -o1
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap RFSIMULATOR=server ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/sl_sync_ref.conf --sl-mode 2 --sa --sync-ref --brokerip 10.201.1.100  --thread-pool -1,-1
```

### Launch UE2 (namespace 2):
```
sudo ./multi-ue.sh -o2
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue1.conf --sl-mode 2 --sa --brokerip 10.202.1.100 --device_id 1  --thread-pool -1,-1 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log 
```

### Launch UE3 (namespace 3):
```
sudo ./multi-ue.sh -o3
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue2.conf --sl-mode 2 --sa --brokerip 10.203.1.100 --device_id 2  --thread-pool -1,-1 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log
```

### Launch UE4 (namespace 4):
```
sudo ./multi-ue.sh -o4
cd ran_build/build/
sudo OAI_TUNTAP_MODE=tap ./nr-uesoftmodem --rfsim -O ../../../targets/PROJECTS/NR-SIDELINK/CONF/ue3.conf --sl-mode 2 --sa --brokerip 10.204.1.100 --device_id 3  --thread-pool -1,-1 | sudo tee $HOME/sl-oai-release4/rat-selection/logs/sl.log
```

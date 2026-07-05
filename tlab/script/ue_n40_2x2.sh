sudo ~/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem \
    -O ~/6g/ue.conf \
    -r 106 --numerology 1 --band 40 -C 2349750000 \
    --ue-fo-compensation -E \
    --clock-source 2 --time-source 2 \
    --ue-nb-ant-rx 2 --ue-nb-ant-tx 2 \
    --usrp-args "addr=192.168.40.2"

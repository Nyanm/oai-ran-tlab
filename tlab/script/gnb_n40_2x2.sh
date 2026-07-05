sudo ~/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem \
    -O ~/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band40.fr1.106PRB.2x2.usrpx310.conf \
    --gNBs.[0].min_rxtxtime 6 \
    --usrp-tx-thread-config 1 \
    -E \
    --continuous-tx \
    --T_stdout 2 --T_nowait


#!/bin/bash
if [ "$#" -lt 2 ]; then
    echo "Too few arguments. Usage: ./run_l2_proxy.sh NUM_GNBS NUM_UES [CH_TRACE_FILE]"
    exit
fi

NUM_GNB=$1
NUM_UE=$2

echo "Starting L2 Proxy for $NUM_GNB gNBs and $NUM_UE UEs"

PROXY_IP=$(hostname -i)
echo "Proxy IP = $PROXY_IP"

GNB_IP_LIST=""
for GNB_IDX in $(seq 0 $((NUM_GNB-1)))
do
    GNB_ID=$(printf "oai-gnb-%02d" $GNB_IDX)
    GNB_IP=$(getent hosts $GNB_ID | awk '{print $1}')
    GNB_IP_LIST+="$GNB_IP "
    # echo "IP of GNB $GNB_ID = $GNB_IP"
done

UE_IP_LIST=""
for UE_IDX in $(seq 0 $((NUM_UE-1)))
do
    UE_ID=$(printf "oai-ue-%02d" $UE_IDX)
    UE_IP=$(getent hosts $UE_ID | awk '{print $1}')
    UE_IP_LIST+="$UE_IP "
    # echo "IP of UE $UE_ID = $UE_IP"
done

if [ "$#" -eq 3 ]; then
    CH_TRACE_FILE=$3
    echo "Channel trace file: $CH_TRACE_FILE"
    CMD="/opt/oai-ran/l2-fapi-proxy/build/proxy --ch $CH_TRACE_FILE --gnb $GNB_IP_LIST --proxy $PROXY_IP --ue $UE_IP_LIST"
else
    CMD="/opt/oai-ran/l2-fapi-proxy/build/proxy --gnb $GNB_IP_LIST --proxy $PROXY_IP --ue $UE_IP_LIST"
fi

echo "Running command: $CMD"
exec $CMD


# for POD in $UE_POD_LIST
# do
#     UE_ID=$(echo $POD | cut -d '-' -f5)
#     IPERF_PORT="52$UE_ID"
#     echo "Connecting UE $UE_ID iperf client to traffic server $TRAFFIC_SERV_DNN_IP port $IPERF_PORT..."
#     #kubectl exec -n oai $POD -- iperf3 -c $TRAFFIC_SERV_DNN_IP -p $IPERF_PORT -t 0 --logfile /tmp/iperf.log &
# done

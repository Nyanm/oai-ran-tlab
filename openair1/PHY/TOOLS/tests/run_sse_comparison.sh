#!/bin/bash

# A script to build and compare the performance and output of four different
# versions of the multipath_channel function.

# Note: This script assumes two executables exist:
# 1. test_cpu_std  (Compiled WITHOUT -DCHANNEL_SSE)
# 2. test_cpu_sse  (Compiled WITH    -DCHANNEL_SSE)

rm -f test_channel.bin

echo "--- Full Channel Implementation Comparison ---"
echo ">>> Generating reference channel file..."
./test_cpu_std > /dev/null 2>&1
echo "------------------------------------------------------------------"


echo ">>> Running benchmarks..."
std_double_out=$(./test_cpu_std --use-channel --mode std_double 2>/dev/null | grep "Mode:")
sse_double_out=$(./test_cpu_sse --use-channel --mode sse_double 2>/dev/null | grep "Mode:")
std_float_out=$(./test_cpu_std --use-channel --mode std_float 2>/dev/null | grep "Mode:")
sse_float_out=$(./test_cpu_sse --use-channel --mode sse_float 2>/dev/null | grep "Mode:")
echo "------------------------------------------------------------------"

time_std_double=$(echo $std_double_out | awk -F '[,:]' '{print $4}')
csum_std_double=$(echo $std_double_out | awk -F '[,:]' '{print $6}')

time_sse_double=$(echo $sse_double_out | awk -F '[,:]' '{print $4}')
csum_sse_double=$(echo $sse_double_out | awk -F '[,:]' '{print $6}')

time_std_float=$(echo $std_float_out | awk -F '[,:]' '{print $4}')
csum_std_float=$(echo $std_float_out | awk -F '[,:]' '{print $6}')

time_sse_float=$(echo $sse_float_out | awk -F '[,:]' '{print $4}')
csum_sse_float=$(echo $sse_float_out | awk -F '[,:]' '{print $6}')

speedup_double=$(echo "scale=2; $time_std_double / $time_sse_double" | bc)
speedup_float=$(echo "scale=2; $time_std_float / $time_sse_float" | bc)
speedup_std=$(echo "scale=2; $time_std_double / $time_std_float" | bc)
speedup_sse=$(echo "scale=2; $time_sse_double / $time_sse_float" | bc)

echo ">>> Comparison Results:"
printf "+-----------------------+---------------------+--------------------------+--------------------+\n"
printf "| %-21s | %-19s | %-24s | %-18s |\n" "Implementation" "Avg. Time (us)" "Output Checksum" "Notes"
printf "+-----------------------+---------------------+--------------------------+--------------------+\n"
printf "| %-21s | %-19s | %-24s | %-18s |\n" "Standard C (double)" "$time_std_double" "$csum_std_double" "Baseline"
printf "| %-21s | %-19s | %-24s | %-18s |\n" "SSE (double)" "$time_sse_double" "$csum_sse_double" "${speedup_double}x vs std"
printf "| %-21s | %-19s | %-24s | %-18s |\n" "Standard C (float)" "$time_std_float" "$csum_std_float" "${speedup_std}x vs double"
printf "| %-21s | %-19s | %-24s | %-18s |\n" "SSE (float)" "$time_sse_float" "$csum_sse_float" "${speedup_float}x vs std"
printf "+-----------------------+---------------------+--------------------------+--------------------+\n"

rm -f test_channel.bin
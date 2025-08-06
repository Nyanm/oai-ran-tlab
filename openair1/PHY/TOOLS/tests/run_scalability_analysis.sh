#!/bin/bash

# A script to automate the execution and analysis of the channel scalability benchmark.

# --- Configuration ---
BENCHMARK_EXE="./test_channel_scalability"
GPU_LOG="gpu_stats.csv"
BENCHMARK_LOG="benchmark_output.txt"
PASSTHROUGH_ARGS="$@"

# --- Check for Help Flag ---
if [[ " $@ " =~ " --help " ]] || [[ " $@ " =~ " -h " ]]; then
    $BENCHMARK_EXE --help
    exit 0
fi

rm -f $GPU_LOG $BENCHMARK_LOG
echo "--- Starting Automated Scalability Analysis ---"

# --- Step 1: Start GPU Monitoring ---
echo "Starting GPU monitor..."
# NEW: Added memory utilization, memory usage, and graphics clock speed to the query
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.gr \
           --format=csv,noheader,nounits -lms 100 -f $GPU_LOG &
NVSMI_PID=$!
sleep 1

# --- Step 2: Run the Benchmark ---
echo "Running benchmark executable with args: $PASSTHROUGH_ARGS"
$BENCHMARK_EXE $PASSTHROUGH_ARGS | tee $BENCHMARK_LOG

# --- Step 3: Stop GPU Monitoring ---
echo "Benchmark finished. Stopping GPU monitor."
kill $NVSMI_PID
wait $NVSMI_PID 2>/dev/null

# --- Step 4: Parse Results and Generate Report ---
echo ""
echo "--- Analysis Report ---"

# Parse the machine-readable KEY:VALUE output
cpu_time=$(grep "CPU_Total_Time_us" $BENCHMARK_LOG | cut -d':' -f2)
gpu_time=$(grep "GPU_Total_Time_us" $BENCHMARK_LOG | cut -d':' -f2)
avg_gpu_time=$(grep "GPU_Avg_Time_us" $BENCHMARK_LOG | cut -d':' -f2)
gpu_status=$(grep "GPU_Status" $BENCHMARK_LOG | cut -d':' -f2)
speedup=$(grep "Speedup" $BENCHMARK_LOG | cut -d':' -f2)

# Parse the GPU stats log, now with more columns
if [ -s "$GPU_LOG" ]; then
    # Use awk to calculate peak and average for each metric
    # Col 1: GPU Util, Col 2: Mem Util, Col 3: Mem Used, Col 4: Power, Col 5: Clock
    peak_gpu_util=$(awk -F, 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}' $GPU_LOG)
    avg_gpu_util=$(awk -F, '{gsub(/ /,""); sum+=$1} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    
    peak_mem_util=$(awk -F, 'BEGIN{max=0} {if ($2>max) max=$2} END{print max}' $GPU_LOG)
    avg_mem_util=$(awk -F, '{gsub(/ /,""); sum+=$2} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)

    peak_mem_used=$(awk -F, 'BEGIN{max=0} {if ($3>max) max=$3} END{print max}' $GPU_LOG)

    peak_power=$(awk -F, 'BEGIN{max=0} {if ($4>max) max=$4} END{print max}' $GPU_LOG)
    avg_power=$(awk -F, '{gsub(/ /,""); sum+=$4} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)

    avg_clock=$(awk -F, '{gsub(/ /,""); sum+=$5} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
else
    # Set defaults if the log is empty
    peak_gpu_util="N/A"; avg_gpu_util="N/A"; peak_mem_util="N/A"; avg_mem_util="N/A"; 
    peak_mem_used="N/A"; peak_power="N/A"; avg_power="N/A"; avg_clock="N/A";
fi

# --- Display Table ---
printf "+----------------------------------+--------------------------+\n"
printf "| %-32s | %-24s |\n" "Metric" "Value"
printf "+----------------------------------+--------------------------+\n"
if [ ! -z "$cpu_time" ]; then
    printf "| %-32s | %-24s |\n" "Total CPU Time (us)" "$cpu_time"
fi
if [ ! -z "$gpu_time" ]; then
    printf "| %-32s | %-24s |\n" "Total GPU Time (us)" "$gpu_time"
    printf "| %-32s | %-24s |\n" "Average GPU Time per Channel (us)" "$avg_gpu_time"
    printf "| %-32s | %-24s |\n" "Real-time Target (< 500 us)" "$gpu_status"
fi
if [ ! -z "$speedup" ]; then
    printf "| %-32s | %-24s |\n" "Speedup (CPU/GPU)" "$speedup"
fi
if [ "$peak_gpu_util" != "N/A" ]; then
    printf "+----------------------------------+--------------------------+\n"
    printf "| %-32s | %-24s |\n" "GPU Resource Usage" ""
    printf "+----------------------------------+--------------------------+\n"
    printf "| %-32s | %-24s |\n" "Peak GPU Utilization (%)" "$peak_gpu_util"
    printf "| %-32s | %-24.2f |\n" "Average GPU Utilization (%)" "$avg_gpu_util"
    printf "| %-32s | %-24s |\n" "Peak Memory Ctrl Utilization (%)" "$peak_mem_util"
    printf "| %-32s | %-24.2f |\n" "Average Memory Ctrl Util (%)" "$avg_mem_util"
    printf "| %-32s | %-24s |\n" "Peak Memory Used (MiB)" "$peak_mem_used"
    printf "| %-32s | %-24s |\n" "Peak Power Draw (W)" "$peak_power"
    printf "| %-32s | %-24.2f |\n" "Average Power Draw (W)" "$avg_power"
    printf "| %-32s | %-24.0f |\n" "Average Graphics Clock (MHz)" "$avg_clock"
fi
printf "+----------------------------------+--------------------------+\n"

# --- Final Cleanup ---
rm -f $GPU_LOG
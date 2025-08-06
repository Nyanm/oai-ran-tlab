#!/bin/bash

# A script to automate the execution and analysis of the channel scalability benchmark.
# It runs the benchmark, concurrently monitors GPU usage, and generates a consolidated report.

# --- Configuration ---
BENCHMARK_EXE="./test_channel_scalability"
GPU_LOG="gpu_stats.csv"
BENCHMARK_LOG="benchmark_output.txt"
PASSTHROUGH_ARGS="$@" # Pass all script arguments to the C executable

# --- Check for Help Flag ---
if [[ " $@ " =~ " --help " ]] || [[ " $@ " =~ " -h " ]]; then
    echo "Usage: ./run_scalability_analysis.sh [options]"
    echo "This script is a wrapper for the test_channel_scalability executable."
    echo "It accepts all the same arguments, such as:"
    echo "  --num-channels <N>, --n-tx <N>, --n-rx <N>, --fs <Hz>, --mu <N>, --mode <cpu|gpu|all>"
    echo ""
    # Run the executable with --help to show all its options
    $BENCHMARK_EXE --help
    exit 0
fi

# --- Pre-run Cleanup ---
rm -f $GPU_LOG $BENCHMARK_LOG

echo "--- Starting Automated Scalability Analysis ---"

# --- Step 1: Start GPU Monitoring in the Background ---
# The format is CSV with no headers and no units for easy parsing. Logging every 100ms.
echo "Starting GPU monitor..."
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits -lms 100 -f $GPU_LOG &
NVSMI_PID=$!

# Give nvidia-smi a moment to start up
sleep 1

# --- Step 2: Run the Benchmark Executable in the Foreground ---
# All arguments passed to this script are forwarded to the C program.
echo "Running benchmark executable with args: $PASSTHROUGH_ARGS"
$BENCHMARK_EXE $PASSTHROUGH_ARGS > $BENCHMARK_LOG

# --- Step 3: Stop GPU Monitoring ---
echo "Benchmark finished. Stopping GPU monitor."
kill $NVSMI_PID
# wait is used to prevent the "Terminated" message from cluttering the output
wait $NVSMI_PID 2>/dev/null

# --- Step 4: Parse Results and Generate Report ---
echo ""
echo "--- Analysis Report ---"

# Parse the main benchmark output
cpu_time=$(grep "Total CPU Time" $BENCHMARK_LOG | awk '{print $6}')
gpu_time=$(grep "Total GPU Time" $BENCHMARK_LOG | awk '{print $6}')
gpu_status=$(grep "GPU Status" $BENCHMARK_LOG | awk '{print $3}')
speedup=$(grep "Speedup" $BENCHMARK_LOG | awk '{print $2}')

# Check if the GPU log has data before trying to parse it
if [ -s "$GPU_LOG" ]; then
    # Use awk to calculate peak and average from the nvidia-smi log
    peak_gpu_util=$(awk -F, 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}' $GPU_LOG)
    avg_gpu_util=$(awk -F, '{sum+=$1} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    peak_power=$(awk -F, 'BEGIN{max=0} {if ($2>max) max=$2} END{print max}' $GPU_LOG)
    avg_power=$(awk -F, '{sum+=$2} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
else
    peak_gpu_util="N/A"
    avg_gpu_util="N/A"
    peak_power="N/A"
    avg_power="N/A"
fi

# Display the consolidated report
printf "+----------------------------------+--------------------------+\n"
printf "| %-32s | %-24s |\n" "Metric" "Value"
printf "+----------------------------------+--------------------------+\n"
if [ ! -z "$cpu_time" ]; then
    printf "| %-32s | %-24s |\n" "Total CPU Time (us)" "$cpu_time"
fi
if [ ! -z "$gpu_time" ]; then
    printf "| %-32s | %-24s |\n" "Total GPU Time (us)" "$gpu_time"
    printf "| %-32s | %-24s |\n" "Real-time Target (< 500 us)" "$gpu_status"
fi
if [ ! -z "$speedup" ]; then
    printf "| %-32s | %-24s |\n" "Speedup (CPU/GPU)" "$speedup"
fi
if [ "$peak_gpu_util" != "N/A" ]; then
    printf "| %-32s | %-24s |\n" "Peak GPU Utilization (%)" "$peak_gpu_util"
    printf "| %-32s | %-24.2f |\n" "Average GPU Utilization (%)" "$avg_gpu_util"
    printf "| %-32s | %-24s |\n" "Peak Power Draw (W)" "$peak_power"
    printf "| %-32s | %-24.2f |\n" "Average Power Draw (W)" "$avg_power"
fi
printf "+----------------------------------+--------------------------+\n"

# --- Final Cleanup ---
rm -f $GPU_LOG $BENCHMARK_LOG
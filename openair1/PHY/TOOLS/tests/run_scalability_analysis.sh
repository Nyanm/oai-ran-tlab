#!/bin/bash

# A script to automate the execution and analysis of the channel scalability benchmark.
# It runs the benchmark, concurrently monitors GPU usage, and generates a consolidated report.

# --- Configuration ---
BENCHMARK_EXE="./test_channel_scalability"
GPU_LOG="gpu_stats.csv"
BENCHMARK_LOG="benchmark_output.txt"
REPORT_FILE="analysis_report.txt" # File to save the final report
PASSTHROUGH_ARGS="$@" # Pass all script arguments to the C executable

# --- Check for Help Flag ---
if [[ " $@ " =~ " --help " ]] || [[ " $@ " =~ " -h " ]]; then
    echo "Usage: ./run_scalability_analysis.sh [options]"
    echo "This script is a wrapper for the test_channel_scalability executable."
    echo "It accepts all the same arguments, such as:"
    echo "  --num-channels <N>, --nb-tx <N>, --nb-rx <N>, --num-samples <N>"
    echo ""
    # Run the executable with --help to show all its options
    $BENCHMARK_EXE --help
    exit 0
fi

# --- Pre-run Cleanup ---
rm -f $GPU_LOG $BENCHMARK_LOG $REPORT_FILE

echo "--- Starting Automated Scalability Analysis ---"
echo "Configuration: $PASSTHROUGH_ARGS"

# --- Step 1: Start GPU Monitoring in the Background ---
# The format is CSV with no headers and no units for easy parsing. Logging every 100ms.
# NEW: Added memory.used, temperature.gpu, and clock speeds.
echo "Starting GPU monitor..."
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader,nounits -lms 100 -f $GPU_LOG &
NVSMI_PID=$!

# Give nvidia-smi a moment to start up
sleep 1

# --- Step 2: Run the Benchmark Executable in the Foreground ---
# All arguments passed to this script are forwarded to the C program.
echo "Running benchmark executable..."
$BENCHMARK_EXE $PASSTHROUGH_ARGS > $BENCHMARK_LOG

# --- Step 3: Stop GPU Monitoring ---
echo "Benchmark finished. Stopping GPU monitor."
kill $NVSMI_PID
# wait is used to prevent the "Terminated" message from cluttering the output
wait $NVSMI_PID 2>/dev/null

# --- Step 4: Parse Results and Generate Report ---
echo "Generating analysis report..."

# Parse the main benchmark output from the C program
# Note: This uses the table format we designed previously.
cpu_time=$(grep "Total CPU Time (us)" $BENCHMARK_LOG | awk -F'|' '{print $3}' | xargs)
gpu_time=$(grep "Total GPU Time (us)" $BENCHMARK_LOG | awk -F'|' '{print $3}' | xargs)
speedup=$(grep "Speedup (CPU/GPU)" $BENCHMARK_LOG | awk -F'|' '{print $3}' | xargs)

# Check if the GPU log has data before trying to parse it
if [ -s "$GPU_LOG" ]; then
    # Use awk to calculate peak and average from the nvidia-smi log
    # Original metrics
    peak_gpu_util=$(awk -F, 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}' $GPU_LOG)
    avg_gpu_util=$(awk -F, '{sum+=$1} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    peak_power=$(awk -F, 'BEGIN{max=0} {if ($2>max) max=$2} END{print max}' $GPU_LOG)
    avg_power=$(awk -F, '{sum+=$2} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    # NEW: Calculate stats for the new metrics
    peak_mem_used=$(awk -F, 'BEGIN{max=0} {if ($3>max) max=$3} END{print max}' $GPU_LOG)
    peak_temp=$(awk -F, 'BEGIN{max=0} {if ($4>max) max=$4} END{print max}' $GPU_LOG)
    avg_core_clock=$(awk -F, '{sum+=$5} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    avg_mem_clock=$(awk -F, '{sum+=$6} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
else
    # Set to N/A if no GPU stats were collected
    peak_gpu_util="N/A"; avg_gpu_util="N/A"; peak_power="N/A"; avg_power="N/A"
    peak_mem_used="N/A"; peak_temp="N/A"; avg_core_clock="N/A"; avg_mem_clock="N/A"
fi

# --- Step 5: Display and Save the Consolidated Report ---
# The `tee` command prints to the console AND appends to the report file.
{
    # Copy the configuration from the benchmark output directly into the report
    grep -A 7 "+--Configuration" $BENCHMARK_LOG
    # Print the performance table
    printf "+----------------------------------+--------------------------+\n"
    printf "| %-32s | %-24s |\n" "Performance Metric" "Value"
    printf "+----------------------------------+--------------------------+\n"
    if [ ! -z "$cpu_time" ]; then
        printf "| %-32s | %-24s |\n" "Total CPU Time (us)" "$cpu_time"
    fi
    if [ ! -z "$gpu_time" ]; then
        printf "| %-32s | %-24s |\n" "Total GPU Time (us)" "$gpu_time"
    fi
    if [ ! -z "$speedup" ]; then
        printf "| %-32s | %-24s |\n" "Speedup (CPU/GPU)" "$speedup"
    fi
    # Print GPU resource usage
    if [ "$peak_gpu_util" != "N/A" ]; then
        printf "+----------------------------------+--------------------------+\n"
        printf "| %-32s | %-24s |\n" "GPU Resource Usage" "Value"
        printf "+----------------------------------+--------------------------+\n"
        printf "| %-32s | %-24s |\n" "Peak GPU Utilization (%)" "$peak_gpu_util"
        printf "| %-32s | %-24.2f |\n" "Average GPU Utilization (%)" "$avg_gpu_util"
        printf "| %-32s | %-24s |\n" "Peak Memory Used (MiB)" "$peak_mem_used"
        printf "| %-32s | %-24s |\n" "Peak Power Draw (W)" "$peak_power"
        printf "| %-32s | %-24.2f |\n" "Average Power Draw (W)" "$avg_power"
        printf "| %-32s | %-24s |\n" "Peak Temperature (C)" "$peak_temp"
        printf "| %-32s | %-24.0f |\n" "Average Core Clock (MHz)" "$avg_core_clock"
        printf "| %-32s | %-24.0f |\n" "Average Memory Clock (MHz)" "$avg_mem_clock"
    fi
    printf "+----------------------------------+--------------------------+\n"

} | tee -a $REPORT_FILE

echo -e "\nReport saved to $REPORT_FILE"

# --- Final Cleanup ---
rm -f $GPU_LOG $BENCHMARK_LOG
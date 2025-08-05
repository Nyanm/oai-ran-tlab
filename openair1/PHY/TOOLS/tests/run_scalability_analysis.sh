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
echo "Starting GPU monitor..."
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader,nounits -lms 100 -f $GPU_LOG &
NVSMI_PID=$!

# Give nvidia-smi a moment to start up
sleep 1

# --- Step 2: Run the Benchmark Executable in the Foreground ---
echo "Running benchmark executable..."
$BENCHMARK_EXE $PASSTHROUGH_ARGS > $BENCHMARK_LOG

# --- Step 3: Stop GPU Monitoring ---
echo "Benchmark finished. Stopping GPU monitor."
kill $NVSMI_PID
wait $NVSMI_PID 2>/dev/null

# --- Step 4: Parse GPU Statistics ---
echo "Generating analysis report..."
if [ -s "$GPU_LOG" ]; then
    peak_gpu_util=$(awk -F, 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}' $GPU_LOG)
    avg_gpu_util=$(awk -F, '{sum+=$1} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    peak_power=$(awk -F, 'BEGIN{max=0} {if ($2>max) max=$2} END{print max}' $GPU_LOG)
    avg_power=$(awk -F, '{sum+=$2} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    peak_mem_used=$(awk -F, 'BEGIN{max=0} {if ($3>max) max=$3} END{print max}' $GPU_LOG)
    peak_temp=$(awk -F, 'BEGIN{max=0} {if ($4>max) max=$4} END{print max}' $GPU_LOG)
    avg_core_clock=$(awk -F, '{sum+=$5} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
    avg_mem_clock=$(awk -F, '{sum+=$6} END{if (NR>0) print sum/NR; else print 0}' $GPU_LOG)
else
    peak_gpu_util="N/A"; avg_gpu_util="N/A"; peak_power="N/A"; avg_power="N/A"
    peak_mem_used="N/A"; peak_temp="N/A"; avg_core_clock="N/A"; avg_mem_clock="N/A"
fi

# --- Step 5: Display and Save the Consolidated Report ---
# The `{...} | tee` command group prints all output to the console AND saves it to the report file.
{
    # First, print the entire pre-formatted report from the C benchmark.
    cat $BENCHMARK_LOG

    # Second, append the detailed GPU resource usage table.
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
        printf "+----------------------------------+--------------------------+\n"
    fi

} | tee $REPORT_FILE

echo -e "\nAnalysis complete. Full report saved to $REPORT_FILE"

# --- Final Cleanup ---
rm -f $GPU_LOG $BENCHMARK_LOG
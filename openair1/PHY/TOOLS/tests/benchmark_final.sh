#!/bin/bash

# ===================================================================================
# OAI 5G Channel Simulation - Final Benchmark Runner
#
# This script runs the fast, dedicated 'test_multipath' executable and saves
# its comprehensive report to a file.
#
# How to Use:
# 1. Ensure 'test_multipath.c' is updated with the latest version.
# 2. Re-compile the test:
#    make test_multipath
# 3. Run this script from the build directory:
#    ./benchmark_final.sh
# ===================================================================================

# --- Configuration ---
EXECUTABLE="./test_multipath"
REPORT_FILE="final_performance_report.txt"

# --- Script Start ---

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found."
    echo "Please build 'test_multipath' first: make test_multipath"
    exit 1
fi

echo "Running the comprehensive benchmark harness..."
echo "This will be much faster. Results will be saved to $REPORT_FILE"

# Run the executable and use 'tee' to show output on screen AND save to file
$EXECUTABLE | tee "$REPORT_FILE"

echo ""
echo "Benchmark complete. Full report saved to $REPORT_FILE"

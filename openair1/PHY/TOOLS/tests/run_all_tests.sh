#!/bin/bash

# A comprehensive script to automate benchmark tests, saving all results to a single file.

# --- Configuration ---
BENCHMARK_SCRIPT="./run_scalability_analysis.sh"
LOG_DIR="comprehensive_results_$(date +%F_%H-%M-%S)"
MODES_TO_TEST=("serial" "stream" "batch")

# MODIFIED: Define a single master log file for all results
MASTER_LOG_FILE="$LOG_DIR/all_results.txt"

# Baseline parameters for all tests
SAMPLES=122880
MIMO_CONFIG="-t 4 -r 4"
TRIALS="-n 25" # Number of trials for averaging

# --- Script Start ---
echo "Starting comprehensive benchmark run..."
mkdir -p "$LOG_DIR"
# Create the log file and add a header
echo "Benchmark run started at $(date)" > "$MASTER_LOG_FILE"
echo "All results will be saved to: $MASTER_LOG_FILE"
echo "------------------------------------------------------------------"


# --- Test Suite 1: Scalability vs. Channel Count ---
# MODIFIED: Appending a header for this test suite to the master log file
echo "### STARTING TEST SUITE 1: Scalability vs. Channel Count ###" | tee -a "$MASTER_LOG_FILE"
CHANNELS_TO_TEST=(16 256 1024)
CH_LENGTH="-l 32"

for c in "${CHANNELS_TO_TEST[@]}"; do
  for m in "${MODES_TO_TEST[@]}"; do

    COMMAND="$BENCHMARK_SCRIPT -c $c $MIMO_CONFIG -s $SAMPLES $CH_LENGTH -m $m $TRIALS"
    
    # MODIFIED: Group commands and append their output to the master log file
    {
      echo ""
      echo "--- Running with config: -c $c -m $m ---"
      $COMMAND
      echo "--- Finished config: -c $c -m $m ---"
    } >> "$MASTER_LOG_FILE" 2>&1
    
    echo "Done with config: -c $c -m $m. Results appended."
  done
done
echo "### TEST SUITE 1 COMPLETE ###" | tee -a "$MASTER_LOG_FILE"
echo "------------------------------------------------------------------"


# --- Test Suite 2: Performance vs. Channel Complexity ---
echo "### STARTING TEST SUITE 2: Performance vs. Channel Complexity ###" | tee -a "$MASTER_LOG_FILE"
CHANNELS_FOR_LEN_TEST="-c 1024"
LENGTHS_TO_TEST=(16 64 128)

for l in "${LENGTHS_TO_TEST[@]}"; do
  for m in "${MODES_TO_TEST[@]}"; do
    COMMAND="$BENCHMARK_SCRIPT $CHANNELS_FOR_LEN_TEST $MIMO_CONFIG -s $SAMPLES -l $l -m $m $TRIALS"
    
    {
      echo ""
      echo "--- Running with config: -l $l -m $m ---"
      $COMMAND
      echo "--- Finished config: -l $l -m $m ---"
    } >> "$MASTER_LOG_FILE" 2>&1
    
    echo "Done with config: -l $l -m $m. Results appended."
  done
done
echo "### TEST SUITE 2 COMPLETE ###" | tee -a "$MASTER_LOG_FILE"
echo "------------------------------------------------------------------"


# --- Test Suite 4: Performance vs. MIMO Configuration ---
echo "### STARTING TEST SUITE 4: Performance vs. MIMO Configuration ###" | tee -a "$MASTER_LOG_FILE"
MIMO_CONFIGS_TO_TEST=("-t 2 -r 2" "-t 4 -r 4" "-t 8 -r 8")
CHANNELS_FOR_MIMO_TEST="-c 1024"

for mimo in "${MIMO_CONFIGS_TO_TEST[@]}"; do
  for m in "${MODES_TO_TEST[@]}"; do
    COMMAND="$BENCHMARK_SCRIPT $CHANNELS_FOR_MIMO_TEST $mimo -s $SAMPLES -l 32 -m $m $TRIALS"

    {
      echo ""
      echo "--- Running with config: $mimo -m $m ---"
      $COMMAND
      echo "--- Finished config: $mimo -m $m ---"
    } >> "$MASTER_LOG_FILE" 2>&1

    echo "Done with config: $mimo -m $m. Results appended."
  done
done
echo "### TEST SUITE 4 COMPLETE ###" | tee -a "$MASTER_LOG_FILE"
echo "------------------------------------------------------------------"


# --- Test Suite 5: Performance vs. Signal Samples ---
echo "### STARTING TEST SUITE 5: Performance vs. Signal Samples ###" | tee -a "$MASTER_LOG_FILE"
SAMPLES_TO_TEST=(30720 61440 122880)
CHANNELS_FOR_SAMPLES_TEST="-c 1024"

for s in "${SAMPLES_TO_TEST[@]}"; do
  for m in "${MODES_TO_TEST[@]}"; do
    COMMAND="$BENCHMARK_SCRIPT $CHANNELS_FOR_SAMPLES_TEST $MIMO_CONFIG -s $s -l 32 -m $m $TRIALS"
    
    {
      echo ""
      echo "--- Running with config: -s $s -m $m ---"
      $COMMAND
      echo "--- Finished config: -s $s -m $m ---"
    } >> "$MASTER_LOG_FILE" 2>&1

    echo "Done with config: -s $s -m $m. Results appended."
  done
done
echo "### TEST SUITE 5 COMPLETE ###" | tee -a "$MASTER_LOG_FILE"
echo "------------------------------------------------------------------"

echo "All tests have finished. Master log file is located at: $MASTER_LOG_FILE"
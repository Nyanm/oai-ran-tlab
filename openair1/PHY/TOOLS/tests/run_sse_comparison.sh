#!/bin/bash

# A script to compare the performance and output of the standard C and SSE
# versions of the multipath_channel function using an identical channel.

echo "--- SSE vs. Standard C Channel Comparison ---"
echo ""
echo ">>> Generating reference channel with test_cpu_std..."
./test_cpu_std

echo ""
echo ">>> Running Standard C version with saved channel..."
./test_cpu_std --use-channel

echo ""
echo ">>> Running SSE version with saved channel..."
./test_cpu_sse --use-channel

echo ""
echo "Comparison complete. Compare the checksums and execution times above."
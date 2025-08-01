#!/bin/bash

# A script to compare the performance and output of the standard C and SSE
# versions of the multipath_channel function using an identical channel.

echo "--- SSE vs. Standard C Channel Comparison ---"

# Step 1: Build the test executables
echo "Building test executables..."
ninja test_cpu_std test_cpu_sse
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

# Step 2: Generate the reference channel using the standard C version
echo ""
echo ">>> Generating reference channel with test_cpu_std..."
./test_cpu_std

# Step 3: Run the standard C version again, this time using the saved channel
echo ""
echo ">>> Running Standard C version with saved channel..."
./test_cpu_std --use-channel

# Step 4: Run the SSE version using the exact same saved channel
echo ""
echo ">>> Running SSE version with saved channel..."
./test_cpu_sse --use-channel

echo ""
echo "Comparison complete. Compare the checksums and execution times above."
#!/bin/bash

# A wrapper script to execute a command inside the OAI development container.

set -e # Exit immediately if a command exits with a non-zero status.

# 1. Define Container and Project Paths
CONTAINER_NAME="oai-dev-ubuntu24"
PROJECT_ROOT_IN_CONTAINER="/opt/oai"

# 2. Check for Script Argument
if [ -z "$1" ]; then
    echo "Error: No script path provided."
    echo "Usage: $0 <path/to/script_to_execute.py>"
    echo "Example: $0 python/Examples/polartest.py"
    exit 1
fi

SCRIPT_PATH_ON_HOST=$1

# 2b. Ensure computation results are written to a dedicated folder in the
#     directory where this wrapper is executed.
RESULTS_DIR_ON_HOST="${PWD}/tmp_comp_results"
mkdir -p "${RESULTS_DIR_ON_HOST}"

# 3. Check if Container is Running
# The `docker ps -q` command returns the container ID if it's running, otherwise it's empty.
if ! docker ps -q -f "name=${CONTAINER_NAME}" | grep -q .; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please ensure the container is up with: docker-compose up -d"
    exit 1
fi

# 4. Construct the Full Path to the Script inside the Container
# The volume mounts the project root to /opt/oai, so we prepend that.
SCRIPT_PATH_IN_CONTAINER="${PROJECT_ROOT_IN_CONTAINER}/${SCRIPT_PATH_ON_HOST}"
RESULTS_DIR_IN_CONTAINER="${PROJECT_ROOT_IN_CONTAINER}/tmp_comp_results"

# 5. Execute the Script
echo "--- Executing '${SCRIPT_PATH_IN_CONTAINER}' in container '${CONTAINER_NAME}' ---"
echo "--- Writing computation results to '${RESULTS_DIR_ON_HOST}' ---"
echo ""

# The `docker exec` command runs the python3 interpreter on the script path inside the container.
# We set LD_LIBRARY_PATH to make sure the oaipylib shared object is found.
OAI_LIB_PATH="${PROJECT_ROOT_IN_CONTAINER}/build/python"
PYTHON_SRC_PATH="${PROJECT_ROOT_IN_CONTAINER}/python"
docker exec \
    --workdir "${RESULTS_DIR_IN_CONTAINER}" \
    --env PYTHONPATH=${OAI_LIB_PATH}:${PYTHON_SRC_PATH} \
    --env LD_LIBRARY_PATH=${OAI_LIB_PATH} \
    "${CONTAINER_NAME}" \
    python3 "${SCRIPT_PATH_IN_CONTAINER}"

echo ""
echo "--- Execution finished ---"

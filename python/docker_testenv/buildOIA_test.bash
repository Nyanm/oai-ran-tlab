#!/usr/bin/env bash
set -euo pipefail

rm -rf build
mkdir -p build
cd build

cmake .. -GNinja
ninja liboaipy dfts ldpc

install -m 755 libdfts.so python/libdfts.so
install -m 755 libldpc.so python/libldpc.so

# Make liboaipy discoverable without requiring manual LD_LIBRARY_PATH exports.
install -m 755 python/liboaipy.so /usr/local/lib/liboaipy.so
ldconfig

# Remove any stale local symlink that can shadow the real Python extension module.
rm -f python/oaipylib.so

cd ../python
python3 -m pip install --force-reinstall . --break-system-packages


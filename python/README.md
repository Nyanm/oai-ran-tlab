# oaipylib

Minimal CPython extension skeleton in C for wrapping the OAI C library.

## Layout

- `src/oai_lib_api.h`: placeholder header for API into OAI functions
- `src/oaipylib_wrappers.h`: Python wrapper declarations
- `src/oaipylib_wrappers.c`: Python/C wrapper implementations
- `src/oaipylib_module.c`: Python module definition
- `setup.py`: setuptools build script
- `pyproject.toml`: build-system config

## Building liboai

From OAI top-level

```bash
mkdir build
cd build
cmake .. -GNinja
ninja liboaipy dfts ldpc
cp libdfts.so python
cp libldpc.so python
export LD_LIBRARY_PATH="path to openairinterface5g/build/python":$LD_LIBRARY_PATH
```
## Build


```bash
python -m pip install -U pip setuptools wheel
python -m pip install .
```

## To test
```bash
python -c "import os ; import sys ; sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL) ; import oaipylib as oai ; oai.init() ; encoded_output = oai.nr_polar_encoder(0x12345678,0,0,0,32,0) ; print(encoded_output[0],hex(encoded_output[0]))"
```
The printed encoder output should match what is shown in the debugging traces (which will be turned off soon)

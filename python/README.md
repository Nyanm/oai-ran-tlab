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
python Examples/polartest.py
```
You should see that the output matches the input (0x12345678) at SNR=0dB. Debugging traces will be removed

## Docker test enviroment
There is a work in progress doocker test enviroment located in the dokcer_testenv folder. Currently there is an issue with kernel crashes of jupyter notebooks when running any testing scripts for the OAIPYLIB, these issues could be caused by the docker enviroment, thus it is not recemended to use this envroment for work on the library or any adjacent script. Until the cause of the kernel crashes is confirmed to be unrelated to the docker container setup 
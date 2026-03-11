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
ninja liboaipy

## Build


```bash
python -m pip install -U pip setuptools wheel
python -m pip install .

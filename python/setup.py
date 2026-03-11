from setuptools import Extension, setup

ext = Extension(
    "oaipylib",
    sources=[
        "src/oaipylib_module.c",
        "src/oaipylib_wrappers.c",
    ],
    include_dirs=["src"],
    libraries=["oaipy"],
    library_dirs=["../build/python"],
    extra_compile_args=["-O2"],
)

setup(
    name="oaipylib",
    version="0.1.0",
    ext_modules=[ext],
)

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pu_cka",
        ["src/pu/cpp/pycka.cpp", "src/pu/cpp/cka.cpp"],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="pu_cka",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
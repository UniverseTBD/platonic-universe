import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _build_ext_modules():
    # Allow opting out entirely (e.g. PU_SKIP_CPP=1 uv pip install .)
    if os.environ.get("PU_SKIP_CPP"):
        return []

    sources = ["src/pu/cpp/pycka.cpp", "src/pu/cpp/cka.cpp"]

    if sys.platform == "darwin":
        # Apple Clang doesn't ship OpenMP; try Homebrew libomp, otherwise skip.
        brew_prefixes = [
            os.environ.get("LIBOMP_PREFIX"),
            "/opt/homebrew/opt/libomp",   # Apple Silicon
            "/usr/local/opt/libomp",      # Intel
        ]
        libomp = next(
            (p for p in brew_prefixes if p and os.path.exists(os.path.join(p, "include", "omp.h"))),
            None,
        )
        if libomp is None:
            print(
                "pu: libomp not found; skipping pu_cka C++ extension build. "
                "Install via `brew install libomp` or set PU_SKIP_CPP=1 to silence.",
                file=sys.stderr,
            )
            return []
        extra_compile_args = [
            "-O3",
            "-std=c++17",
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{libomp}/include",
        ]
        extra_link_args = [f"-L{libomp}/lib", "-lomp"]
    else:
        extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]
        extra_link_args = ["-fopenmp"]

    return [
        Pybind11Extension(
            "pu_cka",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


setup(
    name="pu_cka",
    version="0.0.1",
    ext_modules=_build_ext_modules(),
    cmdclass={"build_ext": build_ext},
)

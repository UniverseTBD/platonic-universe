import platform
import subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def has_openmp():
    """Check if OpenMP is available."""
    if platform.system() == "Darwin":
        # macOS: check if libomp is installed via Homebrew
        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    return True  # Assume available on Linux


extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []

if has_openmp():
    if platform.system() == "Darwin":
        # macOS with Homebrew libomp
        import subprocess
        libomp_prefix = subprocess.run(
            ["brew", "--prefix", "libomp"],
            capture_output=True, text=True
        ).stdout.strip()
        extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{libomp_prefix}/include"]
        extra_link_args += [f"-L{libomp_prefix}/lib", "-lomp"]
    else:
        extra_compile_args += ["-fopenmp"]
        extra_link_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "pu_cka",
        ["src/pu/cpp/pycka.cpp", "src/pu/cpp/cka.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="pu_cka",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
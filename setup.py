from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

here = os.path.abspath(os.path.dirname(__file__))
eigen_include = os.path.join(here, "src", "pu", "cpp", "eigen")

# Common compile args for all extensions
common_compile_args = ["-O3", "-std=c++17", "-fopenmp", "-march=native"]
common_link_args = ["-fopenmp"]

ext_modules = [
    # CKA - Centered Kernel Alignment
    Pybind11Extension(
        "pu_cka",
        ["src/pu/cpp/pycka.cpp", "src/pu/cpp/cka.cpp"],
        extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    # Procrustes Distance
    Pybind11Extension(
        "pu_procrustes",
        ["src/pu/cpp/pyprocrustes.cpp", "src/pu/cpp/procrustes.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Cosine Similarity
    Pybind11Extension(
        "pu_cosine",
        ["src/pu/cpp/pycosine.cpp", "src/pu/cpp/cosine.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # RSA - Representational Similarity Analysis
    Pybind11Extension(
        "pu_rsa",
        ["src/pu/cpp/pyrsa.cpp", "src/pu/cpp/rsa.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Linear R2
    Pybind11Extension(
        "pu_linear_r2",
        ["src/pu/cpp/pylinear_r2.cpp", "src/pu/cpp/linear_r2.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # MMD - Maximum Mean Discrepancy
    Pybind11Extension(
        "pu_mmd",
        ["src/pu/cpp/pymmd.cpp", "src/pu/cpp/mmd.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Frechet Distance
    Pybind11Extension(
        "pu_frechet",
        ["src/pu/cpp/pyfrechet.cpp", "src/pu/cpp/frechet.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # SVCCA - Singular Vector Canonical Correlation Analysis
    Pybind11Extension(
        "pu_svcca",
        ["src/pu/cpp/pysvcca.cpp", "src/pu/cpp/svcca.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # PWCCA - Projection Weighted CCA
    Pybind11Extension(
        "pu_pwcca",
        ["src/pu/cpp/pypwcca.cpp", "src/pu/cpp/pwcca.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Tucker Congruence Coefficient
    Pybind11Extension(
        "pu_tucker",
        ["src/pu/cpp/pytucker.cpp", "src/pu/cpp/tucker.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Eigenspectrum Distance
    Pybind11Extension(
        "pu_eigenspectrum",
        ["src/pu/cpp/pyeigenspectrum.cpp", "src/pu/cpp/eigenspectrum.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Riemannian Distance
    Pybind11Extension(
        "pu_riemannian",
        ["src/pu/cpp/pyriemannian.cpp", "src/pu/cpp/riemannian.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # KL Divergence (Gaussian assumption)
    Pybind11Extension(
        "pu_kl_divergence",
        ["src/pu/cpp/pykl_divergence.cpp", "src/pu/cpp/kl_divergence.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # JS Divergence (Gaussian assumption)
    Pybind11Extension(
        "pu_js_divergence",
        ["src/pu/cpp/pyjs_divergence.cpp", "src/pu/cpp/js_divergence.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
    # Mutual Information (Gaussian assumption)
    Pybind11Extension(
        "pu_mutual_information",
        ["src/pu/cpp/pymutual_information.cpp", "src/pu/cpp/mutual_information.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    ),
]

setup(
    name="platonic-universe",
    version="0.1.0",
    description="Metrics for comparing neural network representations",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

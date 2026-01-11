#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "frechet.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_frechet, m) {
    m.doc() = "Fréchet Distance module - C++ implementation with Eigen";

    py::class_<pu::FrechetResult>(m, "FrechetResult")
        .def_readonly("distance", &pu::FrechetResult::distance)
        .def_readonly("mean1", &pu::FrechetResult::mean1)
        .def_readonly("mean2", &pu::FrechetResult::mean2)
        .def_readonly("trace_cov1", &pu::FrechetResult::trace_cov1)
        .def_readonly("trace_cov2", &pu::FrechetResult::trace_cov2)
        .def_readonly("trace_sqrt_product", &pu::FrechetResult::trace_sqrt_product);

    m.def("frechet_distance", &pu::frechet_distance,
          py::arg("X"), py::arg("Y"),
          "Compute Fréchet distance between two sample distributions");

    m.def("matrix_sqrt", &pu::matrix_sqrt,
          py::arg("A"),
          "Compute matrix square root using eigendecomposition");
}

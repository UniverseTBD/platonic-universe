#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "mmd.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_mmd, m) {
    m.doc() = "Maximum Mean Discrepancy module - C++ implementation with Eigen";

    py::class_<pu::MMDResult>(m, "MMDResult")
        .def_readonly("mmd_squared", &pu::MMDResult::mmd_squared)
        .def_readonly("mmd", &pu::MMDResult::mmd)
        .def_readonly("kernel", &pu::MMDResult::kernel)
        .def_readonly("gamma", &pu::MMDResult::gamma);

    m.def("mmd_squared", &pu::mmd_squared,
          py::arg("X"), py::arg("Y"),
          py::arg("kernel") = "rbf", py::arg("gamma") = -1.0,
          "Compute squared MMD (biased estimator)");

    m.def("mmd_unbiased", &pu::mmd_unbiased,
          py::arg("X"), py::arg("Y"),
          py::arg("kernel") = "rbf", py::arg("gamma") = -1.0,
          "Compute squared MMD (unbiased estimator)");

    m.def("rbf_kernel", &pu::rbf_kernel,
          py::arg("X"), py::arg("Y"), py::arg("gamma"),
          "Compute RBF kernel matrix");
}

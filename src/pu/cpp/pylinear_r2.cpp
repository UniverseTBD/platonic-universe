#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "linear_r2.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_linear_r2, m) {
    m.doc() = "Linear R² module - C++ implementation with Eigen";

    py::class_<pu::LinearR2Result>(m, "LinearR2Result")
        .def_readonly("r2", &pu::LinearR2Result::r2)
        .def_readonly("r2_x_to_y", &pu::LinearR2Result::r2_x_to_y)
        .def_readonly("r2_y_to_x", &pu::LinearR2Result::r2_y_to_x)
        .def_readonly("rmse", &pu::LinearR2Result::rmse);

    m.def("linear_r2", &pu::linear_r2, py::arg("X"), py::arg("Y"),
          "Compute linear R² from X to Y");

    m.def("bidirectional_linear_r2", &pu::bidirectional_linear_r2,
          py::arg("X"), py::arg("Y"),
          "Compute bidirectional linear R²");
}

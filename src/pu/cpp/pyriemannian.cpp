/**
 * pybind11 wrapper for Riemannian metrics
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "riemannian.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_riemannian, m) {
    m.doc() = "Riemannian geometry metrics for SPD manifolds";

    py::class_<pu::RiemannianResult>(m, "RiemannianResult")
        .def_readonly("affine_invariant_distance", &pu::RiemannianResult::affine_invariant_distance)
        .def_readonly("log_euclidean_distance", &pu::RiemannianResult::log_euclidean_distance)
        .def_readonly("stein_divergence", &pu::RiemannianResult::stein_divergence)
        .def("__repr__", [](const pu::RiemannianResult& r) {
            return "<RiemannianResult ai_dist=" + std::to_string(r.affine_invariant_distance) + ">";
        });

    m.def("compute_riemannian", &pu::compute_riemannian,
          py::arg("X"), py::arg("Y"), py::arg("eps") = 1e-10,
          "Compute Riemannian metrics between embedding matrices");
}

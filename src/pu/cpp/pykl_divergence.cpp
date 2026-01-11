/**
 * pybind11 wrapper for KL Divergence
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "kl_divergence.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_kl_divergence, m) {
    m.doc() = "KL Divergence for comparing embedding distributions";

    py::class_<pu::KLDivergenceResult>(m, "KLDivergenceResult")
        .def_readonly("kl_pq", &pu::KLDivergenceResult::kl_pq)
        .def_readonly("kl_qp", &pu::KLDivergenceResult::kl_qp)
        .def_readonly("symmetric_kl", &pu::KLDivergenceResult::symmetric_kl)
        .def_readonly("log_det_term", &pu::KLDivergenceResult::log_det_term)
        .def_readonly("trace_term", &pu::KLDivergenceResult::trace_term)
        .def_readonly("mahalanobis_term", &pu::KLDivergenceResult::mahalanobis_term)
        .def("__repr__", [](const pu::KLDivergenceResult& r) {
            return "<KLDivergenceResult kl_pq=" + std::to_string(r.kl_pq) +
                   " kl_qp=" + std::to_string(r.kl_qp) +
                   " symmetric=" + std::to_string(r.symmetric_kl) + ">";
        });

    m.def("compute_kl_divergence", &pu::compute_kl_divergence,
          py::arg("X"), py::arg("Y"), py::arg("eps") = 1e-10,
          "Compute KL divergence between embedding matrices (Gaussian assumption)");
}

/**
 * pybind11 wrapper for JS Divergence
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "js_divergence.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_js_divergence, m) {
    m.doc() = "Jensen-Shannon Divergence for comparing embedding distributions";

    py::class_<pu::JSDivergenceResult>(m, "JSDivergenceResult")
        .def_readonly("js_divergence", &pu::JSDivergenceResult::js_divergence)
        .def_readonly("js_divergence_normalized", &pu::JSDivergenceResult::js_divergence_normalized)
        .def_readonly("js_distance", &pu::JSDivergenceResult::js_distance)
        .def_readonly("kl_pm", &pu::JSDivergenceResult::kl_pm)
        .def_readonly("kl_qm", &pu::JSDivergenceResult::kl_qm)
        .def("__repr__", [](const pu::JSDivergenceResult& r) {
            return "<JSDivergenceResult js=" + std::to_string(r.js_divergence) +
                   " normalized=" + std::to_string(r.js_divergence_normalized) +
                   " distance=" + std::to_string(r.js_distance) + ">";
        });

    m.def("compute_js_divergence", &pu::compute_js_divergence,
          py::arg("X"), py::arg("Y"), py::arg("eps") = 1e-10,
          "Compute JS divergence between embedding matrices (Gaussian assumption)");
}

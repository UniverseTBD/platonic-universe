/**
 * pybind11 wrapper for Mutual Information
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "mutual_information.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_mutual_information, m) {
    m.doc() = "Mutual Information for measuring dependence between embedding spaces";

    py::class_<pu::MutualInformationResult>(m, "MutualInformationResult")
        .def_readonly("mutual_information", &pu::MutualInformationResult::mutual_information)
        .def_readonly("mutual_information_bits", &pu::MutualInformationResult::mutual_information_bits)
        .def_readonly("normalized_mi", &pu::MutualInformationResult::normalized_mi)
        .def_readonly("H_X", &pu::MutualInformationResult::H_X)
        .def_readonly("H_Y", &pu::MutualInformationResult::H_Y)
        .def_readonly("H_XY", &pu::MutualInformationResult::H_XY)
        .def("__repr__", [](const pu::MutualInformationResult& r) {
            return "<MutualInformationResult mi=" + std::to_string(r.mutual_information) +
                   " bits=" + std::to_string(r.mutual_information_bits) +
                   " normalized=" + std::to_string(r.normalized_mi) + ">";
        });

    m.def("compute_mutual_information", &pu::compute_mutual_information,
          py::arg("X"), py::arg("Y"), py::arg("eps") = 1e-10,
          "Compute mutual information between embedding matrices (Gaussian assumption)");
}

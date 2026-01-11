/**
 * pybind11 wrapper for Tucker congruence metric
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "tucker.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_tucker, m) {
    m.doc() = "Tucker congruence coefficient computation module";

    // Expose the TuckerResult struct
    py::class_<pu::TuckerResult>(m, "TuckerResult")
        .def_readonly("mean_congruence", &pu::TuckerResult::mean_congruence,
            "Mean Tucker congruence across all pairs")
        .def_readonly("median_congruence", &pu::TuckerResult::median_congruence,
            "Median Tucker congruence")
        .def_readonly("std_congruence", &pu::TuckerResult::std_congruence,
            "Standard deviation of congruences")
        .def_readonly("min_congruence", &pu::TuckerResult::min_congruence,
            "Minimum congruence value")
        .def_readonly("max_congruence", &pu::TuckerResult::max_congruence,
            "Maximum congruence value")
        .def_readonly("congruences", &pu::TuckerResult::congruences,
            "Vector of all congruence values")
        .def("__repr__", [](const pu::TuckerResult& r) {
            return "<TuckerResult mean=" + std::to_string(r.mean_congruence) +
                   " median=" + std::to_string(r.median_congruence) + ">";
        });

    // Expose functions
    m.def("tucker_congruence",
          &pu::tucker_congruence,
          py::arg("v1"),
          py::arg("v2"),
          "Compute Tucker congruence between two vectors");

    m.def("tucker_columnwise",
          &pu::tucker_columnwise,
          py::arg("X"),
          py::arg("Y"),
          R"doc(
          Compute column-wise Tucker congruence between two matrices.

          Parameters
          ----------
          X : numpy.ndarray
              First matrix (n_samples, n_features)
          Y : numpy.ndarray
              Second matrix (n_samples, n_features)

          Returns
          -------
          TuckerResult
              Statistics over all column pair congruences
          )doc");

    m.def("tucker_rowwise",
          &pu::tucker_rowwise,
          py::arg("X"),
          py::arg("Y"),
          R"doc(
          Compute row-wise Tucker congruence between two matrices.

          Parameters
          ----------
          X : numpy.ndarray
              First matrix (n_samples, n_features)
          Y : numpy.ndarray
              Second matrix (n_samples, n_features)

          Returns
          -------
          TuckerResult
              Statistics over all row pair congruences
          )doc");

    m.def("tucker_overall",
          &pu::tucker_overall,
          py::arg("X"),
          py::arg("Y"),
          R"doc(
          Compute overall Tucker congruence treating matrices as flattened vectors.

          Parameters
          ----------
          X : numpy.ndarray
              First matrix
          Y : numpy.ndarray
              Second matrix (must have same dimensions as X)

          Returns
          -------
          float
              Overall Tucker congruence coefficient
          )doc");
}

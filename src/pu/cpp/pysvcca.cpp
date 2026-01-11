/**
 * pybind11 wrapper for SVCCA metric
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "svcca.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_svcca, m) {
    m.doc() = "SVCCA (Singular Vector Canonical Correlation Analysis) computation module";

    // Expose the SVCCAResult struct
    py::class_<pu::SVCCAResult>(m, "SVCCAResult")
        .def_readonly("similarity", &pu::SVCCAResult::similarity,
            "Mean canonical correlation (SVCCA score)")
        .def_readonly("correlations", &pu::SVCCAResult::correlations,
            "Vector of canonical correlations")
        .def_readonly("n_components_x", &pu::SVCCAResult::n_components_x,
            "Number of components kept for first matrix")
        .def_readonly("n_components_y", &pu::SVCCAResult::n_components_y,
            "Number of components kept for second matrix")
        .def_readonly("variance_explained_x", &pu::SVCCAResult::variance_explained_x,
            "Fraction of variance explained by kept components (matrix 1)")
        .def_readonly("variance_explained_y", &pu::SVCCAResult::variance_explained_y,
            "Fraction of variance explained by kept components (matrix 2)")
        .def("__repr__", [](const pu::SVCCAResult& r) {
            return "<SVCCAResult similarity=" + std::to_string(r.similarity) +
                   " n_components=(" + std::to_string(r.n_components_x) + ", " +
                   std::to_string(r.n_components_y) + ")>";
        });

    // Expose the compute_svcca function
    m.def("compute_svcca",
          &pu::compute_svcca,
          py::arg("X"),
          py::arg("Y"),
          py::arg("variance_threshold") = 0.99,
          R"doc(
          Compute SVCCA (Singular Vector Canonical Correlation Analysis) between two embedding matrices.

          SVCCA = SVD preprocessing + CCA
          1. Performs SVD on each embedding matrix
          2. Keeps top components that explain variance_threshold of variance
          3. Runs CCA on the reduced representations

          Parameters
          ----------
          X : numpy.ndarray
              First embedding matrix of shape (n_samples, n_features1)
          Y : numpy.ndarray
              Second embedding matrix of shape (n_samples, n_features2)
          variance_threshold : float, optional
              Fraction of variance to retain (default: 0.99)

          Returns
          -------
          SVCCAResult
              Result object containing:
              - similarity: Mean canonical correlation (SVCCA score)
              - correlations: Vector of canonical correlations
              - n_components_x: Components kept for X
              - n_components_y: Components kept for Y
              - variance_explained_x: Variance explained for X
              - variance_explained_y: Variance explained for Y
          )doc");

    // Expose the compute_cca function
    m.def("compute_cca",
          &pu::compute_cca,
          py::arg("X"),
          py::arg("Y"),
          R"doc(
          Compute CCA (Canonical Correlation Analysis) between two matrices.

          Parameters
          ----------
          X : numpy.ndarray
              First matrix (n_features1, n_samples)
          Y : numpy.ndarray
              Second matrix (n_features2, n_samples)

          Returns
          -------
          numpy.ndarray
              Vector of canonical correlations
          )doc");
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "procrustes.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_procrustes, m) {
    m.doc() = "Procrustes analysis module - C++ implementation with Eigen";

    // Expose the result struct
    py::class_<pu::ProcrustesResult>(m, "ProcrustesResult")
        .def_readonly("distance", &pu::ProcrustesResult::distance)
        .def_readonly("alignment_score", &pu::ProcrustesResult::alignment_score)
        .def_readonly("rotation_matrix", &pu::ProcrustesResult::rotation_matrix)
        .def_readonly("mse", &pu::ProcrustesResult::mse)
        .def_readonly("frobenius_norm_residual", &pu::ProcrustesResult::frobenius_norm_residual);

    // Expose the main function
    m.def("compute_procrustes",
          &pu::compute_procrustes,
          py::arg("X"),
          py::arg("Y"),
          py::arg("scale") = false,
          R"pbdoc(
              Compute Procrustes analysis between two embedding matrices.

              Finds the optimal orthogonal transformation (rotation/reflection)
              to align Y to X, then measures the residual distance.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)
              scale : bool, optional
                  If True, scale matrices to unit norm before alignment (default: False)

              Returns
              -------
              ProcrustesResult
                  Object containing:
                  - distance: Normalized Procrustes distance [0, 1+]
                  - alignment_score: 1 - distance (higher is better)
                  - rotation_matrix: Optimal orthogonal transformation matrix
                  - mse: Mean squared error of residuals
                  - frobenius_norm_residual: Frobenius norm of residual matrix
          )pbdoc");

    // Expose convenience function
    m.def("procrustes_distance",
          &pu::procrustes_distance,
          py::arg("X"),
          py::arg("Y"),
          py::arg("scale") = false,
          R"pbdoc(
              Compute only the Procrustes distance between two embedding matrices.

              This is a convenience function that returns only the distance value.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)
              scale : bool, optional
                  If True, scale matrices to unit norm before alignment (default: False)

              Returns
              -------
              float
                  Normalized Procrustes distance
          )pbdoc");
}

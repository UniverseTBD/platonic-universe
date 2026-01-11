/**
 * pybind11 wrapper for Eigenspectrum metrics
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "eigenspectrum.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_eigenspectrum, m) {
    m.doc() = "Eigenspectrum comparison metrics module";

    // Expose the EigenspectrumResult struct
    py::class_<pu::EigenspectrumResult>(m, "EigenspectrumResult")
        .def_readonly("spectral_distance", &pu::EigenspectrumResult::spectral_distance,
            "L2 distance between eigenvalue spectra")
        .def_readonly("log_spectral_distance", &pu::EigenspectrumResult::log_spectral_distance,
            "L2 distance between log eigenvalue spectra (scale-invariant)")
        .def_readonly("spectral_similarity", &pu::EigenspectrumResult::spectral_similarity,
            "Cosine similarity between eigenvalue spectra")
        .def_readonly("effective_rank_x", &pu::EigenspectrumResult::effective_rank_x,
            "Effective rank of first matrix")
        .def_readonly("effective_rank_y", &pu::EigenspectrumResult::effective_rank_y,
            "Effective rank of second matrix")
        .def_readonly("effective_rank_distance", &pu::EigenspectrumResult::effective_rank_distance,
            "Absolute difference in effective ranks")
        .def_readonly("eigenvalues_x", &pu::EigenspectrumResult::eigenvalues_x,
            "Sorted eigenvalues of first matrix covariance")
        .def_readonly("eigenvalues_y", &pu::EigenspectrumResult::eigenvalues_y,
            "Sorted eigenvalues of second matrix covariance")
        .def("__repr__", [](const pu::EigenspectrumResult& r) {
            return "<EigenspectrumResult spectral_similarity=" +
                   std::to_string(r.spectral_similarity) +
                   " effective_ranks=(" + std::to_string(r.effective_rank_x) +
                   ", " + std::to_string(r.effective_rank_y) + ")>";
        });

    // Expose functions
    m.def("compute_eigenspectrum",
          &pu::compute_eigenspectrum,
          py::arg("X"),
          py::arg("Y"),
          py::arg("eps") = 1e-10,
          R"doc(
          Compute eigenspectrum-based similarity metrics between two embedding matrices.

          Parameters
          ----------
          X : numpy.ndarray
              First embedding matrix (n_samples, n_features)
          Y : numpy.ndarray
              Second embedding matrix (n_samples, n_features)
          eps : float
              Regularization parameter for numerical stability

          Returns
          -------
          EigenspectrumResult
              Contains spectral_distance, log_spectral_distance, spectral_similarity,
              effective_rank_x, effective_rank_y, effective_rank_distance,
              eigenvalues_x, eigenvalues_y
          )doc");

    m.def("compute_sorted_eigenvalues",
          &pu::compute_sorted_eigenvalues,
          py::arg("X"),
          py::arg("eps") = 1e-10,
          "Compute sorted eigenvalues of covariance matrix");

    m.def("compute_effective_rank",
          &pu::compute_effective_rank,
          py::arg("eigenvalues"),
          py::arg("eps") = 1e-10,
          "Compute effective rank from eigenvalues");
}

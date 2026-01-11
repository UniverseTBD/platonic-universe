#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "rsa.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_rsa, m) {
    m.doc() = "RSA (Representational Similarity Analysis) module - C++ implementation with Eigen";

    // Expose the result struct
    py::class_<pu::RSAResult>(m, "RSAResult")
        .def_readonly("correlation", &pu::RSAResult::correlation)
        .def_readonly("p_value", &pu::RSAResult::p_value)
        .def_readonly("method", &pu::RSAResult::method)
        .def_readonly("distance_metric", &pu::RSAResult::distance_metric);

    m.def("compute_rdm",
          &pu::compute_rdm,
          py::arg("embeddings"),
          py::arg("metric") = "correlation",
          R"pbdoc(
              Compute Representational Dissimilarity Matrix (RDM).

              Parameters
              ----------
              embeddings : numpy.ndarray
                  Embedding matrix of shape (n_samples, n_features)
              metric : str, optional
                  Distance metric: "correlation", "euclidean", "cosine" (default: "correlation")

              Returns
              -------
              numpy.ndarray
                  RDM matrix of shape (n_samples, n_samples)
          )pbdoc");

    m.def("rsa_similarity",
          &pu::rsa_similarity,
          py::arg("X"),
          py::arg("Y"),
          py::arg("method") = "spearman",
          py::arg("distance_metric") = "correlation",
          R"pbdoc(
              Compute RSA similarity between two embedding spaces.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)
              method : str, optional
                  Correlation method: "spearman" or "pearson" (default: "spearman")
              distance_metric : str, optional
                  Distance metric for RDM: "correlation", "euclidean", "cosine" (default: "correlation")

              Returns
              -------
              RSAResult
                  Object containing correlation, p_value, method, and distance_metric
          )pbdoc");

    m.def("spearman_correlation",
          &pu::spearman_correlation,
          py::arg("x"),
          py::arg("y"),
          "Compute Spearman rank correlation between two vectors");

    m.def("pearson_correlation",
          &pu::pearson_correlation,
          py::arg("x"),
          py::arg("y"),
          "Compute Pearson correlation between two vectors");
}

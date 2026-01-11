#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "cosine.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_cosine, m) {
    m.doc() = "Cosine similarity metrics module - C++ implementation with Eigen";

    // Expose the stats struct
    py::class_<pu::CosineStats>(m, "CosineStats")
        .def_readonly("mean", &pu::CosineStats::mean)
        .def_readonly("median", &pu::CosineStats::median)
        .def_readonly("std_dev", &pu::CosineStats::std_dev)
        .def_readonly("min_val", &pu::CosineStats::min_val)
        .def_readonly("max_val", &pu::CosineStats::max_val);

    // Expose the result struct
    py::class_<pu::CosineResult>(m, "CosineResult")
        .def_readonly("value", &pu::CosineResult::value)
        .def_readonly("stats", &pu::CosineResult::stats);

    // Expose functions
    m.def("mean_cosine_similarity",
          &pu::mean_cosine_similarity,
          py::arg("X"),
          py::arg("Y"),
          R"pbdoc(
              Compute mean cosine similarity between corresponding rows of two matrices.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)

              Returns
              -------
              CosineResult
                  Object containing value (mean similarity) and stats
          )pbdoc");

    m.def("mean_cosine_distance",
          &pu::mean_cosine_distance,
          py::arg("X"),
          py::arg("Y"),
          R"pbdoc(
              Compute mean cosine distance (1 - similarity) between corresponding rows.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)

              Returns
              -------
              CosineResult
                  Object containing value (mean distance) and stats
          )pbdoc");

    m.def("centered_cosine_similarity",
          &pu::centered_cosine_similarity,
          py::arg("X"),
          py::arg("Y"),
          R"pbdoc(
              Compute centered cosine similarity (center matrices before computing).

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)

              Returns
              -------
              float
                  Centered cosine similarity value
          )pbdoc");

    m.def("mean_angular_distance",
          &pu::mean_angular_distance,
          py::arg("X"),
          py::arg("Y"),
          R"pbdoc(
              Compute mean angular distance (arccos(cosine) / pi).

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)

              Returns
              -------
              CosineResult
                  Object containing value (mean angular distance) and stats
          )pbdoc");

    m.def("compute_cosine_similarities",
          &pu::compute_cosine_similarities,
          py::arg("X"),
          py::arg("Y"),
          R"pbdoc(
              Compute row-wise cosine similarities.

              Parameters
              ----------
              X : numpy.ndarray
                  First embedding matrix of shape (n_samples, n_features)
              Y : numpy.ndarray
                  Second embedding matrix of shape (n_samples, n_features)

              Returns
              -------
              numpy.ndarray
                  Vector of cosine similarities for each row pair
          )pbdoc");
}

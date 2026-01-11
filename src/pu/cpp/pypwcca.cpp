/**
 * pybind11 wrapper for PWCCA metric
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "pwcca.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_pwcca, m) {
    m.doc() = "PWCCA (Projection Weighted Canonical Correlation Analysis) module";

    // Expose the PWCCAResult struct
    py::class_<pu::PWCCAResult>(m, "PWCCAResult")
        .def_readonly("similarity", &pu::PWCCAResult::similarity,
            "PWCCA similarity score (weighted mean of canonical correlations)")
        .def_readonly("correlations", &pu::PWCCAResult::correlations,
            "Vector of canonical correlations")
        .def_readonly("weights", &pu::PWCCAResult::weights,
            "Projection weights for each canonical direction")
        .def("__repr__", [](const pu::PWCCAResult& r) {
            return "<PWCCAResult similarity=" + std::to_string(r.similarity) + ">";
        });

    // Expose the compute_pwcca function
    m.def("compute_pwcca",
          &pu::compute_pwcca,
          py::arg("X"),
          py::arg("Y"),
          R"doc(
          Compute PWCCA (Projection Weighted Canonical Correlation Analysis).

          PWCCA combines CCA with projection-based weighting that emphasizes
          the importance of each canonical direction based on the variance
          in the original space that it captures.

          Parameters
          ----------
          X : numpy.ndarray
              First embedding matrix of shape (n_samples, n_features1)
          Y : numpy.ndarray
              Second embedding matrix of shape (n_samples, n_features2)

          Returns
          -------
          PWCCAResult
              Result object containing:
              - similarity: PWCCA score (weighted mean of correlations)
              - correlations: Vector of canonical correlations
              - weights: Projection weights for each direction

          Reference
          ---------
          Morcos et al. "Insights on representational similarity in
          neural networks with canonical correlation" (NeurIPS 2018)
          )doc");
}

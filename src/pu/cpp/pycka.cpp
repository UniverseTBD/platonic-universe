#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cka.h"

namespace py = pybind11;

PYBIND11_MODULE(pu_cka, m) {
    m.doc() = "CKA memory-mapped binding";
    m.def("compute_cka",
          &compute_cka_from_files,
          py::arg("file1"),
          py::arg("file2"),
          py::arg("n_rows"),
          py::arg("n_cols"),
          "Compute CKA from two memory-mapped kernel files");
}
#ifndef PU_CKA_H
#define PU_CKA_H

#include <string>
#include <cstddef>

double compute_cka_from_files(const std::string& file1,
                              const std::string& file2,
                              size_t n_rows,
                              size_t n_cols);

#endif // PU_CKA_H
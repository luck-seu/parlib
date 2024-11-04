#ifndef CLIENT_MATRIX_GENERATOR_H
#define CLIENT_MATRIX_GENERATOR_H

#include <stdint.h>

#include <iostream>
#include <random>

namespace seu::luck::hybridcomp::client {

class MatrixGenerator {
 public:
  MatrixGenerator(uint64_t row, int col) : row_(row), col_(col) {
    matrix_ = new uint64_t*[row_];
    for (uint64_t i = 0; i < row_; i++) {
      matrix_[i] = new uint64_t[col_];
    }

    matrix_transpose_ = nullptr;
  }

  ~MatrixGenerator() {
    for (uint64_t i = 0; i < row_; i++) {
      delete[] matrix_[i];
    }
    delete[] matrix_;

    if (matrix_transpose_ != nullptr) {
      for (uint64_t i = 0; i < col_; i++) {
        delete[] matrix_transpose_[i];
      }
      delete[] matrix_transpose_;
    }
  }

  void GenerateRandomMatrix(uint64_t lb, uint64_t ub, uint64_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> dis(lb, ub);
    for (uint64_t i = 0; i < row_; i++) {
      for (uint64_t j = 0; j < col_; j++) {
        matrix_[i][j] = dis(gen);
      }
    }
  }

  void ShowMatrix() {
    for (uint64_t i = 0; i < row_; i++) {
      for (uint64_t j = 0; j < col_; j++) {
        std::cout << matrix_[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  uint64_t** GetMatrix() { return matrix_; }

  uint64_t** GetMatrixTranspose() {
    // Lazy initialization.
    if (matrix_transpose_ == nullptr) {
      matrix_transpose_ = new uint64_t*[col_];
      for (uint64_t i = 0; i < col_; i++) {
        matrix_transpose_[i] = new uint64_t[row_];
      }
      for (uint64_t i = 0; i < row_; i++) {
        for (uint64_t j = 0; j < col_; j++) {
          matrix_transpose_[j][i] = matrix_[i][j];
        }
      }
    }
    return matrix_transpose_;
  }

 private:
  uint64_t row_;
  uint64_t col_;
  uint64_t** matrix_;

  uint64_t** matrix_transpose_;
};
}  // namespace seu::luck::hybridcomp::client

#endif  // CLIENT_MATRIX_GENERATOR_H

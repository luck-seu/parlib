#ifndef CLIENT_FLAT_MATRIX_GENERATOR_H
#define CLIENT_FLAT_MATRIX_GENERATOR_H

#include <stdint.h>

#include <iostream>
#include <random>

namespace luck::hybridcomp::client {

class FlatMatrixGenerator {
 public:
  FlatMatrixGenerator(uint64_t row, int col) : row_(row), col_(col) {
    matrix_ = new uint32_t[row_ * col_];
    matrix_transpose_ = nullptr;
  }

  ~FlatMatrixGenerator() {
    delete[] matrix_;
    if (matrix_transpose_ != nullptr) delete[] matrix_transpose_;
  }

  void GenerateRandomMatrix(uint64_t lb, uint64_t ub, uint64_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> dis(lb, ub);
    for (uint64_t i = 0; i < row_; i++) {
      for (uint64_t j = 0; j < col_; j++) {
        matrix_[i * row_ + j] = dis(gen);
      }
    }
  }

  void ShowMatrix() {
    for (uint64_t i = 0; i < row_; i++) {
      for (uint64_t j = 0; j < col_; j++) {
        std::cout << matrix_[i * row_ + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  uint32_t* GetMatrix() { return matrix_; }

  uint32_t* GetMatrixTranspose() {
    if (matrix_transpose_ == nullptr) {
      matrix_transpose_ = new uint32_t[row_ * col_];
      for (uint64_t i = 0; i < row_; i++) {
        for (uint64_t j = 0; j < col_; j++) {
          matrix_transpose_[j * col_ + i] = matrix_[i * row_ + j];
        }
      }
    }
    return matrix_transpose_;
  }

 private:
  uint32_t row_;
  uint32_t col_;
  uint32_t* matrix_;

  uint32_t* matrix_transpose_;
};
}  // namespace seu::luck::hybridcomp::client

#endif  // CLIENT_FLAT_MATRIX_GENERATOR_H

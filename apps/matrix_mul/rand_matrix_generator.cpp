#include "apps/matrix_mul/rand_matrix_generator.h"

namespace luck::parlib::apps::matrix_mul {

RandMatrixGenerator::RandMatrixGenerator(uint32_t row, uint32_t col)
    : row_(row), col_(col) {
  matrix_ = new uint32_t[row_ * col_];
  matrix_transpose_ = new uint32_t[col_ * row_];
}

RandMatrixGenerator::~RandMatrixGenerator() {
  delete[] matrix_;
  if (matrix_transpose_ != nullptr) delete[] matrix_transpose_;
}

void RandMatrixGenerator::GenerateRandomMatrix(uint32_t lb, uint32_t ub,
                                               uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(lb, ub);
  for (uint32_t i = 0; i < row_; i++) {
    for (uint32_t j = 0; j < col_; j++) {
      matrix_[i * col_ + j] = dis(gen);
    }
  }

  for (uint32_t i = 0; i < row_; i++) {
    for (uint32_t j = 0; j < col_; j++) {
      matrix_transpose_[j * row_ + i] = matrix_[i * col_ + j];
    }
  }
}

void RandMatrixGenerator::ShowMatrix() const {
  for (uint32_t i = 0; i < row_; i++) {
    for (uint32_t j = 0; j < col_; j++) {
      std::cout << matrix_[i * col_ + j] << " ";
    }
    std::cout << std::endl;
  }
}

void RandMatrixGenerator::ShowMatrixTranspose() const {
  for (uint32_t i = 0; i < col_; i++) {
    for (uint32_t j = 0; j < row_; j++) {
      std::cout << matrix_transpose_[i * row_ + j] << " ";
    }
    std::cout << std::endl;
  }
}
}  // namespace luck::parlib::apps::matrix_mul
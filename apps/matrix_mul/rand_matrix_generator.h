#ifndef APPS_FLAT_MATRIX_GENERATOR_H
#define APPS_FLAT_MATRIX_GENERATOR_H

#include <stdint.h>

#include <iostream>
#include <random>

namespace luck::parlib::apps::matrix_mul {

class RandMatrixGenerator {
 public:
  RandMatrixGenerator(uint32_t row, uint32_t col);

  ~RandMatrixGenerator();

  void GenerateRandomMatrix(uint32_t lb, uint32_t ub, uint32_t seed);

  void ShowMatrix() const;

  void ShowMatrixTranspose() const;

  uint32_t* GetMatrix() const { return matrix_; }

  uint32_t* GetMatrixTranspose() const { return matrix_transpose_; }

 private:
  uint32_t row_;
  uint32_t col_;
  uint32_t* matrix_;

  uint32_t* matrix_transpose_;
};
}  // namespace luck::parlib::apps::matrix_mul

#endif  // APPS_FLAT_MATRIX_GENERATOR_H

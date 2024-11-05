#ifndef APPS_MATRIX_MULTIPLICATION_CPU_H
#define APPS_MATRIX_MULTIPLICATION_CPU_H

#include <stdint.h>

#include <chrono>
#include <iostream>

#include "cpu_par/thread_pool.h"

namespace luck::parlib::apps::matrix_mul {

class MatrixMultiplicationCPU {
 private:
  using Task = luck::parlib::cpu::Task;
  using TaskPackage = luck::parlib::cpu::TaskPackage;
  using ThreadPool = luck::parlib::cpu::ThreadPool;

 public:
  MatrixMultiplicationCPU(uint32_t parallelism, uint32_t n_workers)
      : thread_pool_(parallelism), n_workers_(n_workers), result_(nullptr) {}

  ~MatrixMultiplicationCPU() { DestoryResult(); }

  uint32_t* Mul(uint32_t* lhs, uint32_t* rhs, uint32_t lhs_n_rows,
                uint32_t lhs_n_cols, uint32_t rhs_n_rows, uint32_t rhs_n_cols);

  void ShowResult() const;

  void ResetResult(uint32_t n_rows, uint32_t n_cols);

  void DestoryResult();

 private:
  uint32_t* result_;
  uint32_t result_n_rows_;
  uint32_t result_n_cols_;

  ThreadPool thread_pool_;
  uint32_t n_workers_;
};
}  // namespace luck::parlib::apps::matrix_mul

#endif  // APPS_MATRIX_MULTIPLICATION_CPU_H
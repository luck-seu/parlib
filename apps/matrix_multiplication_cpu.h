#ifndef CLIENT_MATRIX_MULTIPLICATION_CPU_H
#define CLIENT_MATRIX_MULTIPLICATION_CPU_H

#include <stdint.h>

#include <chrono>
#include <iostream>

#include "cpu_par/thread_pool.h"

namespace seu::luck::hybridcomp::client {

class MatrixMultiplicationCPU {
 private:
  using Task = com::graph::core::common::Task;
  using TaskPackage = com::graph::core::common::TaskPackage;
  using ThreadPool = com::graph::core::common::ThreadPool;

 public:
  MatrixMultiplicationCPU(uint64_t parallelism, uint64_t n_workers)
      : thread_pool_(parallelism), n_workers_(n_workers) {}

  ~MatrixMultiplicationCPU() { DistoryResult(); }

  uint64_t** Mul(uint64_t** lhs, uint64_t** rhs, uint64_t lhs_n_rows,
                 uint64_t lhs_n_cols) {
    std::cout << "[MM-CPU] Start " << "(" << lhs_n_rows << ", " << lhs_n_cols
              << ")" << " X " << "(" << lhs_n_cols << ", " << lhs_n_rows
              << ")"
                 " matrix multiplication."
              << std::endl;
    ResetResult(lhs_n_rows, lhs_n_cols);
    std::cout << "[MM-CPU] Initialized the result matrix." << std::endl;

    TaskPackage task_package;
    task_package.reserve(n_workers_);
    uint64_t task_size = lhs_n_rows * lhs_n_rows / n_workers_ + 1;

    for (uint64_t i = 0; i < n_workers_; i++) {
      Task task = [this, lhs, rhs, lhs_n_rows, lhs_n_cols, i, task_size]() {
        for (uint64_t t = task_size * i; t < task_size * (i + 1); t++) {
          if (t >= lhs_n_rows * lhs_n_rows) break;
          uint64_t row_idx = t / lhs_n_rows;
          uint64_t col_idx = t % lhs_n_rows;

          for (uint64_t k = 0; k < lhs_n_cols; k++) {
            result_[row_idx][col_idx] += lhs[row_idx][k] * rhs[k][col_idx];
          }
        }
      };
      task_package.push_back(task);
    }
    std::cout << "[MM-CPU] Submit the task set to the thread pool."
              << std::endl;
    auto start = std::chrono::system_clock::now();
    thread_pool_.SubmitSync(task_package);
    auto end = std::chrono::system_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "[MM-CPU] Finished the matrix multiplication in " << duration
              << " sec." << std::endl;

    return result_;
  }

  void ShowResult() {
    for (uint64_t i = 0; i < result_n_rows_; i++) {
      for (uint64_t j = 0; j < result_n_cols_; j++) {
        std::cout << result_[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  void ResetResult(uint64_t lhs_n_rows, uint64_t lhs_n_cols) {
    result_n_rows_ = lhs_n_rows;
    result_n_cols_ = lhs_n_cols;
    result_ = new uint64_t*[result_n_rows_];
    for (uint64_t i = 0; i < result_n_rows_; i++) {
      result_[i] = new uint64_t[result_n_cols_];
      for (uint64_t j = 0; j < result_n_cols_; j++) {
        result_[i][j] = 0;
      }
    }
  }

  void DistoryResult() {
    for (uint64_t i = 0; i < result_n_rows_; i++) {
      delete[] result_[i];
    }
    delete[] result_;
  }

 private:
  uint64_t** result_;
  uint64_t result_n_rows_;
  uint64_t result_n_cols_;

  ThreadPool thread_pool_;
  uint64_t n_workers_;
};
}  // namespace seu::luck::hybridcomp::client

#endif  // CLIENT_MATRIX_MULTIPLICATION_CPU_H
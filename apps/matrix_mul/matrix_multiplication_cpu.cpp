#include "apps/matrix_mul/matrix_multiplication_cpu.h"

namespace luck::parlib::apps::matrix_mul {

uint32_t* MatrixMultiplicationCPU::Mul(uint32_t* lhs, uint32_t* rhs,
                                       uint32_t lhs_n_rows, uint32_t lhs_n_cols,
                                       uint32_t rhs_n_rows,
                                       uint32_t rhs_n_cols) {
  std::cout << "[MM-CPU] Start " << "(" << lhs_n_rows << ", " << lhs_n_cols
            << ")" << " X " << "(" << rhs_n_rows << ", " << rhs_n_cols
            << ")"
               " matrix multiplication."
            << std::endl;
  ResetResult(lhs_n_rows, rhs_n_cols);
  std::cout << "[MM-CPU] Initialized the result matrix." << std::endl;

  TaskPackage task_package;
  task_package.reserve(n_workers_);
  uint32_t task_size = lhs_n_rows * rhs_n_cols / n_workers_ + 1;

  for (uint32_t i = 0; i < n_workers_; i++) {
    Task task = [this, lhs, rhs, lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols,
                 i, task_size]() {
      for (uint32_t t = task_size * i; t < task_size * (i + 1); t++) {
        if (t >= lhs_n_rows * rhs_n_cols) break;
        uint32_t row_idx = t / rhs_n_cols;
        uint32_t col_idx = t % rhs_n_cols;

        for (uint32_t k = 0; k < lhs_n_cols; k++) {
          result_[row_idx * rhs_n_cols + col_idx] +=
              lhs[row_idx * lhs_n_cols + k] * rhs[k * rhs_n_cols + col_idx];
        }
      }
    };
    task_package.push_back(task);
  }
  std::cout << "[MM-CPU] Submit the task set to the thread pool." << std::endl;
  auto start = std::chrono::system_clock::now();
  thread_pool_.SubmitSync(task_package);
  auto end = std::chrono::system_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();
  std::cout << "[MM-CPU] Finished the matrix multiplication in " << duration
            << " sec." << std::endl;

  return result_;
}

void MatrixMultiplicationCPU::ShowResult() const {
  for (uint32_t i = 0; i < result_n_rows_; i++) {
    for (uint32_t j = 0; j < result_n_cols_; j++) {
      std::cout << result_[i * result_n_cols_ + j] << " ";
    }
    std::cout << std::endl;
  }
}

void MatrixMultiplicationCPU::ResetResult(uint32_t n_rows, uint32_t n_cols) {
  result_n_rows_ = n_rows;
  result_n_cols_ = n_cols;
  result_ = new uint32_t[result_n_rows_ * result_n_cols_];
  for (uint32_t i = 0; i < result_n_rows_; i++) {
    for (uint32_t j = 0; j < result_n_cols_; j++) {
      result_[i * result_n_cols_ + j] = 0;
    }
  }
}

void MatrixMultiplicationCPU::DestoryResult() {
  if (result_ != nullptr) delete[] result_;
}
}  // namespace luck::parlib::apps::matrix_mul
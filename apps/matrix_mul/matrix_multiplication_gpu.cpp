#include "apps/matrix_mul/matrix_multiplication_gpu.h"

namespace luck::parlib::apps::matrix_mul {

void MatrixMultiplicationGPU::Mul(uint32_t* lhs, uint32_t* rhs,
                                  uint32_t lhs_n_rows, uint32_t lhs_n_cols,
                                  uint32_t rhs_n_rows, uint32_t rhs_n_cols) {
  std::cout << "[MM-GPU] Start " << "(" << lhs_n_rows << ", " << lhs_n_cols
            << ")" << " X " << "(" << rhs_n_rows << ", " << rhs_n_cols
            << ")"
               " matrix multiplication."
            << std::endl;
  ResetResult(lhs_n_rows, rhs_n_cols);
  std::cout << "[MM-GPU] Initialized the result matrix." << std::endl;

  HostMatrixData lhs_mat(lhs, lhs_n_rows, lhs_n_cols);
  HostMatrixData rhs_mat(rhs, rhs_n_rows, rhs_n_cols);

  MatrixMulHostTaskData host_input(lhs_mat, rhs_mat);
  HostMatrixData host_output(result_, lhs_n_rows, rhs_n_cols);

  GPUTaskManager gpu_task_manager;
  auto start = std::chrono::system_clock::now();
  gpu_task_manager.SubmitTaskSync(0, DeviceTaskType::kMatrixMultiplication,
                                  MatrixMultiplicationWrap::Do, &host_input,
                                  nullptr, &host_output, 0);
  auto end = std::chrono::system_clock::now();
  double duration = std::chrono::duration<double>(end - start).count();
  std::cout << "[MM-GPU] Finished the matrix multiplication in " << duration
            << " sec." << std::endl;
}

void MatrixMultiplicationGPU::ShowResult() const {
  for (uint32_t i = 0; i < result_n_rows_; i++) {
    for (uint32_t j = 0; j < result_n_cols_; j++) {
      std::cout << result_[i * result_n_rows_ + j] << " ";
    }
    std::cout << std::endl;
  }
}

void MatrixMultiplicationGPU::ResetResult(uint32_t lhs_n_rows,
                                          uint64_t lhs_n_cols) {
  result_n_rows_ = lhs_n_rows;
  result_n_cols_ = lhs_n_cols;
  result_ = new uint32_t[result_n_rows_ * result_n_cols_];
  for (uint32_t i = 0; i < result_n_rows_; i++) {
    for (uint32_t j = 0; j < result_n_cols_; j++) {
      result_[i * result_n_rows_ + j] = 0;
    }
  }
}

void MatrixMultiplicationGPU::DestoryResult() {
  if (result_ != nullptr) delete[] result_;
}
}  // namespace luck::parlib::apps::matrix_mul
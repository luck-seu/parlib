#ifndef CLIENT_MATRIX_MULTIPLICATION_CPU_H
#define CLIENT_MATRIX_MULTIPLICATION_CPU_H

#include <stdint.h>

#include <chrono>
#include <iostream>

#include "gpu/app/matrix_multiplication_wrap.cuh"
#include "gpu/gpu_task_manager.cuh"
#include "gpu/data_collections/host_data_collections/matrix_mul_host_task_data.cuh"
#include "multithreading/thread_pool.h"

namespace luck::hybridcomp::client {

class MatrixMultiplicationGPU {
 private:
  using GPUTaskManager = luck::gpu::GPUTaskManager;
  using MatrixMulHostTaskData = luck::gpu::data::host::MatrixMulHostTaskData;
  using HostMatrixData = luck::gpu::data::host::HostMatrixData;

 public:
  MatrixMultiplicationGPU() = default;

  ~MatrixMultiplicationGPU() { DistoryResult(); }

  void Mul(uint32_t* lhs, uint32_t* rhs, uint32_t lhs_n_rows, uint32_t lhs_n_cols) {
    std::cout << "[MM-GPU] Start " << "(" << lhs_n_rows << ", " << lhs_n_cols
              << ")" << " X " << "(" << lhs_n_cols << ", " << lhs_n_rows
              << ")"
                 " matrix multiplication."
              << std::endl;
    ResetResult(lhs_n_rows, lhs_n_cols);
    std::cout << "[MM-GPU] Initialized the result matrix." << std::endl;

    HostMatrixData lhs_mat(lhs, lhs_n_rows, lhs_n_cols);
    HostMatrixData rhs_mat(rhs, lhs_n_cols, lhs_n_rows);

    MatrixMulHostTaskData host_input(lhs_mat, rhs_mat);
    HostMatrixData host_output(result_, lhs_n_cols, lhs_n_rows);

    GPUTaskManager gpu_task_manager;
    auto start = std::chrono::system_clock::now();
    gpu_task_manager.SubmitTaskSync(0, luck::gpu::DeviceTaskType::kMatrixMultiplication,
                                    luck::gpu::app::MatrixMultiplicationWrap::Do,
                                    &host_input, nullptr, &host_output, 0);
                                        auto end = std::chrono::system_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "[MM-GPU] Finished the matrix multiplication in " << duration
              << " sec." << std::endl;
  }

  void ShowResult() {
    for (uint32_t i = 0; i < result_n_rows_; i++) {
      for (uint32_t j = 0; j < result_n_cols_; j++) {
        std::cout << result_[i * result_n_rows_ + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  void ResetResult(uint32_t lhs_n_rows, uint64_t lhs_n_cols) {
    result_n_rows_ = lhs_n_rows;
    result_n_cols_ = lhs_n_cols;
    result_ = new uint32_t[result_n_rows_ * result_n_cols_];
    for (uint32_t i = 0; i < result_n_rows_; i++) {
      for (uint32_t j = 0; j < result_n_cols_; j++) {
        result_[i * result_n_rows_ + j] = 0;
      }
    }
  }

  void DistoryResult() {
    delete[] result_;
  }

 private:
  uint32_t* result_;
  uint32_t result_n_rows_;
  uint32_t result_n_cols_;
};
}  // namespace seu::luck::hybridcomp::client

#endif  // CLIENT_MATRIX_MULTIPLICATION_CPU_H
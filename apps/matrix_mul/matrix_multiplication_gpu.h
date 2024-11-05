#ifndef APPS_MATRIX_MULTIPLICATION_GPU_H
#define APPS_MATRIX_MULTIPLICATION_GPU_H

#include <stdint.h>

#include <chrono>
#include <iostream>

#include "cpu_par/thread_pool.h"
#include "gpu_par/data_collections/host_data_collections/matrix_mul_host_task_data.cuh"
#include "gpu_par/kernel/matrix_mul/matrix_multiplication_wrap.cuh"
#include "gpu_par/util/gpu_task_manager.cuh"

namespace luck::parlib::apps::matrix_mul {

class MatrixMultiplicationGPU {
 private:
  using GPUTaskManager = luck::parlib::gpu::util::GPUTaskManager;
  using MatrixMulHostTaskData =
      luck::parlib::gpu::data::host::MatrixMulHostTaskData;
  using HostMatrixData = luck::parlib::gpu::data::host::HostMatrixData;
  using DeviceTaskType = luck::parlib::gpu::util::DeviceTaskType;
  using MatrixMultiplicationWrap =
      luck::parlib::gpu::kernel::matrix_mul::MatrixMultiplicationWrap;

 public:
  MatrixMultiplicationGPU() = default;

  ~MatrixMultiplicationGPU() { DestoryResult(); }

  void Mul(uint32_t* lhs, uint32_t* rhs, uint32_t lhs_n_rows,
           uint32_t lhs_n_cols, uint32_t rhs_n_rows, uint32_t rhs_n_cols);

  void ShowResult() const;

  void ResetResult(uint32_t lhs_n_rows, uint64_t lhs_n_cols);

  void DestoryResult();

 private:
  uint32_t* result_;
  uint32_t result_n_rows_;
  uint32_t result_n_cols_;
};
}  // namespace luck::parlib::apps::matrix_mul

#endif  // APPS_MATRIX_MULTIPLICATION_GPU_H
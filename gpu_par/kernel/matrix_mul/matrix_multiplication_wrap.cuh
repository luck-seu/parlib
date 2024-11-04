#ifndef GPU_PAR_KERNEL_MATRIX_MUL_MATRIX_MULTIPLICATION_WRAP_CUH
#define GPU_PAR_KERNEL_MATRIX_MUL_MATRIX_MULTIPLICATION_WRAP_CUH

#include <stdint.h>

#include "gpu_par/data_collections/device_data_collections/matrix_mul_device_task_data.cuh"

namespace luck::parlib::gpu::kernel::matrix_mul {

class MatrixMultiplicationWrap {
 private:
  using DeviceTaskData = luck::parlib::gpu::data::device::DeviceTaskData;

 public:
  MatrixMultiplicationWrap(const MatrixMultiplicationWrap& wrap) = delete;

  void operator=(const MatrixMultiplicationWrap& wrap) = delete;

  static MatrixMultiplicationWrap* GetInstance();

  static void Do(const cudaStream_t& stream, DeviceTaskData* device_input,
                 DeviceTaskData* device_exec_plan,
                 DeviceTaskData* device_output);

 private:
  MatrixMultiplicationWrap() = default;

  inline static MatrixMultiplicationWrap* ptr_ = nullptr;
};
}  // namespace luck::parlib::gpu::kernel::matrix_mul

#endif  // GPU_PAR_KERNEL_MATRIX_MUL_MATRIX_MULTIPLICATION_WRAP_CUH
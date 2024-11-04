#ifndef LUCK_GPU_APP_MATRIX_MULTIPLICATION_WRAP_CUH
#define LUCK_GPU_APP_MATRIX_MULTIPLICATION_WRAP_CUH

#include <stdint.h>

#include "gpu/data_collections/device_data_collections/matrix_mul_device_task_data.cuh"

namespace luck {
namespace gpu {
namespace app {

class MatrixMultiplicationWrap {
 public:
  MatrixMultiplicationWrap(const MatrixMultiplicationWrap& wrap) = delete;

  void operator=(const MatrixMultiplicationWrap& wrap) = delete;

  static MatrixMultiplicationWrap* GetInstance();

  static void Do(
      const cudaStream_t& stream,
      luck::gpu::data::device::DeviceTaskData* device_input,
      luck::gpu::data::device::DeviceTaskData* device_exec_plan,
      luck::gpu::data::device::DeviceTaskData* device_output);

 private:
  MatrixMultiplicationWrap() = default;

  inline static MatrixMultiplicationWrap* ptr_ = nullptr;
};
}  // namespace app
}  // namespace gpu
}  // namespace luck

#endif  // LUCK_GPU_APP_MATRIX_MULTIPLICATION_WRAP_CUH
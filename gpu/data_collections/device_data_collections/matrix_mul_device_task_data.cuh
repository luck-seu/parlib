#ifndef LUCK_GPU_DATA_COLLECTIONS_DEVICE_DATA_COLLECTIONS_MATRIX_MUL_DEVICE_TASK_DATA_CUH
#define LUCK_GPU_DATA_COLLECTIONS_DEVICE_DATA_COLLECTIONS_MATRIX_MUL_DEVICE_TASK_DATA_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#include "gpu/data_collections/device_buffer.cuh"
#include "gpu/data_collections/device_task_data.cuh"
#include "gpu/data_collections/host_data_collections/matrix_mul_host_task_data.cuh"
#include "gpu/util/cuda_check.cuh"

namespace luck {
namespace gpu {
namespace data {
namespace device {

class MatrixMulInputDeviceTaskData : public DeviceTaskData {
 public:
  MatrixMulInputDeviceTaskData() = default;
  MatrixMulInputDeviceTaskData(uint32_t task_id) : DeviceTaskData(task_id) {}

  void SetData(const host::MatrixMulHostTaskData& host_data,
               const cudaStream_t& stream);

  DeviceOwnedBuffer<uint32_t> lhs_matrix_;
  DeviceOwnedBuffer<uint32_t> rhs_matrix_;

  uint32_t n_lhs_rows_;
  uint32_t n_lhs_cols_;
};

class MatrixMulOutputDeviceTaskData : public DeviceTaskData {
 public:
  MatrixMulOutputDeviceTaskData() = default;
  MatrixMulOutputDeviceTaskData(uint64_t task_id) : DeviceTaskData(task_id) {}

  void SetData(const host::HostMatrixData& host_data,
               const cudaStream_t& stream);

 DeviceOwnedBuffer<uint32_t> result_matrix;
 uint32_t n_rows_;
 uint32_t n_cols_;
};
}  // namespace device
}  // namespace data
}  // namespace gpu
}  // namespace luck

#endif  // LUCK_GPU_DATA_COLLECTIONS_DEVICE_DATA_COLLECTIONS_MATRIX_MUL_DEVICE_TASK_DATA_CUH
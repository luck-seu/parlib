#ifndef GPU_DATA_STRUCTURES_GPU_TASK_DATA_CUH
#define GPU_DATA_STRUCTURES_GPU_TASK_DATA_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace luck {
namespace gpu {
namespace data {
namespace device {

// @Description: Class to store common task data for GPU execution.
// A GPU task might have multiple DeviceOwnedBuffer class. It is output of
// PrepareBufferForKernel function.
class DeviceTaskData {
 public:
  DeviceTaskData() = default;
  DeviceTaskData(int task_id) : task_id_(task_id) {};

  virtual ~DeviceTaskData() = default;

  uint64_t get_task_id() const { return task_id_; }

 protected:
  uint64_t task_id_;
};
}  // namespace device
}  // namespace data
}  // namespace gpu
}  // namespace luck

#endif  // GPU_DATA_STRUCTURES_GPU_TASK_DATA_CUH
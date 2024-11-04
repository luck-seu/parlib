#ifndef GPU_GPU_TASK_MANAGER_CUH
#define GPU_GPU_TASK_MANAGER_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#include <mutex>
#include <unordered_map>

#include "gpu_par/data_collections/device_data_collections/matrix_mul_device_task_data.cuh"
#include "gpu_par/data_collections/device_task_data.cuh"
#include "gpu_par/data_collections/host_data_collections/matrix_mul_host_task_data.cuh"
#include "gpu_par/data_collections/host_task_data.cuh"
#include "gpu_par/kernel_func/function_ptrs.cuh"

namespace luck {
namespace gpu {

enum DeviceTaskType {
  kMatrixMultiplication = 1,
};

class GPUTaskManager {
 private:
  using DeviceTaskData = luck::gpu::data::device::DeviceTaskData;
  using HostTaskData = luck::gpu::data::host::HostTaskData;
  using KernelWrap = luck::gpu::kernel_func::KernelWrap;

 public:
  GPUTaskManager() = default;

  ~GPUTaskManager();

  void SubmitTaskSync(uint64_t task_id, DeviceTaskType task_type,
                      KernelWrap kernel_wrap, HostTaskData* host_input,
                      HostTaskData* host_args, HostTaskData* host_output,
                      int bin_id);

  void SubmitTaskASync(uint64_t task_id, DeviceTaskType task_type,
                       KernelWrap kernel_wrap, HostTaskData* host_input,
                       HostTaskData* host_args, HostTaskData* host_output,
                       int bin_id);

  bool IsTaskFinished(size_t task_id);

  DeviceTaskData* GetInputDeviceTaskDataPtr(size_t task_id);

  DeviceTaskData* GetOutputDeviceTaskDataPtr(size_t task_id);

  DeviceTaskData* CreateInputDeviceTaskData(size_t task_id,
                                            DeviceTaskType task_type,
                                            const cudaStream_t& stream,
                                            HostTaskData* host_input);

  DeviceTaskData* CreateOutputDeviceTaskData(size_t task_id,
                                             DeviceTaskType task_type,
                                             const cudaStream_t& stream,
                                             HostTaskData* host_output);

  DeviceTaskData* CreateArgsDeviceTaskData(size_t task_id,
                                           DeviceTaskType task_type,
                                           const cudaStream_t& stream,
                                           HostTaskData* host_args);

  void EvictDeviceData(size_t task_id);

  void ReleaseStream(size_t task_id);

  void RemoveDeviceInput(size_t task_id);

  void RemoveDeviceOutput(size_t task_id);

  void RemoveDeviceArgs(size_t task_id);

  void DeviceSynchronize();

 private:
  cudaStream_t* GetStream(size_t task_id);

  void InsertInputDeviceTaskData(size_t task_id, DeviceTaskData* input);

  void InsertArgsDeviceTaskData(size_t task_id, DeviceTaskData* input);

  void InsertOutputDeviceTaskData(size_t task_id, DeviceTaskData* output);

  void TransferResultFromDevice2Host(DeviceTaskType task_type,
                                     const cudaStream_t& stream,
                                     DeviceTaskData* device_output,
                                     HostTaskData* host_output);

  std::mutex streams_mtx_;
  std::mutex input_data_mtx_;
  std::mutex output_data_mtx_;
  std::mutex args_mtx_;

  std::unordered_map<size_t, cudaStream_t*> streams_by_task_id_;
  std::unordered_map<size_t, DeviceTaskData*> input_ptr_by_task_id_;
  std::unordered_map<size_t, DeviceTaskData*> args_ptr_by_task_id_;
  std::unordered_map<size_t, DeviceTaskData*> output_ptr_by_task_id_;
};

}  // namespace gpu
}  // namespace luck

#endif  // GPU_GPU_TASK_MANAGER_CUH
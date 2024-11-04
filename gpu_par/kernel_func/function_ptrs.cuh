#ifndef GPU_FUNCTION_PTRS_CUH
#define GPU_FUNCTION_PTRS_CUH

#include <cuda_runtime.h>

#include "gpu_par/data_collections/device_buffer.cuh"
#include "gpu_par/data_collections/device_task_data.cuh"
#include "gpu_par/data_collections/host_buffer.cuh"

namespace luck {
namespace gpu {
namespace kernel_func {

using DeviceTaskData = luck::gpu::data::device::DeviceTaskData;

// Typedef for function pointer to wrap kernel function
// @Parameters:
//    device_input: Pointer to the input DeviceOwnedBuffer on device.
//    device_output: Pointer to the output DeviceOwnedBuffer on device
//    stream: Reference to the CUDA stream for asynchronous execution
typedef void (*KernelWrap)(const cudaStream_t& stream,
                           DeviceTaskData* device_input, DeviceTaskData* device_args,
                           DeviceTaskData* device_output);
}  // namespace kernel_func
}  // namespace gpu
}  // namespace luck

#endif  // GPU_FUNCTION_PTRS_CUH
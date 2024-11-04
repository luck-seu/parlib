#include "gpu/gpu_task_manager.cuh"

namespace luck {
namespace gpu {

using KernelWrap = luck::gpu::kernel_func::KernelWrap;
using HostTaskData = luck::gpu::data::host::HostTaskData;
using DeviceTaskData = luck::gpu::data::device::DeviceTaskData;
using HostMatrixData = luck::gpu::data::host::HostMatrixData;
using MatrixMulHostTaskData = luck::gpu::data::host::MatrixMulHostTaskData;
using MatrixMulInputDeviceTaskData =
    luck::gpu::data::device::MatrixMulInputDeviceTaskData;
using MatrixMulOutputDeviceTaskData =
    luck::gpu::data::device::MatrixMulOutputDeviceTaskData;

GPUTaskManager::~GPUTaskManager() {
  for (auto iter = streams_by_task_id_.begin();
       iter != streams_by_task_id_.end();) {
    cudaStreamDestroy(*(iter->second));
    iter = streams_by_task_id_.erase(iter);
  }
  for (auto iter = input_ptr_by_task_id_.begin();
       iter != input_ptr_by_task_id_.end();) {
    delete iter->second;
    iter = input_ptr_by_task_id_.erase(iter);
  }
  for (auto iter = output_ptr_by_task_id_.begin();
       iter != output_ptr_by_task_id_.end();) {
    delete iter->second;
    iter = output_ptr_by_task_id_.erase(iter);
  }
  for (auto iter = args_ptr_by_task_id_.begin();
       iter != args_ptr_by_task_id_.end();) {
    delete iter->second;
    iter = args_ptr_by_task_id_.erase(iter);
  }
}

void GPUTaskManager::SubmitTaskSync(uint64_t task_id, DeviceTaskType task_type,
                                    KernelWrap kernel_wrap,
                                    HostTaskData* host_input,
                                    HostTaskData* host_args,
                                    HostTaskData* host_output, int bin_id) {
  cudaSetDevice(bin_id);

  cudaStream_t* p_stream = GetStream(task_id);

  auto device_input =
      CreateInputDeviceTaskData(task_id, task_type, *p_stream, host_input);
  auto device_output =
      CreateOutputDeviceTaskData(task_id, task_type, *p_stream, host_output);
  auto device_args =
      CreateArgsDeviceTaskData(task_id, task_type, *p_stream, host_args);

  InsertInputDeviceTaskData(task_id, device_input);
  InsertOutputDeviceTaskData(task_id, device_output);
  InsertArgsDeviceTaskData(task_id, device_args);

  kernel_wrap(*p_stream, device_input, device_args, device_output);
  TransferResultFromDevice2Host(task_type, *p_stream, device_output,
                                host_output);

  cudaStreamSynchronize(*p_stream);
}

void GPUTaskManager::SubmitTaskASync(uint64_t task_id, DeviceTaskType task_type,
                                     KernelWrap kernel_wrap,
                                     HostTaskData* host_input,
                                     HostTaskData* host_args,
                                     HostTaskData* host_output, int bin_id) {
  cudaSetDevice(bin_id);

  cudaStream_t* p_stream = GetStream(task_id);

  auto device_input =
      CreateInputDeviceTaskData(task_id, task_type, *p_stream, host_input);
  auto device_output =
      CreateOutputDeviceTaskData(task_id, task_type, *p_stream, host_output);
  auto device_args =
      CreateArgsDeviceTaskData(task_id, task_type, *p_stream, host_args);

  InsertInputDeviceTaskData(task_id, device_input);
  InsertOutputDeviceTaskData(task_id, device_output);
  InsertArgsDeviceTaskData(task_id, device_args);

  kernel_wrap(*p_stream, device_input, device_args, device_output);
  TransferResultFromDevice2Host(task_type, *p_stream, device_output,
                                host_output);
}

bool GPUTaskManager::IsTaskFinished(size_t task_id) {
  auto iter = streams_by_task_id_.find(task_id);
  if (iter == streams_by_task_id_.end()) {
    return false;
  } else {
    cudaError_t err = cudaStreamQuery(*iter->second);
    if (err == cudaSuccess)
      return true;
    else if (err == cudaErrorNotReady)
      return false;
  }
  return true;
}

DeviceTaskData* GPUTaskManager::GetInputDeviceTaskDataPtr(size_t task_id) {
  std::lock_guard<std::mutex> lock(input_data_mtx_);

  auto iter = input_ptr_by_task_id_.find(task_id);
  if (iter != input_ptr_by_task_id_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

DeviceTaskData* GPUTaskManager::GetOutputDeviceTaskDataPtr(size_t task_id) {
  std::lock_guard<std::mutex> lock(output_data_mtx_);

  auto iter = output_ptr_by_task_id_.find(task_id);
  if (iter != output_ptr_by_task_id_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

DeviceTaskData* GPUTaskManager::CreateInputDeviceTaskData(
    size_t task_id, DeviceTaskType task_type, const cudaStream_t& stream,
    HostTaskData* host_input) {
  DeviceTaskData* p_task_data = nullptr;
  switch (task_type) {
    case kMatrixMultiplication: {
      auto* host_input_data =
          reinterpret_cast<MatrixMulHostTaskData*>(host_input);
      auto* device_input_data = new MatrixMulInputDeviceTaskData(task_id);
      device_input_data->SetData(*host_input_data, stream);
      p_task_data = reinterpret_cast<DeviceTaskData*>(device_input_data);
      break;
    }
    default:
      break;
  }
  return p_task_data;
}

DeviceTaskData* GPUTaskManager::CreateOutputDeviceTaskData(
    size_t task_id, DeviceTaskType task_type, const cudaStream_t& stream,
    HostTaskData* host_output) {
  DeviceTaskData* p_task_data = nullptr;
  switch (task_type) {
    case kMatrixMultiplication: {
      auto* host_output_data = reinterpret_cast<HostMatrixData*>(host_output);
      auto* new_output_task_data = new MatrixMulOutputDeviceTaskData(task_id);
      new_output_task_data->SetData(*host_output_data, stream);
      p_task_data = reinterpret_cast<DeviceTaskData*>(new_output_task_data);
      break;
    }
    default:
      break;
  }
  return p_task_data;
}

DeviceTaskData* GPUTaskManager::CreateArgsDeviceTaskData(
    size_t task_id, DeviceTaskType task_type, const cudaStream_t& stream,
    HostTaskData* host_args) {
  DeviceTaskData* p_task_data = nullptr;
  switch (task_type) {
    case kMatrixMultiplication:
      break;
    default:
      break;
  }
  return p_task_data;
}

void GPUTaskManager::EvictDeviceData(size_t task_id) {
  if (!IsTaskFinished(task_id)) exit(EXIT_FAILURE);
  ReleaseStream(task_id);
  RemoveDeviceInput(task_id);
  RemoveDeviceOutput(task_id);
  RemoveDeviceArgs(task_id);
}

void GPUTaskManager::ReleaseStream(size_t task_id) {
  std::lock_guard<std::mutex> lock(streams_mtx_);
  // Find the stream for the task
  auto iter = streams_by_task_id_.find(task_id);
  if (iter != streams_by_task_id_.end()) {
    // If stream exist destroy the stream
    cudaStreamDestroy(*(iter->second));
    // Erase the stream from the map
    streams_by_task_id_.erase(iter);
  }
}

void GPUTaskManager::RemoveDeviceInput(size_t task_id) {
  std::lock_guard<std::mutex> lock(input_data_mtx_);
  auto iter = input_ptr_by_task_id_.find(task_id);
  if (iter != input_ptr_by_task_id_.end()) {
    delete iter->second;
    input_ptr_by_task_id_.erase(iter);
  }
}

void GPUTaskManager::RemoveDeviceOutput(size_t task_id) {
  std::lock_guard<std::mutex> lock(output_data_mtx_);
  auto iter = output_ptr_by_task_id_.find(task_id);
  if (iter != output_ptr_by_task_id_.end()) {
    delete iter->second;
    output_ptr_by_task_id_.erase(iter);
  }
}

void GPUTaskManager::RemoveDeviceArgs(size_t task_id) {
  std::lock_guard<std::mutex> lock(args_mtx_);
  auto iter = args_ptr_by_task_id_.find(task_id);
  if (iter != args_ptr_by_task_id_.end()) {
    delete iter->second;
    args_ptr_by_task_id_.erase(iter);
  }
}

void GPUTaskManager::DeviceSynchronize() { cudaDeviceSynchronize(); }

cudaStream_t* GPUTaskManager::GetStream(size_t task_id) {
  std::lock_guard<std::mutex> lock(streams_mtx_);
  auto iter = streams_by_task_id_.find(task_id);
  if (iter == streams_by_task_id_.end()) {
    // If stream doesn't exist create a new CUDA stream
    cudaStream_t* p_stream = new cudaStream_t;
    cudaStreamCreate(p_stream);
    streams_by_task_id_.insert(std::make_pair(task_id, p_stream));
    return p_stream;
  } else {
    // Return the existing stream
    return iter->second;
  }
}

void GPUTaskManager::InsertInputDeviceTaskData(size_t task_id,
                                               DeviceTaskData* input) {
  std::lock_guard<std::mutex> lock(input_data_mtx_);
  input_ptr_by_task_id_.insert(std::make_pair(task_id, input));
}

void GPUTaskManager::InsertArgsDeviceTaskData(size_t task_id,
                                              DeviceTaskData* input) {
  std::lock_guard<std::mutex> lock(args_mtx_);
  args_ptr_by_task_id_.insert(std::make_pair(task_id, input));
}

void GPUTaskManager::InsertOutputDeviceTaskData(size_t task_id,
                                                DeviceTaskData* output) {
  std::lock_guard<std::mutex> lock(output_data_mtx_);
  output_ptr_by_task_id_.insert(std::make_pair(task_id, output));
}

void GPUTaskManager::TransferResultFromDevice2Host(
    DeviceTaskType task_type, const cudaStream_t& stream,
    DeviceTaskData* device_output, HostTaskData* host_output) {
  switch (task_type) {
    case kMatrixMultiplication: {
      auto* device_result =
          reinterpret_cast<MatrixMulOutputDeviceTaskData*>(device_output);
      auto* host_result = reinterpret_cast<HostMatrixData*>(host_output);

      cudaMemcpyAsync(host_result->data.GetRawData(),
                      device_result->result_matrix.GetPtr(),
                      device_result->result_matrix.GetSize(),
                      cudaMemcpyDeviceToHost, stream);
    } break;
  }
}

}  // namespace gpu
}  // namespace luck
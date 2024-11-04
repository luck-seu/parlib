#include "gpu_par/data_collections/device_data_collections/matrix_mul_device_task_data.cuh"

namespace luck {
namespace gpu {
namespace data {
namespace device {

void MatrixMulInputDeviceTaskData::SetData(
    const host::MatrixMulHostTaskData& host_data, const cudaStream_t& stream) {
  // Check the validity of the input data.
  if (host_data.lhs_matrix.n_cols != host_data.rhs_matrix.n_rows) {
    exit(EXIT_FAILURE);
  }
  if (host_data.lhs_matrix.data.size_byte != host_data.lhs_matrix.n_rows *
                                                 host_data.lhs_matrix.n_cols *
                                                 sizeof(uint32_t)) {
    exit(EXIT_FAILURE);
  }
  if (host_data.rhs_matrix.data.size_byte != host_data.rhs_matrix.n_rows *
                                                 host_data.rhs_matrix.n_cols *
                                                 sizeof(uint32_t))
    exit(EXIT_FAILURE);
  // Initialize the device input.
  n_lhs_rows_ = host_data.lhs_matrix.n_rows;
  n_lhs_cols_ = host_data.lhs_matrix.n_cols;
  lhs_matrix_.Init(host_data.lhs_matrix.data, stream);
  rhs_matrix_.Init(host_data.rhs_matrix.data, stream);
}

void MatrixMulOutputDeviceTaskData::SetData(
    const host::HostMatrixData& host_data, const cudaStream_t& stream) {
  n_rows_ = host_data.n_rows;
  n_cols_ = host_data.n_cols;
  result_matrix.Init(host_data.data, stream);
}

}  // namespace device
}  // namespace data
}  // namespace gpu
}  // namespace luck
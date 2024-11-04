#include "gpu_par/kernel/matrix_mul/matrix_multiplication_wrap.cuh"

namespace luck::parlib::gpu::kernel::matrix_mul {

using DeviceTaskData = luck::parlib::gpu::data::device::DeviceTaskData;
using MatrixMulInputDeviceTaskData =
    luck::parlib::gpu::data::device::MatrixMulInputDeviceTaskData;
using MatrixMulOutputDeviceTaskData =
    luck::parlib::gpu::data::device::MatrixMulOutputDeviceTaskData;

struct Params {
  uint32_t* lhs_matrix;
  uint32_t* rhs_matrix;
  uint32_t lhs_n_rows;
  uint32_t lhs_n_cols;

  uint32_t* output_matrix;
  uint32_t output_n_rows;
  uint32_t output_n_cols;
};

static __global__ void MatrixMultiplicationKernel(Params params) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  uint32_t computation_cost = params.output_n_rows * params.output_n_cols;

  for (unsigned int i = tid; i < computation_cost; i += step) {
    uint32_t row = i / params.output_n_cols;
    uint32_t col = i % params.output_n_cols;

    uint32_t sum = 0;
    for (uint32_t j = 0; j < params.lhs_n_cols; j++) {
      sum += params.lhs_matrix[row * params.lhs_n_cols + j] *
             params.rhs_matrix[j * params.output_n_cols + col];
    }

    params.output_matrix[row * params.output_n_cols + col] = sum;
  }
}

MatrixMultiplicationWrap* MatrixMultiplicationWrap::GetInstance() {
  if (ptr_ == nullptr) {
    ptr_ = new MatrixMultiplicationWrap();
  }
  return ptr_;
}

void MatrixMultiplicationWrap::Do(const cudaStream_t& stream,
                                  DeviceTaskData* device_input,
                                  DeviceTaskData* device_args,
                                  DeviceTaskData* device_output) {
  auto input = reinterpret_cast<MatrixMulInputDeviceTaskData*>(device_input);
  auto output = reinterpret_cast<MatrixMulOutputDeviceTaskData*>(device_output);

  Params params{.lhs_matrix = input->lhs_matrix_.GetPtr(),
                .rhs_matrix = input->rhs_matrix_.GetPtr(),
                .lhs_n_rows = input->n_lhs_rows_,
                .lhs_n_cols = input->n_lhs_cols_,
                .output_matrix = output->result_matrix.GetPtr(),
                .output_n_rows = output->n_rows_,
                .output_n_cols = output->n_cols_};

  dim3 dimBlock(64);
  dim3 dimGrid(64);
  MatrixMultiplicationKernel<<<dimGrid, dimBlock, 48 * 1024, stream>>>(params);
}

}  // namespace luck::parlib::gpu::kernel::matrix_mul
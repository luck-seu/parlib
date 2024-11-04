#ifndef LUCK_GPU_DATA_STRUCTURES_HOST_DATA_COLLECTIONS_MATRIX_MUL_HOST_TASK_DATA_CUH
#define LUCK_GPU_DATA_STRUCTURES_HOST_DATA_COLLECTIONS_MATRIX_MUL_HOST_TASK_DATA_CUH

#include <stdint.h>

#include "gpu_par/data_collections/host_buffer.cuh"
#include "gpu_par/data_collections/host_task_data.cuh"

namespace luck {
namespace gpu {
namespace data {
namespace host {

struct HostMatrixData : public HostTaskData {
  HostMatrixData(uint32_t* dat, uint32_t n_rows, uint32_t n_cols)
      : n_rows(n_rows), n_cols(n_cols) {
    data.data = dat;
    data.size_byte = n_rows * n_cols * sizeof(uint32_t);
  }

  uint32_t GetElement(uint32_t row, uint32_t col) const {
    return data.GetElement(row * n_cols + col);
  }

  HostBuffer<uint32_t> data;
  uint32_t n_rows;
  uint32_t n_cols;
};

struct MatrixMulHostTaskData : public HostTaskData {
  MatrixMulHostTaskData(const HostMatrixData& lhs_matrix,
                        const HostMatrixData& rhs_matrix)
      : lhs_matrix(lhs_matrix), rhs_matrix(rhs_matrix) {}

  HostMatrixData lhs_matrix;
  HostMatrixData rhs_matrix;
};
}  // namespace host
}  // namespace data
}  // namespace gpu
}  // namespace luck

#endif  // LUCK_GPU_DATA_STRUCTURES_HOST_DATA_COLLECTIONS_MATRIX_MUL_HOST_TASK_DATA_CUH
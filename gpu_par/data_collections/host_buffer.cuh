#ifndef GPU_PAR_DATA_STRUCTURES_HOST_BUFFER_CUH
#define GPU_PAR_DATA_STRUCTURES_HOST_BUFFER_CUH

#include <stdint.h>

namespace luck::parlib::gpu::data::host {

// Remark: `HostBuffer` does not own the data.
template <typename T>
struct HostBuffer {
  T* data;
  size_t size_byte;

  T* GetRawData() const { return data; }

  T GetElement(size_t index) const { return data[index]; }

  uint64_t GetElementSize() const { return sizeof(T); }
};
}  // namespace luck::parlib::gpu::data::host

#endif  // GPU_PAR_DATA_STRUCTURES_HOST_BUFFER_CUH
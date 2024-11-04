#ifndef GPU_DATA_STRUCTURES_HOST_BUFFER_CUH
#define GPU_DATA_STRUCTURES_HOST_BUFFER_CUH

#include <stdint.h>

namespace luck {
namespace gpu {
namespace data {
namespace host {

// Remark: `HostBuffer` does not own the data.
template <typename T>
struct HostBuffer {
  T* data;
  size_t size_byte;

  T* GetRawData() const { return data; }

  T GetElement(size_t index) const { return data[index]; }

  uint64_t GetElementSize() const { return sizeof(T); }
};
}  // namespace host
}  // namespace data
}  // namespace gpu
}  // namespace luck

#endif  // GPU_DATA_STRUCTURES_HOST_BUFFER_CUH
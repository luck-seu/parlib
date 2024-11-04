#ifndef GPU_UTIL_CUDA_CHECK_CUH_
#define GPU_UTIL_CUDA_CHECK_CUH_

#include <iostream>

static void HandleError(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(error) << " in " << file
              << " at line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(error) HandleError(error, __FILE__, __LINE__)

static void LogDebug(int val, const char* file, int line) {
  std::cout << val << " in " << file << " at line " << line << std::endl;
}

static void LogDebug(const std::string& str, const char* file, int line) {
  std::cout << str << " in " << file << " at line " << line << std::endl;
}

#define CUDA_LOG_DEBUG(info) LogDebug(info, __FILE__, __LINE__)

#endif  // GPU_UTIL_CUDA_CHECK_CUH_
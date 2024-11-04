cmake_minimum_required(VERSION 3.18)
project(parlib_gpu_par LANGUAGES CUDA)

#######################
# Cmake options
#######################

# Set Cmake Standard
set(CMAKE_CXX_STANDARD 17)

# Directories shortcuts
set(GPU_PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Libraries
#######################
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# CUDA
if (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
    message("ARCH_FLAGS = ${ARCH_FLAGS}")
    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
else ()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif ()

#######################
# Artifacts
#######################
file(GLOB GPU_PAR_SOURCES
        "${GPU_PROJECT_ROOT_DIR}/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/app/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/device_data_collections/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/host_data_collections/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/kernel_func/*.cu"
        "${GPU_PROJECT_ROOT_DIR}/util/*.cu"
        )

file(GLOB GPU_PAR_HEADERS
        "${GPU_PROJECT_ROOT_DIR}/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/app/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/device_data_collections/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/data_collections/host_data_collections/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/kernel_func/*.cuh"
        "${GPU_PROJECT_ROOT_DIR}/util/*.cuh"
        )

add_library(lib_gpu_par ${GPU_PAR_SOURCES} ${GPU_PAR_HEADERS})

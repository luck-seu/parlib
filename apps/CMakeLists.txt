cmake_minimum_required(VERSION 3.12)
project(parlib_apps LANGUAGES CXX CUDA)

# Directories
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

# Directories shortcuts
set(APPS_PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#######################
# Libraries
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
        ${PROJECT_ROOT_DIR}/cpu_par
        ${PROJECT_ROOT_DIR}/gpu_par
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#######################
# Artifacts
#######################
file(GLOB APPS_SOURCES
        "*.cpp"
        "${APPS_PROJECT_ROOT_DIR}/matrix_mul/*.cpp"
        )
message(STATUS "APPS source files: ${APPS_SOURCES}")

file(GLOB APPS_HEADERS
        "*.h"
        "${APPS_PROJECT_ROOT_DIR}/matrix_mul/*.h"
        )
message(STATUS "APPS header files: ${APPS_HEADERS}")

add_subdirectory(client)

#######################
# Generate parlib_apps
#######################
add_library(lib_apps ${APPS_SOURCES} ${APPS_HEADERS})
target_link_libraries(lib_apps
        ${FOLLY_LIBRARIES}
        )

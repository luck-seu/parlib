cmake_minimum_required(VERSION 3.12)
project(parlib)

###### Cmake options ######
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -Wall")
add_compile_definitions(FLAG_GPU_DEBUG=1)

# Set default cmake type to Debug
if (NOT CMAKE_BUILD_TYPE)
    # cmake default flags with relwithdebinfo is -O2 -g
    # cmake default flags with release is -O3 -DNDEBUG
    set(CMAKE_BUILD_TYPE "Debug")
endif ()
message(STATUS "[ParLib] will build in type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")

###### Directories shortcuts ######
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)

#######################
# Libraries
#######################
# folly
find_package(FOLLY REQUIRED)
include_directories(SYSTEM ${FOLLY_INCLUDE_DIR})

###### Add third party to the project, if necessary ######
# gflags
find_package(gflags CONFIG REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})

#######################
# Artifacts
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
)
add_subdirectory(cpu_par)
add_subdirectory(gpu_par)
add_subdirectory(apps)

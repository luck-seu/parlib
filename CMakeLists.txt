cmake_minimum_required(VERSION 3.12)
project(miniclean)

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
message(STATUS "[Graph Systems] will build in type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")

###### Directories shortcuts ######
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(CMAKE_MODULE_PATH ${PROJECT_ROOT_DIR}/cmake)

#######################
# Libraries
#######################
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake" OPTIONAL)
    if (NOT JEMALLOC_FOUND)
        message(STATUS "jemalloc not found, build without jemalloc")
    else ()
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
    endif ()
endif ()

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
add_subdirectory(multithreading)
add_subdirectory(gpu)
add_subdirectory(miniclean)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/multithreading
)

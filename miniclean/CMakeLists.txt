cmake_minimum_required(VERSION 3.12)
project(graph_systems_miniclean LANGUAGES CXX CUDA)

# Directories
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Libraries
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
        ${PROJECT_ROOT_DIR}/multithreading
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#######################
# Tests
#######################
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Artifacts
#######################
file(GLOB GRAPH_MINICLEAN_SOURCES
        "*.cpp"
        )
message(STATUS "CORE source files: ${GRAPH_MINICLEAN_SOURCES}")
list(FILTER GRAPH_MINICLEAN_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")

file(GLOB GRAPH_MINICLEAN_HEADERS
        "*.h"
        )
message(STATUS "CORE header files: ${GRAPH_MINICLEAN_HEADERS}")

add_subdirectory(apps)

#############################
# Generate libgraph_miniclean
#############################
add_library(graph_systems_miniclean ${GRAPH_MINICLEAN_SOURCES} ${GRAPH_MINICLEAN_HEADERS})
target_link_libraries(graph_systems_miniclean
        ${FOLLY_LIBRARIES}
        ${LIBFMT_LIBRARY}
        )

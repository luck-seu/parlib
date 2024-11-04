cmake_minimum_required(VERSION 3.12)
project(graph_system_miniclean_apps CXX CUDA)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/miniclean)

#######################
# Artifacts
#######################
# libgraph_sys_gpu
link_directories(${PROJECT_ROOT_DIR}/gpu/lib)

file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (appfile ${appfiles})
    message(STATUS "Found app: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile})
    if (${app} MATCHES ".*gpu.*")
        target_link_libraries("${app}_exec"
                graph_systems_miniclean
                graph_sys_gpu
                graph_systems_core
                yaml-cpp
                gflags
                cudart
                rt
                )
    else ()
        target_link_libraries("${app}_exec"
                graph_systems_miniclean
                graph_systems_core
                yaml-cpp
                gflags
                )
    endif ()
endforeach ()
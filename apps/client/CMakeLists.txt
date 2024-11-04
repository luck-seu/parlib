cmake_minimum_required(VERSION 3.12)
project(parlib_apps_client CXX CUDA)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)

#######################
# Artifacts
#######################
link_directories(${PROJECT_ROOT_DIR}/gpu/lib)

file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach (appfile ${appfiles})
    message(STATUS "Found app: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile})
    if (${app} MATCHES ".*gpu.*")
        target_link_libraries("${app}_exec"
                lib_apps
                lib_gpu_par
                lib_cpu_par
                gflags
                cudart
                rt
                )
    else ()
        target_link_libraries("${app}_exec"
                lib_apps
                lib_cpu_par
                gflags
                )
    endif ()
endforeach ()
project(parlib_cpu_par)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Libraries
#######################
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

#######################
# Artifacts
#######################

file(GLOB CPU_PAR_SOURCES
        "*.cpp"
        )
message(STATUS "CPU_PAR source files: ${CPU_PAR_SOURCES}")

file(GLOB CPU_PAR_HEADERS
        "*.h"
        )
message(STATUS "CPU_PAR header files: ${CPU_PAR_HEADERS}")

#########################
# Generate parlib_cpu_par
#########################
add_library(lib_cpu_par ${CPU_PAR_SOURCES} ${CPU_PAR_HEADERS})

target_link_libraries(lib_cpu_par
        gflags
        ${FOLLY_LIBRARIES}
        )

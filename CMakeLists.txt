cmake_minimum_required(VERSION 3.20)

project(CUDANet
        LANGUAGES CXX CUDA
)

find_package(CUDAToolkit QUIET)

# Set USE_CUDA option based on whether CUDA was found
if(CUDAToolkit_FOUND)
    option(USE_CUDA "Use CUDA implementation" ON)
else()
    option(USE_CUDA "Use CUDA implementation" OFF)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "Building library with CUDA support")
else()
    message(STATUS "CUDA not found or disabled. Defaulting to CPU implementation.")
endif()


file(GLOB_RECURSE CPU_SOURCES
    src/*.cpp
    src/layers/*.cpp
)

set(LIBRARY_SOURCES ${CPU_SOURCES})

if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        src/backends/cuda/*.cu
        src/backends/cuda/kernels/*.cu
    )
    set(LIBRARY_SOURCES ${LIBRARY_SOURCES} ${CUDA_SOURCES})
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build static library
add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES})

if(USE_CUDA)
    # Enable relocatable device code for proper template instantiation across translation units
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
    )
    target_link_libraries(${PROJECT_NAME} CUDA::cudart)
endif()

# Set include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# Add testing subdirectory
add_subdirectory(test)
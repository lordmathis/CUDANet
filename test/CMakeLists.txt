find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(USE_CUDA)
    file(GLOB CUDA_SOURCES
        cuda/*.cu
    )
endif()

add_executable(test_main 
    ${CUDA_SOURCES}
    test_main.cpp
)

if(USE_CUDA) 
    set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(test_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet)

add_test(NAME TestMain COMMAND test_main)
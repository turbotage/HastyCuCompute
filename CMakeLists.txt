cmake_minimum_required(VERSION 3.28)

project("HastyCuCompute" VERSION 0.1)

include(cmake/CPM.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(HastyCuCompute "lib/main.cpp")

target_sources(HastyCuCompute 
    PUBLIC FILE_SET CXX_MODULES FILES
        "lib/tensor.ixx"
        "lib/nufft.ixx"
)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDAToolkit REQUIRED)


find_package(matx CONFIG REQUIRED)
target_link_libraries(HastyCuCompute PRIVATE matx::matx)

target_link_libraries(HastyCuCompute PRIVATE CUDA::toolkit)
target_link_libraries(HastyCuCompute PRIVATE CUDA::nvrtc)
target_link_libraries(HastyCuCompute PRIVATE CUDA::cudart)

find_library(finufft REQUIRED 
    NAMES 
        libfinufft finufft
    HINTS
        "${finufft_ROOT}/lib" 
)
find_library(cufinufft REQUIRED
    NAMES 
        libcufinufft cufinufft
    HINTS
        "${finufft_ROOT}/lib"
)

message(${finufft})
message(${cufinufft})
#message(${finufft_INCLUDE_DIR})

target_link_libraries(HastyCuCompute PRIVATE ${finufft})
target_link_libraries(HastyCuCompute PRIVATE ${cufinufft})
target_include_directories(HastyCuCompute PRIVATE "${finufft_ROOT}/include")

set_property(TARGET HastyCuCompute PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(HastyCuCompute PRIVATE ${FINUFFT_LIB})
#target_link_libraries(HastyCuCompute PRIVATE ${CUFINUFFT_LIB})
#target_include_directories(HastyCuCompute PRIVATE ${FINUFFT_INCLUDE_DIR}) 


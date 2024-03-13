cmake_minimum_required(VERSION 3.28)

project("HastyCuCompute" VERSION 0.1)

enable_language(CUDA)

include(cmake/CPM.cmake)

#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(HastyCuCompute "lib/main.cpp")

set_property(TARGET HastyCuCompute PROPERTY CXX_STANDARD 23)
set_property(TARGET HastyCuCompute PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD 20)
set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD_REQUIRED ON)

target_sources(HastyCuCompute 
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            "lib/util.ixx"
            "lib/tensor.ixx"
            "lib/nufft.ixx"
            "lib/trace.ixx"
)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDAToolkit REQUIRED)

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

#set_property(TARGET HastyCuCompute PROPERTY CUDA_SEPARABLE_COMPILATION ON)

find_package(Torch REQUIRED)
target_link_libraries(HastyCuCompute PRIVATE ${TORCH_LIBRARIES})

find_package(HighFive CONFIG REQUIRED)
target_link_libraries(HastyCuCompute PRIVATE HighFive)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

#find_package(matx CONFIG REQUIRED)
#target_link_libraries(HastyCuCompute PRIVATE matx::matx)

#find_package(mathdx REQUIRED COMPONENTS cufftdx)
#target_link_libraries(HastyCuCompute PRIVATE mathdx::cufftdx)





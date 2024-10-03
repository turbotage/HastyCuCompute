cmake_minimum_required(VERSION 3.30)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Set experimental flag to enable `import std` support from CMake.
# This must be enabled before C++ language support.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
  # This specific value changes as experimental support evolves. See
  # `Help/dev/experimental.rst` in the CMake source corresponding to
  # your CMake build for the exact value to use.
  "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")

project("HastyCuCompute" VERSION 0.1)


set(CMAKE_CXX_MODULE_STD 1 LANGUAGES CXX CUDA)



#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(HastyCuCompute SHARED
    "lib/interface.cpp"
    "lib/py_interface.cpp"
    "lib/hdf5.cpp"
    "lib/tensor/tensor.cpp"
)
target_compile_features(HastyCuCompute 
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23)

include(cmake/CPM.cmake)
include(FetchContent)

#set_property(TARGET HastyCuCompute PROPERTY CXX_STANDARD 23)
#set_property(TARGET HastyCuCompute PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
#set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD 20)
#set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD_REQUIRED ON)

target_sources(HastyCuCompute
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            "lib/util/util_funcs.ixx"
            "lib/util/util_idx.ixx"
            "lib/util/util_io.ixx"
            "lib/util/util_meta.ixx"
            "lib/util/util_span.ixx"   
            "lib/util/util_torch.ixx"
            "lib/util/util_typing.ixx"
            "lib/util/util.ixx"
            "lib/op/op.ixx"
            "lib/op/cg.ixx"
            "lib/op/min.ixx"
            "lib/mri/trajectory.ixx"
            "lib/mri/sense.ixx"
            "lib/hdf5.ixx"
            "lib/tensor/tensor.ixx"
            "lib/tensor/tensor_impl.ixx"
            "lib/nufft.ixx"
            "lib/pch.ixx"
            "lib/threading.ixx"
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

#find_package(Matplot++ CONFIG REQUIRED)
#target_link_libraries(HastyCuCompute PRIVATE Matplot++::cimg Matplot++::matplot)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET HastyCuCompute
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:HastyCuCompute>)
endif (MSVC)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


target_link_libraries(HastyCuCompute PRIVATE Python3::Python)
target_link_libraries(HastyCuCompute PRIVATE pybind11::module)

set_target_properties(HastyCuCompute PROPERTIES
         INTERPROCEDURAL_OPTIMIZATION ON
         CXX_VISIBILITY_PRESET "hidden"
         VISIBILITY_INLINES_HIDDEN ON
     )

set_target_properties(HastyCuCompute PROPERTIES INSTALL_RPATH "$ORIGIN")


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/pcm)
    # Find the base include directory for LibTorch
    foreach(dir ${TORCH_INCLUDE_DIRS})
        message("Checking " ${dir})
        if(EXISTS "${dir}/ATen/core/ivalue.h")
            set(TORCH_IVALUE_INCLUDE_DIR ${dir})
        endif()
        if(EXISTS "${dir}/torch/torch.h")
            set(TORCH_TORCH_INCLUDE_DIR ${dir})
        endif()
        if(EXISTS "${dir}/torch/jit.h")
            set(TORCH_JIT_INCLUDE_DIR ${dir})
        endif()
    endforeach()

    set(PCM_INCLUDE_DIRS 
            ${TORCH_IVALUE_INCLUDE_DIR} 
            ${TORCH_TORCH_INCLUDE_DIR}
            ${TORCH_JIT_INCLUDE_DIR})
    
    list(REMOVE_DUPLICATES PCM_INCLUDE_DIRS)


    
    set(PCM_INCLUDE_FLAGS "")
    foreach(dir ${PCM_INCLUDE_DIRS})
        list(APPEND PCM_INCLUDE_FLAGS "-I${dir}")
    endforeach()

    message("PCM_INCLUDE_FLAGS:" ${PCM_INCLUDE_FLAGS})
    # Join the include flags into a single string
    string(JOIN \t PCM_INCLUDE_FLAGS ${PCM_INCLUDE_FLAGS})
    message("PCM_INCLUDE_FLAGS:" ${PCM_INCLUDE_FLAGS})

    # Define a custom target for GCM files
    add_custom_target(PCMFiles ALL
        COMMAND ${CMAKE_CXX_COMPILER} -x c++-header -std=gnu++23 -xc++-user-header --precompile
            ${PCM_INCLUDE_FLAGS}
            -o ${CMAKE_BINARY_DIR}/pcm/ivalue.pcm ${TORCH_IVALUE_INCLUDE_DIR}/ATen/core/ivalue.h
        COMMAND ${CMAKE_CXX_COMPILER} -x c++-header -std=gnu++23 -xc++-user-header --precompile
            ${PCM_INCLUDE_FLAGS}
            -o ${CMAKE_BINARY_DIR}/pcm/torch.pcm ${TORCH_TORCH_INCLUDE_DIR}/torch/torch.h
        COMMAND ${CMAKE_CXX_COMPILER} -x c++-header -std=gnu++23 -xc++-user-header --precompile
            ${PCM_INCLUDE_FLAGS}
            -o ${CMAKE_BINARY_DIR}/pcm/jit.pcm ${TORCH_JIT_INCLUDE_DIR}/torch/jit.h
        COMMAND ${CMAKE_CXX_COMPILER} -x c++-header -std=gnu++23 -xc++-user-header --precompile
            -I${CUDA_INCLUDE_DIRS}
            -o ${CMAKE_BINARY_DIR}/pcm/cuComplex.pcm ${CUDA_INCLUDE_DIRS}/cuComplex.h
        DEPENDS ${TORCH_IVALUE_INCLUDE_DIR}/ATen/core/ivalue.h
                ${TORCH_TORCH_INCLUDE_DIR}/torch/torch.h
                ${TORCH_JIT_INCLUDE_DIR}/torch/jit.h
                ${CUDA_INCLUDE_DIRS}/cuComplex.h
    )

    add_dependencies(HastyCuCompute PCMFiles)

    target_compile_options(HastyCuCompute PRIVATE
        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/ivalue.pcm
        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/torch.pcm
        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/jit.pcm
        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/cuComplex.pcm
    )
    #target_include_directories(HastyCuCompute PRIVATE ${CMAKE_BINARY_DIR}/pcm)

else()
    message(FATAL_ERROR "Only Clang is supported for header units")
endif()



#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    target_link_libraries(HastyCuCompute PRIVATE ${Python_DEBUG_LIBRARIES})
#else()
#    target_link_libraries(HastyCuCompute PRIVATE ${Python_LIBRARIES})
#endif()

install(FILES include/interface.hpp include/pch.hpp DESTINATION include/HastyCuCompute)

target_include_directories(HastyCuCompute PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${pybind11_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${Python_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)



add_executable(HastyCuTest "test/main.cpp")
target_link_libraries(HastyCuTest PRIVATE HastyCuCompute pybind11::headers)
target_link_libraries(HastyCuTest PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(HastyCuTest PRIVATE Python3::Python)


#if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#    target_link_libraries(HastyCuTest PRIVATE ${Python_DEBUG_LIBRARIES})
#else()
#    target_link_libraries(HastyCuTest PRIVATE ${Python_LIBRARIES})
#endif()

target_include_directories(HastyCuTest PRIVATE
    $<TARGET_PROPERTY:HastyCuCompute,INTERFACE_INCLUDE_DIRECTORIES>
)
# Ensure that the Python library is linked with the correct flags

#find_package(matx CONFIG REQUIRED)
#target_link_libraries(HastyCuCompute PRIVATE matx::matx)

#find_package(mathdx REQUIRED COMPONENTS cufftdx)
#target_link_libraries(HastyCuCompute PRIVATE mathdx::cufftdx)


# Ensure that the Python library is linked with the correct flags

#find_package(matx CONFIG REQUIRED)
#target_link_libraries(HastyCuCompute PRIVATE matx::matx)

#find_package(mathdx REQUIRED COMPONENTS cufftdx)
#target_link_libraries(HastyCuCompute PRIVATE mathdx::cufftdx)

#set(RUNTIME_DEPENDENCIES
#)

include(GNUInstallDirs)
install(TARGETS HastyCuCompute EXPORT HastyCuComputeTargets
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*"
        POST_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*"
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    FILE_SET HEADERS DESTINATION include/HastyCuCompute
    FILE_SET CXX_MODULES DESTINATION include/HastyCuCompute/modules
)

# Install the export set
install(EXPORT HastyCuComputeTargets
    FILE HastyCuComputeTargets.cmake
    NAMESPACE HastyCuCompute::
    DESTINATION lib/cmake/HastyCuCompute
)

# Install the include files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include/HastyCuCompute
)

# Install the CMake package configuration file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HastyCuComputeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(HastyCuComputeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/HastyCuComputeConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HastyCuComputeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HastyCuComputeConfigVersion.cmake"
    DESTINATION lib/cmake/HastyCuCompute
)

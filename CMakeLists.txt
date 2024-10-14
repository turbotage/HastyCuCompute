cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")

project("HastyCuCompute" VERSION 0.1 LANGUAGES CXX)


# Add the unsupported compiler flag explicitly
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

add_library(HastyCuCompute SHARED
    "lib/hdf5.cpp"
    "lib/tensor/mod_tensor.cpp"
    "lib/interface/interface.cpp"
    "lib/interface/py_interface.cpp"
)
target_compile_features(HastyCuCompute 
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23)

set_target_properties(HastyCuCompute PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES ON
    CXX_MODULE_STD ON
)

target_link_libraries(HastyCuCompute PRIVATE c++ c++abi)

include(cmake/CPM.cmake)
include(FetchContent)

target_sources(HastyCuCompute
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            "lib/mri/trajectory.cppm"
            "lib/mri/sense.cppm"
            "lib/op/op.cppm"
            "lib/op/cg.cppm"
            "lib/op/min.cppm"
            "lib/tensor/impl_tensor_extrinsic_math.cppm"
            "lib/tensor/impl_tensor_extrinsic_operator.cppm"
            "lib/tensor/impl_tensor_intrinsic_inplace.cppm"
            "lib/tensor/impl_tensor_intrinsic_math.cppm"
            "lib/tensor/impl_tensor_intrinsic_operator.cppm"
            "lib/tensor/impl_tensor_intrinsic.cppm"
            "lib/tensor/mod_tensor_base.cppm"
            "lib/tensor/mod_tensor_caching.cppm"
            "lib/tensor/mod_tensor_extrinsic.cppm"
            "lib/tensor/mod_tensor_factory.cppm"
            "lib/tensor/mod_tensor_intrinsic.cppm"
            "lib/tensor/mod_tensor.cppm"
            "lib/util/util_funcs.cppm"
            "lib/util/util_idx.cppm"
            "lib/util/util_io.cppm"
            "lib/util/util_meta.cppm"
            "lib/util/util_span.cppm"   
            "lib/util/util_torch.cppm"
            "lib/util/util_typing.cppm"
            "lib/util/util.cppm"
            "lib/hdf5.cppm"
            "lib/nufft.cppm"
            "lib/threading.cppm"
            "lib/trace.cppm"
)

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

target_link_libraries(HastyCuCompute PRIVATE ${finufft})
target_link_libraries(HastyCuCompute PRIVATE ${cufinufft})
target_include_directories(HastyCuCompute PRIVATE "${finufft_ROOT}/include")

find_package(Torch REQUIRED)
target_link_libraries(HastyCuCompute PRIVATE ${TORCH_LIBRARIES})

find_package(HighFive CONFIG REQUIRED)
target_link_libraries(HastyCuCompute PRIVATE HighFive)

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

target_compile_definitions(HastyCuCompute PUBLIC _GLIBCXX_USE_CXX11_ABI=1)

set_target_properties(HastyCuCompute PROPERTIES INSTALL_RPATH "$ORIGIN")

#target_precompile_headers(HastyCuCompute PRIVATE "include/pch.hpp")

#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#
#    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/pcm)
#    # Find the base include directory for LibTorch
#    foreach(dir ${TORCH_INCLUDE_DIRS})
#        message("Checking " ${dir})
#        if(EXISTS "${dir}/ATen/core/ivalue.h")
#            set(TORCH_IVALUE_INCLUDE_DIR ${dir})
#        endif()
#        if(EXISTS "${dir}/torch/torch.h")
#            set(TORCH_TORCH_INCLUDE_DIR ${dir})
#        endif()
#        if(EXISTS "${dir}/torch/jit.h")
#            set(TORCH_JIT_INCLUDE_DIR ${dir})
#        endif()
#    endforeach()
#
#    set(PCM_INCLUDE_DIRS 
#            ${TORCH_IVALUE_INCLUDE_DIR} 
#            ${TORCH_TORCH_INCLUDE_DIR}
#            ${TORCH_JIT_INCLUDE_DIR})
#    
#    list(REMOVE_DUPLICATES PCM_INCLUDE_DIRS)
#
#
#    
#    set(PCM_INCLUDE_FLAGS "")
#    foreach(dir ${PCM_INCLUDE_DIRS})
#        list(APPEND PCM_INCLUDE_FLAGS "-I${dir}")
#    endforeach()
#
#    # Join the include flags into a single string
#    string(JOIN \t PCM_INCLUDE_FLAGS ${PCM_INCLUDE_FLAGS})
#
#    # Define a custom target for GCM files
#    add_custom_target(PCMFiles ALL
#        COMMAND ${CMAKE_CXX_COMPILER} 
#            -x c++-header -stdlib=libc++ -std=gnu++23 -xc++-user-header --precompile
#            ${PCM_INCLUDE_FLAGS}
#            -o ${CMAKE_BINARY_DIR}/pcm/ivalue.pcm ${TORCH_IVALUE_INCLUDE_DIR}/ATen/core/ivalue.h
#        COMMAND ${CMAKE_CXX_COMPILER} 
#            -x c++-header -stdlib=libc++ -std=gnu++23 -xc++-user-header --precompile
#            ${PCM_INCLUDE_FLAGS}
#            -o ${CMAKE_BINARY_DIR}/pcm/torch.pcm ${TORCH_TORCH_INCLUDE_DIR}/torch/torch.h
#        COMMAND ${CMAKE_CXX_COMPILER} 
#            -x c++-header -stdlib=libc++ -std=gnu++23 -xc++-user-header --precompile
#            ${PCM_INCLUDE_FLAGS}
#            -o ${CMAKE_BINARY_DIR}/pcm/jit.pcm ${TORCH_JIT_INCLUDE_DIR}/torch/jit.h
#        COMMAND ${CMAKE_CXX_COMPILER} 
#            -x c++-header -stdlib=libc++ -std=gnu++23 -xc++-user-header --precompile
#            -I${CUDA_INCLUDE_DIRS}
#            -o ${CMAKE_BINARY_DIR}/pcm/cuComplex.pcm ${CUDA_INCLUDE_DIRS}/cuComplex.h
#        DEPENDS ${TORCH_IVALUE_INCLUDE_DIR}/ATen/core/ivalue.h
#                ${TORCH_TORCH_INCLUDE_DIR}/torch/torch.h
#                ${TORCH_JIT_INCLUDE_DIR}/torch/jit.h
#                ${CUDA_INCLUDE_DIRS}/cuComplex.h
#    )
#
#    add_dependencies(HastyCuCompute PCMFiles)
#
#    target_compile_options(HastyCuCompute PRIVATE
#        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/ivalue.pcm
#        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/torch.pcm
#        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/jit.pcm
#        -fmodule-file=${CMAKE_BINARY_DIR}/pcm/cuComplex.pcm
#    )
#    #target_include_directories(HastyCuCompute PRIVATE ${CMAKE_BINARY_DIR}/pcm)
#
#else()
#    message(FATAL_ERROR "Only Clang is supported for header units")
#endif()









install(FILES include/interface.hpp DESTINATION include/HastyCuCompute)

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



################ HASTY-CU-TEST ####################

add_executable(HastyCuTest "test/main.cpp")

target_compile_features(HastyCuTest 
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23
)

set_target_properties(HastyCuTest PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES ON
    CXX_MODULE_STD ON
)

target_link_libraries(HastyCuTest PRIVATE
    HastyCuCompute
    ${TORCH_LIBRARIES}
    pybind11::headers
    Python3::Python
    c++ 
    c++abi
)

target_include_directories(HastyCuTest PRIVATE
    $<TARGET_PROPERTY:HastyCuCompute,INTERFACE_INCLUDE_DIRECTORIES>
)


include(GNUInstallDirs)
install(TARGETS HastyCuCompute EXPORT HastyCuComputeTargets
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
        POST_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
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

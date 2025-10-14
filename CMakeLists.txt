cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "/opt/gcc-15")

project("HastyCuCompute" VERSION 0.1 LANGUAGES CXX)

set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

message("Hello ${pybind11_DIR}")
# Add the unsupported compiler flag explicitly
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

add_library(HastyCuCompute SHARED
    "lib/hdf5.cpp"
    "lib/tensor/mod_tensor.cpp"
    "lib/interface/interface.cpp"
    "lib/interface/py_interface.cpp"
    "lib/trace/trace.cpp"
    "lib/trace/trace_cache.cpp"
    "lib/fft/fft.cpp"
    "lib/fft/vkfft.cpp"
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

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HastyCuCompute PRIVATE -ftime-trace)
endif()

#target_link_libraries(HastyCuCompute PRIVATE c++ c++abi)

include(cmake/CPM.cmake)
include(FetchContent)

target_sources(HastyCuCompute
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            "lib/extwrap/wrap_torch.cppm"
            "lib/fft/fft.cppm"
            "lib/fft/nufft.cppm"
            "lib/fft/toeplitz.cppm"
            "lib/fft/vkfft.cppm"
            "lib/mri/adjoint.cppm"
            "lib/mri/forward.cppm"
            "lib/mri/misc.cppm"
            "lib/mri/mri.cppm"
            "lib/mri/normal.cppm"
            "lib/mri/trajectory.cppm"
            "lib/op/alg.cppm"
            "lib/op/op.cppm"
            "lib/op/min.cppm"
            "lib/op/cg.cppm"
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
            "lib/util/util_concepts.cppm"
            "lib/util/util_containers.cppm"
            "lib/util/util_funcs.cppm"
            "lib/util/util_idx.cppm"
            "lib/util/util_io.cppm"
            "lib/util/util_meta.cppm"
            "lib/util/util_span.cppm"   
            "lib/util/util_torch.cppm"
            "lib/util/util_typing.cppm"
            "lib/util/util.cppm"
            "lib/hdf5.cppm"
            "lib/threading.cppm"
            "lib/trace/trace.cppm"
            "lib/trace/trace_cache.cppm"
            "lib/nvrtc/nvrtc.cppm"
)

set(DB_PRODUCTION_MODE ON)
FetchContent_Declare(
  battery-embed
  GIT_REPOSITORY https://github.com/batterycenter/embed.git
  GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(HastyCuCompute "lib/fft/kernels/toeplitz_load_2D.cu")
b_embed(HastyCuCompute "lib/fft/kernels/toeplitz_load_3D.cu")

get_target_property(_incs HastyCuCompute INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _incs "${EMBED_BINARY_DIR}/autogen/hastycucompute/include")
set_target_properties(HastyCuCompute PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")

find_package(CUDAToolkit REQUIRED)

target_include_directories(HastyCuCompute
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT"  # folder with VkFFT headers
)
target_compile_definitions(HastyCuCompute PRIVATE VKFFT_BACKEND=1)

target_link_libraries(HastyCuCompute PRIVATE CUDA::toolkit)
target_link_libraries(HastyCuCompute PRIVATE CUDA::nvrtc)
target_link_libraries(HastyCuCompute PRIVATE CUDA::cudart)
target_link_libraries(HastyCuCompute PRIVATE CUDA::cufft)
target_link_libraries(HastyCuCompute PRIVATE CUDA::cuda_driver)

get_target_property(NVRTC_PATH CUDA::nvrtc LOCATION)
get_target_property(CUDART_PATH CUDA::cudart LOCATION)

get_filename_component(NVRTC_FILENAME "${NVRTC_PATH}" NAME)
get_filename_component(CUDART_FILENAME "${CUDART_PATH}" NAME)

add_custom_command(TARGET HastyCuCompute POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVRTC_PATH}" "$<TARGET_FILE_DIR:HastyCuCompute>/${NVRTC_FILENAME}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDART_PATH}" "$<TARGET_FILE_DIR:HastyCuCompute>/${CUDART_FILENAME}"
)

# this gives access to cufinufft
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)
# this disables finufft leaving only cufinufft available
set(FINUFFT_USE_CPU ON CACHE BOOL "" FORCE)
set(FINUFFT_USE_DUCC0 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  finufft
  GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
  GIT_TAG        master #v2.4.1
)

FetchContent_MakeAvailable(finufft)

target_link_libraries(HastyCuCompute PRIVATE $<BUILD_INTERFACE:finufft>)
target_link_libraries(HastyCuCompute PRIVATE $<BUILD_INTERFACE:cufinufft>)

set_target_properties(cufinufft PROPERTIES EXPORT_EXCLUDE TRUE)
set_target_properties(finufft PROPERTIES EXPORT_EXCLUDE TRUE)


FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        12.0.0
) # 10.2.1
FetchContent_MakeAvailable(fmt)

target_link_libraries(HastyCuCompute PRIVATE $<BUILD_INTERFACE:fmt::fmt>)

set_target_properties(fmt PROPERTIES EXPORT_EXCLUDE TRUE)



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
)

target_include_directories(HastyCuTest PRIVATE
    $<TARGET_PROPERTY:HastyCuCompute,INTERFACE_INCLUDE_DIRECTORIES>
)

################ SMALL-TORCH-TESTS ####################
add_executable(HastySmallTests "test/torch_tests.cpp")
target_compile_features(HastySmallTests 
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23
)
set_target_properties(HastySmallTests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES ON
    #CXX_MODULE_STD ON
)
target_link_libraries(HastySmallTests PRIVATE
    ${TORCH_LIBRARIES}
    pybind11::headers
    Python3::Python
)
################# VKFFT TESTS ####################
add_executable(VKFFTTest "test/fft_tests.cpp")
target_compile_features(VKFFTTest 
    PRIVATE cxx_std_23
    INTERFACE cxx_std_23
)
set_target_properties(VKFFTTest PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES ON
    #CXX_MODULE_STD ON
)

target_include_directories(VKFFTTest
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/vkFFT"  # folder with VkFFT headers
)
target_compile_definitions(VKFFTTest PRIVATE VKFFT_BACKEND=1)

target_link_libraries(VKFFTTest PRIVATE CUDA::toolkit)
target_link_libraries(VKFFTTest PRIVATE CUDA::nvrtc)
target_link_libraries(VKFFTTest PRIVATE CUDA::cudart)
target_link_libraries(VKFFTTest PRIVATE CUDA::cufft)
target_link_libraries(VKFFTTest PRIVATE CUDA::cuda_driver)


################ INSTALL ####################
include(GNUInstallDirs)
install(TARGETS HastyCuCompute EXPORT HastyCuComputeTargets
    RUNTIME_DEPENDENCIES
        #PRE_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
        #POST_EXCLUDE_REGEXES ".*libstdc\+\+.*" ".*libgcc.*" ".*libc.so.*" ".*libc\+\+.*"
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

install(FILES "${NVRTC_PATH}" "${CUDART_PATH}" DESTINATION lib)

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

install(DIRECTORY ${finufft_SOURCE_DIR}/include/
    DESTINATION include/finufft
    FILES_MATCHING PATTERN "*.h"
)

install(CODE "file(REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/finufft/examples\")")
cmake_minimum_required(VERSION 3.30)

project("HastyCuCompute" VERSION 0.1)

enable_language(CUDA)

set(CMAKE_BUILD_PARALLEL_LEVEL 8)

include(cmake/CPM.cmake)
include(FetchContent)

#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(HastyCuCompute SHARED
    "lib/interface.cpp"
    "lib/py_interface.cpp"
    "lib/hdf5.cpp"
    "lib/tensor.cpp"
)

set_property(TARGET HastyCuCompute PROPERTY CXX_STANDARD 23)
set_property(TARGET HastyCuCompute PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD 20)
set_property(TARGET HastyCuCompute PROPERTY CUDA_STANDARD_REQUIRED ON)

target_sources(HastyCuCompute
    PUBLIC 
        FILE_SET CXX_MODULES FILES
            "lib/util/util_funcs.ixx"
            "lib/util/util_idx.ixx"
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
            "lib/tensor.ixx"
            "lib/nufft.ixx"
            "lib/threading.ixx"
            "lib/trace.ixx"
)


#FetchContent_Declare(
#  glaze
#  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
#  GIT_TAG main
#  GIT_SHALLOW TRUE
#)
#FetchContent_MakeAvailable(glaze)
#target_link_libraries(HastyCuCompute PRIVATE glaze::glaze)



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

#if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    set(Python_FIND_ABI "ON" "ANY" "ANY")
#else()
#    set(Python_FIND_ABI "OFF" "ANY" "ANY")
#endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

#message(STATUS "Python library: ${Python_LIBRARIES}")
#message(STATUS "Python debug libraries: ${Python_DEBUG_LIBRARIES}")
#message(STATUS "Python sabi libraries: ${Python_SABI_LIBRARIES}")
#message(STATUS "Python link options: ${Python_LINK_OPTIONS}")
#message(STATUS "Python debug postfix: ${Python_DEBUG_POSTFIX}")
#message(STATUS "Python include directories: ${Python_INCLUDE_DIRS}")

#message(STATUS "Python library: ${Python3_LIBRARIES}")
#message(STATUS "Python debug libraries: ${Python3_DEBUG_LIBRARIES}")
#message(STATUS "Python sabi libraries: ${Python3_SABI_LIBRARIES}")
#message(STATUS "Python link options: ${Python3_LINK_OPTIONS}")
#message(STATUS "Python debug postfix: ${Python3_DEBUG_POSTFIX}")
#message(STATUS "Python include directories: ${Python3_INCLUDE_DIRS}")


target_link_libraries(HastyCuCompute PRIVATE Python3::Python)
target_link_libraries(HastyCuCompute PRIVATE pybind11::module)

set_target_properties(HastyCuCompute PROPERTIES
         INTERPROCEDURAL_OPTIMIZATION ON
         CXX_VISIBILITY_PRESET "hidden"
         VISIBILITY_INLINES_HIDDEN ON
     )

set_target_properties(HastyCuCompute PROPERTIES INSTALL_RPATH "$ORIGIN")

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

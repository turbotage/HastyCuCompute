cmake_minimum_required(VERSION 4.0.0 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "/opt/gcc-15")

project("HastyCuCompute" VERSION 0.1 LANGUAGES CXX)

set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

message("Hello ${pybind11_DIR}")
# Add the unsupported compiler flag explicitly
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

add_library(HastyCuCompute SHARED)

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

# LIBRARY CPP SOURCES
target_sources(HastyCuCompute
	PRIVATE
		"cpp/corelib/lib/io/hdf5.cpp"
		"cpp/corelib/lib/tensor/mod_tensor.cpp"
		"cpp/corelib/lib/script/script_cache.cpp"
		"cpp/corelib/lib/fft/fft.cpp"
		"cpp/corelib/lib/fft/vkfft.cpp"
)
# LIBRARY CPP MODULE SOURCES
target_sources(HastyCuCompute
	PUBLIC 
		FILE_SET CXX_MODULES FILES
			"cpp/corelib/lib/extwrap/wrap_torch.cppm"
			"cpp/corelib/lib/fft/fft.cppm"
			"cpp/corelib/lib/fft/nufft.cppm"
			"cpp/corelib/lib/fft/toeplitz.cppm"
			"cpp/corelib/lib/fft/vkfft.cppm"
			"cpp/corelib/lib/io/hdf5.cppm"
			"cpp/corelib/lib/mri/adjoint.cppm"
			"cpp/corelib/lib/mri/forward.cppm"
			"cpp/corelib/lib/mri/misc.cppm"
			"cpp/corelib/lib/mri/mri.cppm"
			"cpp/corelib/lib/nvrtc/nvrtc.cppm"
			"cpp/corelib/lib/mri/normal.cppm"
			"cpp/corelib/lib/mri/trajectory.cppm"
			"cpp/corelib/lib/op/alg.cppm"
			"cpp/corelib/lib/op/cg.cppm"
			"cpp/corelib/lib/op/min.cppm"
			"cpp/corelib/lib/op/op.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_extrinsic_math.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_extrinsic_operator.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_intrinsic_inplace.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_intrinsic_math.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_intrinsic_operator.cppm"
			"cpp/corelib/lib/tensor/impl_tensor_intrinsic.cppm"
			"cpp/corelib/lib/tensor/mod_tensor_base.cppm"
			"cpp/corelib/lib/tensor/mod_tensor_caching.cppm"
			"cpp/corelib/lib/tensor/mod_tensor_extrinsic.cppm"
			"cpp/corelib/lib/tensor/mod_tensor_factory.cppm"
			"cpp/corelib/lib/tensor/mod_tensor_intrinsic.cppm"
			"cpp/corelib/lib/tensor/mod_tensor.cppm"
			"cpp/corelib/lib/script/tensor_proto.cppm"
			"cpp/corelib/lib/script/script_cache.cppm"
			"cpp/corelib/lib/script/script.cppm"
			"cpp/corelib/lib/util/util_concepts.cppm"
			"cpp/corelib/lib/util/util_containers.cppm"
			"cpp/corelib/lib/util/util_funcs.cppm"
			"cpp/corelib/lib/util/util_idx.cppm"
			"cpp/corelib/lib/util/util_io.cppm"
			"cpp/corelib/lib/util/util_meta.cppm"
			"cpp/corelib/lib/util/util_span.cppm"   
			"cpp/corelib/lib/util/util_torch.cppm"
			"cpp/corelib/lib/util/util_typing.cppm"
			"cpp/corelib/lib/util/util.cppm"
			"cpp/corelib/lib/threading/threading.cppm"
)

# LIB INTERFACE CPP SOURCES
target_sources(HastyCuCompute
	PRIVATE
		"cpp/corelib/interface/interface.cpp"
		"cpp/corelib/interface/py_interface.cpp"
)
# LIB INTERFACE CPP MODULE SOURCES
target_sources(HastyCuCompute
	PUBLIC 
		FILE_SET CXX_MODULES FILES
			"cpp/corelib/interface/interface_cache.cppm"
)

set(DB_PRODUCTION_MODE ON)
FetchContent_Declare(
  battery-embed
  GIT_REPOSITORY https://github.com/batterycenter/embed.git
  GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(HastyCuCompute "cpp/corelib/lib/fft/kernels/toeplitz_load_2D.cu")
b_embed(HastyCuCompute "cpp/corelib/lib/fft/kernels/toeplitz_load_3D.cu")

get_target_property(_incs HastyCuCompute INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM _incs "${EMBED_BINARY_DIR}/autogen/hastycucompute/include")
set_target_properties(HastyCuCompute PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_incs}")

find_package(CUDAToolkit REQUIRED)


target_include_directories(HastyCuCompute
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/external/vkFFT>
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


install(FILES cpp/corelib/interface/include/interface.hpp DESTINATION include/HastyCuCompute)

set(HastyCuCompute_INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/corelib/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/corelib/interface/include>
    $<INSTALL_INTERFACE:include>
)

# External dependencies
list(APPEND HastyCuCompute_INCLUDE_DIRS
    $<BUILD_INTERFACE:${pybind11_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Python_INCLUDE_DIRS}>
)

set(MODULE_CACHE_RELATIVE_PATH "module_cache")
set(TENSOR_CACHE_RELATIVE_PATH "tensor_cache")
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cpp/corelib/include/configure_file_settings.hpp.in"
	"${CMAKE_CURRENT_BINARY_DIR}/generated_includes/configure_file_settings.hpp"
)




# Apply to target
target_include_directories(HastyCuCompute 
	PUBLIC
    	${HastyCuCompute_INCLUDE_DIRS}
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated_includes>
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
	"${CMAKE_CURRENT_SOURCE_DIR}/cpp/external/vkFFT"  # folder with VkFFT headers
)
target_compile_definitions(VKFFTTest PRIVATE VKFFT_BACKEND=1)

target_link_libraries(VKFFTTest PRIVATE CUDA::toolkit)
target_link_libraries(VKFFTTest PRIVATE CUDA::nvrtc)
target_link_libraries(VKFFTTest PRIVATE CUDA::cudart)
target_link_libraries(VKFFTTest PRIVATE CUDA::cufft)
target_link_libraries(VKFFTTest PRIVATE CUDA::cuda_driver)


################ INSTALL ####################
include(GNUInstallDirs)

# Copy models folder to build directory (for Debug/Release builds)
add_custom_command(TARGET HastyCuCompute POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_SOURCE_DIR}/models
		${CMAKE_BINARY_DIR}/models
	COMMENT "Copying wavelet models to build directory"
)

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

# Install models folder at top level
install(DIRECTORY ${CMAKE_SOURCE_DIR}/models
	DESTINATION .
	FILES_MATCHING PATTERN "*.pt"
	PATTERN "*.gitkeep" EXCLUDE
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

install(DIRECTORY ${finufft_SOURCE_DIR}/include/
	DESTINATION include/finufft
	FILES_MATCHING PATTERN "*.h"
)

install(CODE "file(REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/finufft/examples\")")
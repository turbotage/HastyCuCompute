module;

#include "pch.hpp"

#include <battery/embed.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <nvrtc.h>
#include <cuda.h>
#define VKFFT_BACKEND 1
#include "vkFFT.h"

export module fft:toeplitz;

import torch_base;
import util;
import tensor;
import nvrtc;
import vkfft;

inline void CUDA_CHECK(cudaError_t err) {
	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + "\n");
	}
}

inline void CUFFT_CHECK(cufftResult err) {
	if (err != CUFFT_SUCCESS) {
		throw std::runtime_error("cuFFT error: " + std::to_string(err) + "\n");
	}
}

inline void CUDA_PRINT_LAST_ERROR() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA last error: " << cudaGetErrorString(err) << "\n";
	}
}

namespace hasty {
	namespace fft {
		
		// If the following enums are changed, 
		// also change the corresponding kernel code in 
		// lib/fft/kernels/toeplitz_load_2D.cu and lib/fft/kernels/toeplitz_load_3D.cu
		export enum class ToeplitzMultType {
			NONE=0,
			MULT=1,
			MULT_CONJ=2
		};

		export enum class ToeplitzAccumulateType {
			NONE=0,
			ACCUMULATE=1
		};

		export void transform_toeplitz_kernel(
			tensor<cuda_t, c64_t, 2>& kernel, bool clear_vkfft_plan = false
		);
		export void transform_toeplitz_kernel(
			tensor<cuda_t, c64_t, 3>& kernel, bool clear_vkfft_plan = false
		);

		export void toeplitz_multiplication(
			const tensor<cuda_t, c64_t, 3>&     input,
			tensor<cuda_t, c64_t, 3>&           output,
			const tensor<cuda_t, c64_t, 2>&     kernel,
			optrefw<tensor<cuda_t, c64_t, 2>>   scratch,
			optcrefw<tensor<cuda_t, c64_t, 2>>  mult1,
			optcrefw<tensor<cuda_t, c64_t, 2>>  mult2,
			ToeplitzMultType                    input_output_mult_type,
			ToeplitzMultType                    input_mult1_type,
			ToeplitzMultType                    output_mult1_type,
			ToeplitzMultType                    input_mult2_type,
			ToeplitzMultType                    output_mult2_type,
			ToeplitzAccumulateType              accumulate_type
		);

		export void toeplitz_multiplication(
			const tensor<cuda_t, c64_t, 4>&     input,
			tensor<cuda_t, c64_t, 4>&           output,
			const tensor<cuda_t, c64_t, 3>&     kernel,
			optrefw<tensor<cuda_t, c64_t, 3>>   scratch,
			optcrefw<tensor<cuda_t, c64_t, 3>>  mult1,
			optcrefw<tensor<cuda_t, c64_t, 3>>  mult2,
			ToeplitzMultType                    input_output_mult_type,
			ToeplitzMultType                    input_mult1_type,
			ToeplitzMultType                    output_mult1_type,
			ToeplitzMultType                    input_mult2_type,
			ToeplitzMultType                    output_mult2_type,
			ToeplitzAccumulateType              accumulate_type
		);

	}
}

// Implementation
namespace hasty {
	namespace fft {

		void launch_toeplitz_load_3D(
			const cuFloatComplex* input, 
			const cuFloatComplex* output, 
			cuFloatComplex* scratch, 
			const cuFloatComplex* mult1,
			const cuFloatComplex* mult2,
			int input_output_mult_type,
			int input_mult1_type,
			int output_mult1_type,
			int input_mult2_type,
			int output_mult2_type,
			int batch_in,
			int batch_out,
			int NX, int NY, int NZ,
			bool accumulate,
			int device_idx,
			int threads_per_block = 256)
		{
			static std::array<hasty::nvrtc::NVRTC_ModFunc, (size_t)device_idx::MAX_CUDA_DEVICES> nvrtc_modules = {};

			static const char* toeplitz_load_3D_code = b::embed<"lib/fft/kernels/toeplitz_load_3D.cu">().data();

			if (!nvrtc_modules[device_idx].module) {
				nvrtc::compile_nvrtc_kernel(
					nvrtc_modules[device_idx],
					toeplitz_load_3D_code,
					"toeplitz_load_3D",
					device_idx
				);
			}
			
			CUdevice cudevice;
			cuDeviceGet(&cudevice, device_idx);
			cudaSetDevice(device_idx);

			int totalThreads = NX * NY * NZ;
			int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

			void* args[] = { 
				(void*)&input, 
				(void*)&output, 
				(void*)&scratch, 
				(void*)&mult1,
				(void*)&mult2,
				(void*)&input_output_mult_type,
				(void*)&input_mult1_type, 
				(void*)&output_mult1_type, 
				(void*)&input_mult2_type, 
				(void*)&output_mult2_type, 
				(void*)&batch_in,
				(void*)&batch_out,
				(void*)&NX, (void*)&NY, (void*)&NZ,
				(void*)&accumulate
			};

			cuLaunchKernel(
				nvrtc_modules[device_idx].function,
				blocks, 1, 1,
				threads_per_block, 1, 1,
				0, 0,
				args, 0
			);

		}

		void launch_toeplitz_load_2D(
			const cuFloatComplex* input, 
			const cuFloatComplex* output, 
			cuFloatComplex* scratch, 
			const cuFloatComplex* mult1,
			const cuFloatComplex* mult2,
			int input_output_mult_type,
			int input_mult1_type,
			int output_mult1_type,
			int input_mult2_type,
			int output_mult2_type,
			int batch_in,
			int batch_out,
			int NX, int NY,
			bool accumulate,
			int device_idx,
			int threads_per_block = 256)
		{
			static std::array<hasty::nvrtc::NVRTC_ModFunc, (size_t)device_idx::MAX_CUDA_DEVICES> nvrtc_modules = {};

			static const char* toeplitz_load_2D_code = b::embed<"lib/fft/kernels/toeplitz_load_2D.cu">().data();

			if (!nvrtc_modules[device_idx].module) {
				nvrtc::compile_nvrtc_kernel(
					nvrtc_modules[device_idx],
					toeplitz_load_2D_code,
					"toeplitz_load_2D",
					device_idx
				);
			}

			CUdevice cudevice;
			cuDeviceGet(&cudevice, device_idx);
			cudaSetDevice(device_idx);

			int totalThreads = NX * NY;
			int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

			void* args[] = { 
				(void*)&input, 
				(void*)&output, 
				(void*)&scratch, 
				(void*)&mult1, 
				(void*)&mult2, 
				(void*)&input_output_mult_type,
				(void*)&input_mult1_type, 
				(void*)&output_mult1_type,
				(void*)&input_mult2_type, 
				(void*)&output_mult2_type, 
				(void*)&batch_in,
				(void*)&batch_out, 
				(void*)&NX, (void*)&NY,
				(void*)&accumulate
			};

			cuLaunchKernel(
				nvrtc_modules[device_idx].function,
				blocks, 1, 1,
				threads_per_block, 1, 1,
				0, 0,
				args, 0
			);

		}

		void perform_toeplitz_multiplication_cuda_2D(
			const hat::Tensor& 				input,
			hat::Tensor 					output,
			const hat::Tensor& 				kernel,
			optrefw<hat::Tensor> 			scratch,
			optcrefw<hat::Tensor> 			mult1,
			optcrefw<hat::Tensor> 			mult2,
			int input_output_mult_type = 	(int)ToeplitzMultType::NONE,
			int input_mult1_type = 			(int)ToeplitzMultType::MULT,
			int output_mult1_type = 		(int)ToeplitzMultType::MULT_CONJ,
			int input_mult2_type = 			(int)ToeplitzMultType::MULT,
			int output_mult2_type = 		(int)ToeplitzMultType::MULT_CONJ,
			int accumulate_type = 			(int)ToeplitzAccumulateType::NONE
		)
		{
			auto device = input.device();
			int nbatch = input.size(0);
			int dim = input.dim();
			int NX = input.size(dim - 1);
			int NY = input.size(dim - 2);
			torch_check(NY > 1 && NX > 1, "input dimensions must be positive");
			int device_idx = device.index();
			bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

			const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.data_ptr<hc10::complex<float>>());
			cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.data_ptr<hc10::complex<float>>());
			const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.data_ptr<hc10::complex<float>>());

			cuFloatComplex* scratch_ptr;
			hat::Tensor scratchmem;
			if (scratch.has_value()) {
				const hat::Tensor& scr = (*scratch).get();
				torch_check(scr.is_cuda(), "scratch was not a CUDA tensor");
				torch_check(scr.scalar_type() == hat::kComplexFloat, "scratch dtype must be complex float");
				torch_check(scr.sizes().equals({ 2 * NY, 2 * NX }), "scratch must have shape (2*NY, 2*NX)");
				torch_check(scr.device() == device, "scratch must be on the same device as input and output");
				torch_check(scr.is_contiguous(), "scratch must be contiguous");
				scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.data_ptr<hc10::complex<float>>());
			} else {
				scratchmem = hat::empty({ 2 * NY, 2 * NX }, input.options());
				scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem.data_ptr<hc10::complex<float>>());
			}

			const cuFloatComplex* mult1_ptr = nullptr;
			if (mult1.has_value()) {
				const hat::Tensor& m1 = (*mult1).get();
				torch_check(m1.is_cuda(), "mult1 was not a CUDA tensor");
				torch_check(m1.scalar_type() == hat::kComplexFloat, "mult1 dtype must be complex float");
				torch_check(m1.sizes().equals({ NY, NX }), "mult1 must have shape (NY, NX)");
				torch_check(m1.device() == device, "mult1 must be on the same device as input and output");
				torch_check(m1.is_contiguous(), "mult1 must be contiguous");
				mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.data_ptr<hc10::complex<float>>());
			}
			const cuFloatComplex* mult2_ptr = nullptr;
			if (mult2.has_value()) {
				const hat::Tensor& m2 = (*mult2).get();
				torch_check(m2.is_cuda(), "mult2 was not a CUDA tensor");
				torch_check(m2.scalar_type() == hat::kComplexFloat, "mult2 dtype must be complex float");
				torch_check(m2.sizes().equals({ NY, NX }), "mult2 must have shape (NY, NX)");
				torch_check(m2.device() == device, "mult2 must be on the same device as input and output");
				torch_check(m2.is_contiguous(), "mult2 must be contiguous");
				mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.data_ptr<hc10::complex<float>>());
			}

			launch_toeplitz_load_2D(
				in_ptr,
				out_ptr,
				scratch_ptr,
				mult1_ptr,
				mult2_ptr,
				input_output_mult_type,
				input_mult1_type,
				output_mult1_type,
				input_mult2_type,
				output_mult2_type,
				0,
				-1,
				NX, NY,
				accumulate,
				device_idx
			);

			// Create VkFFT plans and execute FFTs here
			VkFFT_Cache::VkFFT_Key key(device_idx);
			key.performConvolution = true;
			key.size[0] = 2 * NX;
			key.size[1] = 2 * NY;
			key.FFTdim = 2;
			key.performZeropadding[0] = true;
			key.performZeropadding[1] = true;
			key.fft_zeropad_left[0] = 0;
			key.fft_zeropad_left[1] = 0;
			key.fft_zeropad_right[0] = NX;
			key.fft_zeropad_right[1] = NY;

			VkFFTApplication& app = global_vkfft_cache[device_idx].get_or_create(key);

			static void* buffer_ptrs[1];
			buffer_ptrs[0] = scratch_ptr;
			static void* kernel_ptrs[1];
			kernel_ptrs[0] = const_cast<cuFloatComplex*>(kernel_ptr);

			VkFFTLaunchParams launchParams = {};
			launchParams.buffer = buffer_ptrs;
			launchParams.kernel = kernel_ptrs;

			for (int b = 0; b < nbatch; ++b) {
				VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
				if (res != VKFFT_SUCCESS) {
					throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
				}

				// For other accumulate types, adjust outputbatch as needed
				launch_toeplitz_load_2D(
					in_ptr,
					out_ptr,
					scratch_ptr,
					mult1_ptr,
					mult2_ptr,
					input_output_mult_type,
					input_mult1_type,
					output_mult1_type,
					input_mult2_type,
					output_mult2_type,
					(b < (nbatch - 1)) ? (b + 1) : -1,
					b,
					NX, NY,
					accumulate,
					device_idx
				);
			}
		}

		void perform_toeplitz_multiplication_cuda_3D(
			const hat::Tensor& 				input,
			hat::Tensor 					output,
			const hat::Tensor& 				kernel,
			optrefw<hat::Tensor> 			scratch,
			optcrefw<hat::Tensor> 			mult1,
			optcrefw<hat::Tensor> 			mult2,
			int input_output_mult_type = 	(int)ToeplitzMultType::NONE,
			int input_mult1_type = 			(int)ToeplitzMultType::MULT,
			int output_mult1_type = 		(int)ToeplitzMultType::MULT_CONJ,
			int input_mult2_type = 			(int)ToeplitzMultType::MULT,
			int output_mult2_type = 		(int)ToeplitzMultType::MULT_CONJ,
			int accumulate_type = 			(int)ToeplitzAccumulateType::NONE
		)
		{
			auto device = input.device();
			int nbatch = input.size(0);
			int dim = input.dim();
			int NX = input.size(dim - 1);
			int NY = input.size(dim - 2);
			int NZ = input.size(dim - 3);
			torch_check(NZ > 1 && NY > 1 && NX > 1, "input dimensions must be positive");
			int device_idx = device.index();
			bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

			const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.data_ptr<hc10::complex<float>>());
			cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.data_ptr<hc10::complex<float>>());
			const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.data_ptr<hc10::complex<float>>());

			cuFloatComplex* scratch_ptr;
			hat::Tensor scratchmem;
			if (scratch.has_value()) {
				const hat::Tensor& scr = (*scratch).get();
				torch_check(scr.is_cuda(), "scratch was not a CUDA tensor");
				torch_check(scr.scalar_type() == hat::kComplexFloat, "scratch dtype must be complex float");
				torch_check(scr.sizes().equals({ 2 * NZ, 2 * NY, 2 * NX }), "scratch must have shape (2*NZ, 2*NY, 2*NX)");
				torch_check(scr.device() == device, "scratch must be on the same device as input and output");
				torch_check(scr.is_contiguous(), "scratch must be contiguous");
				scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.data_ptr<hc10::complex<float>>());
			} else {
				scratchmem = hat::empty({ 2 * NZ, 2 * NY, 2 * NX }, input.options());
				scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem.data_ptr<hc10::complex<float>>());
			}

			const cuFloatComplex* mult1_ptr = nullptr;
			if (mult1.has_value()) {
				const hat::Tensor& m1 = (*mult1).get();
				torch_check(m1.is_cuda(), "mult1 was not a CUDA tensor");
				torch_check(m1.scalar_type() == hat::kComplexFloat, "mult1 dtype must be complex float");
				torch_check(m1.sizes().equals({ NZ, NY, NX }), "mult1 must have shape (NZ, NY, NX)");
				torch_check(m1.device() == device, "mult1 must be on the same device as input and output");
				torch_check(m1.is_contiguous(), "mult1 must be contiguous");
				mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.data_ptr<hc10::complex<float>>());
			}
			const cuFloatComplex* mult2_ptr = nullptr;
			if (mult2.has_value()) {
				const hat::Tensor& m2 = (*mult2).get();
				torch_check(m2.is_cuda(), "mult2 was not a CUDA tensor");
				torch_check(m2.scalar_type() == hat::kComplexFloat, "mult2 dtype must be complex float");
				torch_check(m2.sizes().equals({ NZ, NY, NX }), "mult2 must have shape (NZ, NY, NX)");
				torch_check(m2.device() == device, "mult2 must be on the same device as input and output");
				torch_check(m2.is_contiguous(), "mult2 must be contiguous");
				mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.data_ptr<hc10::complex<float>>());
			}

			launch_toeplitz_load_3D(
				in_ptr,
				out_ptr,
				scratch_ptr,
				mult1_ptr,
				mult2_ptr,
				input_output_mult_type,
				input_mult1_type,
				output_mult1_type,
				input_mult2_type,
				output_mult2_type,
				0,
				-1,
				NX, NY, NZ,
				accumulate,
				device_idx
			);

			// Create VkFFT plans and execute FFTs here
			VkFFT_Cache::VkFFT_Key key(device_idx);
			key.performConvolution = true;
			key.size[0] = 2 * NX;
			key.size[1] = 2 * NY;
			key.size[2] = 2 * NZ;
			key.FFTdim = 3;
			key.performZeropadding[0] = true;
			key.performZeropadding[1] = true;
			key.performZeropadding[2] = true;
			key.fft_zeropad_left[0] = 0;
			key.fft_zeropad_left[1] = 0;
			key.fft_zeropad_left[2] = 0;
			key.fft_zeropad_right[0] = NX;
			key.fft_zeropad_right[1] = NY;
			key.fft_zeropad_right[2] = NZ;

			VkFFTApplication& app = global_vkfft_cache[device_idx].get_or_create(key);
			
			static void* buffer_ptrs[1];
			buffer_ptrs[0] = scratch_ptr;
			static void* kernel_ptrs[1];
			kernel_ptrs[0] = const_cast<cuFloatComplex*>(kernel_ptr);

			VkFFTLaunchParams launchParams = {};
			launchParams.buffer = buffer_ptrs;
			launchParams.kernel = kernel_ptrs;

			for (int b = 0; b < nbatch; ++b) {
				VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
				if (res != VKFFT_SUCCESS) {
					throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
				}

				// For other accumulate types, adjust outputbatch as needed
				launch_toeplitz_load_3D(
					in_ptr,
					out_ptr,
					scratch_ptr,
					mult1_ptr,
					mult2_ptr,
					input_output_mult_type,
					input_mult1_type,
					output_mult1_type,
					input_mult2_type,
					output_mult2_type,
					(b < (nbatch - 1)) ? b+1 : -1,
					b,
					NX, NY, NZ,
					accumulate,
					device_idx
				);
			}
		}

	}
}

// Interface
namespace hasty {
	namespace fft {

		void transform_toeplitz_kernel(tensor<cuda_t, c64_t, 2>& kernel, bool clear_vkfft_plan) 
		{
			hat::Tensor& ker = kernel.get_tensor();
			
			int NX = ker.size(1);
			int NY = ker.size(0);
			torch_check(NY > 1 && NX > 1, "kernel dimensions must be positive");
			auto device = ker.device();

			VkFFT_Cache::VkFFT_Key key(device.index());
			key.size[0] = NX;
			key.size[1] = NY;
			key.FFTdim = 2;
			key.kernelConvolution = 1;
			
			{
				hat::cuda::CUDAGuard device_guard(ker.device());
				cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.data_ptr<hc10::complex<float>>());
				{
					cufftHandle plan;
					CUFFT_CHECK(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));
					cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
					CUDA_CHECK(cudaDeviceSynchronize());
					CUFFT_CHECK(cufftDestroy(plan));
				}
	
				hat::Tensor scratch = hat::empty_like(ker);
				cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.data_ptr<hc10::complex<float>>());

				VkFFTApplication& app = global_vkfft_cache[device.index()].get_or_create(key);
				VkFFTLaunchParams launchParams = {};
				static void* buffer_ptrs[1];
				buffer_ptrs[0] = scratch_ptr;
				static void* kernel_ptrs[1];
				kernel_ptrs[0] = ker_ptr;
				launchParams.buffer = buffer_ptrs;
				launchParams.kernel = kernel_ptrs;
				VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
				if (res != VKFFT_SUCCESS) {
					throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
				}
				CUDA_CHECK(cudaDeviceSynchronize());
			}

			if (clear_vkfft_plan) {
				global_vkfft_cache[device.index()].erase(key);
			}

		}

		void transform_toeplitz_kernel(tensor<cuda_t, c64_t, 3>& kernel, bool clear_vkfft_plan) 
		{
			hat::Tensor& ker = kernel.get_tensor();
			
			int NX = ker.size(2);
			int NY = ker.size(1);
			int NZ = ker.size(0);
			torch_check(NZ > 1 && NY > 1 && NX > 1, "kernel dimensions must be positive");
			auto device = ker.device();

			VkFFT_Cache::VkFFT_Key key(device.index());
			key.size[0] = NX;
			key.size[1] = NY;
			key.size[2] = NZ;
			key.FFTdim = 3;
			key.kernelConvolution = 1;
			
			{
				hat::cuda::CUDAGuard device_guard(ker.device());
				cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.data_ptr<hc10::complex<float>>());
				{
					cufftHandle plan;
					CUFFT_CHECK(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C));
					cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
					CUDA_CHECK(cudaDeviceSynchronize());
					CUFFT_CHECK(cufftDestroy(plan));
				}

				hat::Tensor scratch = hat::empty_like(ker);
				cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.data_ptr<hc10::complex<float>>());

				VkFFTApplication& app = global_vkfft_cache[device.index()].get_or_create(key);
				VkFFTLaunchParams launchParams = {};
				static void* buffer_ptrs[1];
				buffer_ptrs[0] = scratch_ptr;
				static void* kernel_ptrs[1];
				kernel_ptrs[0] = ker_ptr;
				launchParams.buffer = buffer_ptrs;
				launchParams.kernel = kernel_ptrs;
				VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
				if (res != VKFFT_SUCCESS) {
					throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
				}
				CUDA_CHECK(cudaDeviceSynchronize());
			}

			if (clear_vkfft_plan) {
				global_vkfft_cache[device.index()].erase(key);
			}
		}

		void toeplitz_multiplication(
			const tensor<cuda_t, c64_t, 3>&     input,
			tensor<cuda_t, c64_t, 3>&           output,
			const tensor<cuda_t, c64_t, 2>&     kernel,
			optrefw<tensor<cuda_t, c64_t, 2>>   scratch,
			optcrefw<tensor<cuda_t, c64_t, 2>>  mult1,
			optcrefw<tensor<cuda_t, c64_t, 2>>  mult2,
			ToeplitzMultType                    input_output_mult_type,
			ToeplitzMultType                    input_mult1_type,
			ToeplitzMultType                    output_mult1_type,
			ToeplitzMultType                    input_mult2_type,
			ToeplitzMultType                    output_mult2_type,
			ToeplitzAccumulateType              accumulate_type
		)
		{
			const hat::Tensor& inp = input.get_tensor();
			hat::Tensor& out = output.get_tensor();
			const hat::Tensor& ker = kernel.get_tensor();

			torch_check(inp.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
			torch_check(out.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
			torch_check(ker.scalar_type() == hat::kComplexFloat, "kernel dtype must be complex float");

			int NX = inp.size(2);
			int NY = inp.size(1);
			auto device = inp.device();
			torch_check(inp.sizes().equals(out.sizes()), "input and output must have the same size");
			torch_check(device == out.device(), "input and output must be on the same device");
			torch_check(device == ker.device(), "input and kernel must be on the same device");
			torch_check(ker.sizes().equals({ 2 * NY, 2 * NX }), "scratch must have shape (2*NY, 2*NX)");

			optrefw<hat::Tensor> scr = scratch.has_value() ? std::make_optional(std::ref((*scratch).get().get_tensor())) : std::nullopt;
			optcrefw<hat::Tensor> m1 = mult1.has_value() ? std::make_optional(std::cref((*mult1).get().get_tensor())) : std::nullopt;
			optcrefw<hat::Tensor> m2 = mult2.has_value() ? std::make_optional(std::cref((*mult2).get().get_tensor())) : std::nullopt;

			perform_toeplitz_multiplication_cuda_2D(
				inp,
				out,
				ker,
				std::move(scr),
				std::move(m1),
				std::move(m2),
				(int)input_output_mult_type,
				(int)input_mult1_type,
				(int)output_mult1_type,
				(int)input_mult2_type,
				(int)output_mult2_type,
				(int)accumulate_type
			);
		}

		void toeplitz_multiplication(
			const tensor<cuda_t, c64_t, 4>&     input,
			tensor<cuda_t, c64_t, 4>&           output,
			const tensor<cuda_t, c64_t, 3>&     kernel,
			optrefw<tensor<cuda_t, c64_t, 3>>   scratch,
			optcrefw<tensor<cuda_t, c64_t, 3>>  mult1,
			optcrefw<tensor<cuda_t, c64_t, 3>>  mult2,
			ToeplitzMultType                    input_output_mult_type,
			ToeplitzMultType                    input_mult1_type,
			ToeplitzMultType                    output_mult1_type,
			ToeplitzMultType                    input_mult2_type,
			ToeplitzMultType                    output_mult2_type,
			ToeplitzAccumulateType              accumulate_type
		) 
		{
			const hat::Tensor& inp = input.get_tensor();
			hat::Tensor& out = output.get_tensor();
			const hat::Tensor& ker = kernel.get_tensor();

			torch_check(inp.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
			torch_check(out.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
			torch_check(ker.scalar_type() == hat::kComplexFloat, "kernel dtype must be complex float");

			int NX = inp.size(3);
			int NY = inp.size(2);
			int NZ = inp.size(1);
			auto device = inp.device();
			torch_check(inp.sizes().equals(out.sizes()), "input and output must have the same size");
			torch_check(device == out.device(), "input and output must be on the same device");
			torch_check(device == ker.device(), "input and kernel must be on the same device");
			torch_check(ker.sizes().equals({ 2 * NZ, 2 * NY, 2 * NX }), "scratch must have shape (2*NZ, 2*NY, 2*NX)");

			optrefw<hat::Tensor> scr = scratch.has_value() ? std::make_optional(std::ref((*scratch).get().get_tensor())) : std::nullopt;
			optcrefw<hat::Tensor> m1 = mult1.has_value() ? std::make_optional(std::cref((*mult1).get().get_tensor())) : std::nullopt;
			optcrefw<hat::Tensor> m2 = mult2.has_value() ? std::make_optional(std::cref((*mult2).get().get_tensor())) : std::nullopt;

			perform_toeplitz_multiplication_cuda_3D(
				inp,
				out,
				ker,
				std::move(scr),
				std::move(m1),
				std::move(m2),
				(int)input_output_mult_type,
				(int)input_mult1_type,
				(int)output_mult1_type,
				(int)input_mult2_type,
				(int)output_mult2_type,
				(int)accumulate_type
			);
		}

	}
}


// OPS
namespace hasty {
	namespace fft {

		void ops_toeplitz_multiplication_cuda(
			hat::Tensor input,
			hat::Tensor output,
			hat::Tensor kernel,
			std::optional<hat::Tensor> scratch,
			std::optional<hat::Tensor> mult1,
			std::optional<hat::Tensor> mult2,
			int64_t input_output_mult_type, // ToeplitzMultType
			int64_t input_mult1_type, 	// ToeplitzMultType
			int64_t output_mult1_type, 	// ToeplitzMultType
			int64_t input_mult2_type, 	// ToeplitzMultType
			int64_t output_mult2_type, 	// ToeplitzMultType
			int64_t accumulate_type	// ToeplitzAccumulateType
		)
		{
			auto device = input.device();
			torch_check(input.is_cuda(), "input was not a CUDA tensor");
			torch_check(output.is_cuda(), "output was not a CUDA tensor");
			torch_check(kernel.is_cuda(), "kernel was not a CUDA tensor");
			torch_check(input.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
			torch_check(output.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
			torch_check(input.sizes().equals(output.sizes()), "input and output must have the same size");
			torch_check(device == output.device(), "input and output must be on the same device");
			torch_check(input.is_contiguous(), "input must be contiguous");
			torch_check(output.is_contiguous(), "output must be contiguous");
			int dim = input.dim();
			int nbatch = input.size(0);
			torch_check(dim == 3 || dim == 4, "input and output must be 3D or 4D (with batch)");

			optrefw<hat::Tensor> scratch_opt = scratch.has_value() ? std::make_optional(std::ref((*scratch))) : std::nullopt;
			optcrefw<hat::Tensor> mult1_opt = mult1.has_value() ? std::make_optional(std::cref((*mult1))) : std::nullopt;
			optcrefw<hat::Tensor> mult2_opt = mult2.has_value() ? std::make_optional(std::cref((*mult2))) : std::nullopt;

			if (dim == 4) {
				perform_toeplitz_multiplication_cuda_3D(
					input, output, kernel, 
					std::move(scratch_opt), 
					std::move(mult1_opt), std::move(mult2_opt),
					input_output_mult_type,
					input_mult1_type, output_mult1_type, 
					input_mult2_type, output_mult2_type, 
					accumulate_type
				);
			} else {
				perform_toeplitz_multiplication_cuda_2D(
					input, output, kernel, 
					std::move(scratch_opt), 
					std::move(mult1_opt), std::move(mult2_opt),
					input_output_mult_type,
					input_mult1_type, output_mult1_type, 
					input_mult2_type, output_mult2_type, 
					accumulate_type
				);
			}
		}

		void ops_toeplitz_multiplication(
			hat::Tensor input,
			hat::Tensor output,
			hat::Tensor kernel,
			std::optional<hat::Tensor> scratch,
			std::optional<hat::Tensor> mult1,
			std::optional<hat::Tensor> mult2,
			int64_t input_output_mult_type, // ToeplitzMultType
			int64_t input_mult1_type, 	// ToeplitzMultType
			int64_t output_mult1_type, 	// ToeplitzMultType
			int64_t input_mult2_type, 	// ToeplitzMultType
			int64_t output_mult2_type, 	// ToeplitzMultType
			int64_t accumulate_type	// ToeplitzAccumulateType
		)
		{
			{
				switch(input_output_mult_type) {
					case (int)ToeplitzMultType::NONE:
					case (int)ToeplitzMultType::MULT:
					case (int)ToeplitzMultType::MULT_CONJ:
						break;
					default:
						throw std::runtime_error("Invalid input_output_mult_type");
				}
				switch(input_mult1_type) {
					case (int)ToeplitzMultType::NONE:
					case (int)ToeplitzMultType::MULT:
					case (int)ToeplitzMultType::MULT_CONJ:
						break;
					default:
						throw std::runtime_error("Invalid input_mult_type");
				}
				switch(output_mult1_type) {
					case (int)ToeplitzMultType::NONE:
					case (int)ToeplitzMultType::MULT:
					case (int)ToeplitzMultType::MULT_CONJ:
						break;
					default:
						throw std::runtime_error("Invalid output_mult_type");
				}
				switch(input_mult2_type) {
					case (int)ToeplitzMultType::NONE:
					case (int)ToeplitzMultType::MULT:
					case (int)ToeplitzMultType::MULT_CONJ:
						break;
					default:
						throw std::runtime_error("Invalid input_mult2_type");
				}
				switch(output_mult2_type) {
					case (int)ToeplitzMultType::NONE:
					case (int)ToeplitzMultType::MULT:
					case (int)ToeplitzMultType::MULT_CONJ:
						break;
					default:
						throw std::runtime_error("Invalid output_mult2_type");
				}
				switch(accumulate_type) {
					case (int)ToeplitzAccumulateType::NONE:
					case (int)ToeplitzAccumulateType::ACCUMULATE:
						break;
					default:
						throw std::runtime_error("Invalid accumulate_type");
				}
			}

			if (input.device().is_cuda()) {
				ops_toeplitz_multiplication_cuda(
					input, output, kernel, 
					std::move(scratch), 
					std::move(mult1), std::move(mult2), 
					input_output_mult_type,
					input_mult1_type, output_mult1_type, 
					input_mult2_type, output_mult2_type, 
					accumulate_type
				);
			} else {
				throw std::runtime_error("toeplitz_multiplication: only CUDA tensors are supported currently");
			}

		}

	}
}


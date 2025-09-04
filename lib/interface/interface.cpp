#include "interface.hpp"
#include "interface_includes.hpp"

import util;
import tensor;
import hdf5;
import nufft;

import mri;
import trace;
import trace_cache;

namespace ffi {


	std::vector<at::Tensor> test_simple_invert() {
		c10::InferenceMode im_guard{};
		torch::NoGradGuard no_grad_guard;

		using namespace hasty;

		cache_dir = "/home/turbotage/Documents/hasty_cache/";

		/*
		std::vector<std::regex> matchers = {
			std::regex("^/Kdata/KData_E[01].*"),
			std::regex("^/Kdata/KX_E[01].*"),
			std::regex("^/Kdata/KY_E[01].*"),
			std::regex("^/Kdata/KZ_E[01].*"),
			std::regex("^/Kdata/KW_E[01].*")
		};
		*/
		
		std::vector<std::regex> matchers = {
			std::regex("^/Kdata/KData_E.*"),
			std::regex("^/Kdata/KX_E.*"),
			std::regex("^/Kdata/KY_E.*"),
			std::regex("^/Kdata/KZ_E.*"),
			std::regex("^/Kdata/KW_E.*")
		};

		std::cout << "Importing tensors" << std::endl;
		auto tset = import_tensors(
			"/home/turbotage/Documents/4DRecon/other_data/MRI_Raw.h5", matchers);

		auto shape_getter = []<size_t R>(const at::Tensor& ten) -> std::array<i64,R> 
		{
			if (ten.ndimension() != R) {
				throw std::runtime_error("Invalid number of dimensions");
			}
			std::array<i64,R> shape;
			for_sequence<R>([&](auto i) {
				shape[i] = ten.size(i);
			});
			return shape;
		};

		std::vector<at::Tensor> output_tensors;
		output_tensors.reserve(5);
		for (int e = 0; e < 5; ++e) {

			std::array<cache_tensor<f32_t,1>,3> coords;
			cache_tensor<f32_t,1> weights;

			cache_tensor<c64_t,2> kdata;

			std::cout << "Starting encode " << e << std::endl;

			at::Tensor temp = std::get<at::Tensor>(tset["/Kdata/KX_E" + std::to_string(e)]).flatten();
			temp *= (3.141592 / 160.0);
			coords[0] = cache_tensor<f32_t,1>(
				tensor<cpu_t,f32_t,1>(shape_getter.template operator()<1>(temp), temp),
				std::hash<std::string>{}("KX_E" + std::to_string(e))
			);
			tset.erase("/Kdata/KX_E" + std::to_string(e));

			temp = std::get<at::Tensor>(tset["/Kdata/KY_E" + std::to_string(e)]).flatten();
			temp *= (3.141592 / 160.0);
			coords[1] = cache_tensor<f32_t,1>(
				tensor<cpu_t,f32_t,1>(shape_getter.template operator()<1>(temp), temp),
				std::hash<std::string>{}("KY_E" + std::to_string(e))
			);
			tset.erase("/Kdata/KY_E" + std::to_string(e));

			temp = std::get<at::Tensor>(tset["/Kdata/KZ_E" + std::to_string(e)]).flatten();
			temp *= (3.141592 / 160.0);
			coords[2] = cache_tensor<f32_t,1>(
				tensor<cpu_t,f32_t,1>(shape_getter.template operator()<1>(temp), temp),
				std::hash<std::string>{}("KZ_E" + std::to_string(e))
			);
			tset.erase("/Kdata/KZ_E" + std::to_string(e));

			temp = std::get<at::Tensor>(tset["/Kdata/KW_E" + std::to_string(e)]).flatten();
			weights = cache_tensor<f32_t,1>(
				tensor<cpu_t,f32_t,1>(shape_getter.template operator()<1>(temp), temp),
				std::hash<std::string>{}("KW_E" + std::to_string(e))
			);
			tset.erase("/Kdata/KW_E" + std::to_string(e));

			std::vector<at::Tensor> kdata_tensors;
			kdata_tensors.reserve(48);
			for (int c = 0; true; ++c) {
				auto key = "/Kdata/KData_E" + std::to_string(e) + "_C" + std::to_string(c);
				if (tset.find(key) == tset.end()) {
					break;
				}
				temp = std::get<at::Tensor>(tset[key]).flatten();

				kdata_tensors.push_back(temp);
			
				tset.erase(key);
			}
			auto kdata_tensor = at::stack(kdata_tensors, 0);
			kdata_tensors.clear();

			kdata = cache_tensor<c64_t,2>(
				tensor<cpu_t,c64_t,2>(shape_getter.template operator()<2>(kdata_tensor), std::move(kdata_tensor)),
				std::hash<std::string>{}("KData_E" + std::to_string(e))
			);
			//kdata_tensor = at::empty({0}, at::kFloat);

			std::cout << "Starting nuffts" << std::endl;
			
			
			/*
			std::array<tensor<cuda_t,f64_t,1>,3> coords_gpu;
			for (int i = 0; i < 3; ++i) {
				coords_gpu[i] = move(coords[i].template get<cuda_t>(device_idx::CUDA0).template to<f64_t>());
			}
			*/
			
			
			std::array<tensor<cuda_t,f32_t,1>,3> coords_gpu;
			for (int i = 0; i < 3; ++i) {
				coords_gpu[i] = move(coords[i].template get<cuda_t>(device_idx::CUDA0));
			}

			auto input = weights.template get<cpu_t>().unsqueeze(0) * kdata.template get<cpu_t>();

			coords[0] = cache_tensor<f32_t,1>();
			coords[1] = cache_tensor<f32_t,1>();
			coords[2] = cache_tensor<f32_t,1>();
			weights = cache_tensor<f32_t,1>();
			kdata = cache_tensor<c64_t,2>();

			auto start = std::chrono::high_resolution_clock::now();

			auto output = nufft_backward_cpu_over_cuda(span<3>({320, 320, 320}), input, coords_gpu);        

			auto end = std::chrono::high_resolution_clock::now();

			std::cout << "Encode: " << e << " Time: " << 
						std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

			output_tensors.push_back(output.get_tensor());
		}

		return output_tensors;
	}


	at::Tensor test_offresonance_operator() {

		using namespace hasty;

		hasty::cache_dir = "/home/turbotage/Documents/hasty_cache/";

		int xres = 320;
		int yres = 320;
		int zres = 320;
		int ncoils = 24;
		int offresonance_n = 6;

		
		hasty::cache_tensor<c64_t, 3> phase_offset{
			hasty::make_rand_tensor<cpu_t,c64_t,3>(hasty::span<3>({xres, yres, zres})), 
			std::hash<std::string>{}("phase_offset")
		};
		// This is our stacked diagonals
		hasty::cache_tensor<c64_t, 4> smaps{
			hasty::make_rand_tensor<cpu_t,c64_t,4>(hasty::span<4>({ncoils, xres, yres, zres})),
			std::hash<std::string>{}("smaps")   
		};

		hasty::cache_tensor<c64_t,4> kernels;
		hasty::cache_tensor<c64_t,4> kerneldiags;
		{
			std::vector<hasty::tensor<cpu_t,c64_t,3>> kernelvec;
			std::vector<hasty::tensor<cpu_t,c64_t,3>> kerneldiagvec;
			for (int i = 0; i < offresonance_n; ++i) {
				// Toeplitz kernel

				kernelvec.push_back(
					hasty::make_rand_tensor<cpu_t,c64_t,3>(hasty::span<3>({2*xres, 2*yres, 2*zres}))
				);

				auto ratemap = hasty::make_rand_tensor<cpu_t,c64_t,3>(hasty::span<3>({xres, yres, zres}));

				// Ratemap diagonal * Phase offset diagonal
				kerneldiagvec.push_back(
					phase_offset.template get<cpu_t>() * ratemap
				);
			}

			kernels = hasty::cache_tensor(
				hasty::stack<0>(kernelvec),
				std::hash<std::string>{}("kernels")
			);

			kerneldiags = hasty::cache_tensor(
				hasty::stack<0>(kerneldiagvec),
				std::hash<std::string>{}("kerneldiag")
			);

		}

		auto input = hasty::make_rand_tensor<cuda_t,c64_t,3>(hasty::span<3>({xres, yres, zres}), device_idx::CUDA0);

		//sense_normal_image_offresonance_diagonal<cuda_t, c64_t, 3> sense(smaps, diagonal, kernels, ratemap_diagonals);
		NORMAL_IDT_T1_OP<cuda_t, c64_t, 3> normal_sense(
			std::move(kernels), 
			std::move(kerneldiags),
			std::move(smaps)
		);

		auto output = normal_sense(std::move(input));

		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < 100; ++i) {
			input += output;
			output = normal_sense(std::move(input));
		}

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> duration = end - start;
		std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

		return output.get_tensor();
	}

	at::Tensor test_whitten_offresonance_operator() {

		using namespace hasty;

		cache_dir = "/home/turbotage/Documents/hasty_cache/";

		int xres = 256;
		int yres = 256;
		int zres = 256;
		int ncoils = 24;
		int offresonance_n = 6;

		
		cache_tensor<c64_t, 3> phase_offset{
			make_rand_tensor<cpu_t,c64_t,3>(span<3>({xres, yres, zres})), 
			std::hash<std::string>{}("phase_offset")
		};
		// This is our stacked diagonals
		cache_tensor<c64_t, 4> smaps{
			make_rand_tensor<cpu_t,c64_t,4>(span<4>({ncoils, xres, yres, zres})),
			std::hash<std::string>{}("smaps")   
		};

		cache_tensor<c64_t,4> kernels;
		cache_tensor<c64_t,4> kerneldiags;
		{
			std::vector<tensor<cpu_t,c64_t,3>> kernelvec;
			std::vector<tensor<cpu_t,c64_t,3>> kerneldiagvec;
			for (int i = 0; i < offresonance_n; ++i) {
				// Toeplitz kernel

				kernelvec.push_back(
					make_rand_tensor<cpu_t,c64_t,3>(span<3>({2*xres, 2*yres, 2*zres}))
				);

				auto ratemap = make_rand_tensor<cpu_t,c64_t,3>(span<3>({xres, yres, zres}));
	
				// Ratemap diagonal * Phase offset diagonal
				kerneldiagvec.push_back(
					phase_offset.template get<cpu_t>() * ratemap
				);
			}

			kernels = cache_tensor(
				stack<0>(kernelvec),
				std::hash<std::string>{}("kernels")
			);

			kerneldiags = cache_tensor(
				stack<0>(kerneldiagvec),
				std::hash<std::string>{}("kerneldiag")
			);

		}

		cache_tensor<c64_t, 2> coilweights = cache_tensor(
			make_rand_tensor<cpu_t,c64_t,2>(span<2>({ncoils, ncoils})),
			std::hash<std::string>{}("coilweights")
		);

		auto input = make_rand_tensor<cuda_t,c64_t,3>(
			span<3>({xres, yres, zres}), device_idx::CUDA0
		);


		NORMAL_IDTW_T1_OP<cuda_t, c64_t, 3> normal_sense(
			std::move(kernels), 
			std::move(kerneldiags),
			std::move(smaps),
			std::move(coilweights),
			{1}

		);

		auto output = normal_sense(std::move(input));

		auto start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < 10; ++i) {
			input += output;
			output = normal_sense(std::move(input));
		}

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> duration = end - start;
		std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

		return output.get_tensor();
	}

	at::Tensor test_normal_operators() {
		using namespace hasty;

		cache_dir = "/home/turbotage/Documents/hasty_cache/";
		trace::module_cache_dir = cache_dir / "modules/";

		int xres = 256;
		int yres = 256;
		int zres = 256;
		int ncoils = 24;
		int offresonance_n = 6;

		device_idx didx = device_idx::CUDA0;
		
		cache_tensor<c64_t, 3> phase_offset{
			make_rand_tensor<cpu_t,c64_t,3>(span<3>({xres, yres, zres})), 
			std::hash<std::string>{}("phase_offset")
		};
		// This is our stacked diagonals
		cache_tensor<c64_t, 4> smaps{
			make_rand_tensor<cpu_t,c64_t,4>(span<4>({ncoils, xres, yres, zres})),
			std::hash<std::string>{}("smaps")   
		};

		cache_tensor<c64_t,3> kernel = cache_tensor(
			make_rand_tensor<cpu_t,c64_t,3>(span<3>({2*xres, 2*yres, 2*zres})),
			std::hash<std::string>{}("kernel")
		);

		cache_tensor<c64_t,4> kernels;
		cache_tensor<c64_t,4> kerneldiags;
		{
			std::vector<tensor<cpu_t,c64_t,3>> kernelvec;
			std::vector<tensor<cpu_t,c64_t,3>> kerneldiagvec;
			for (int i = 0; i < offresonance_n; ++i) {
				// Toeplitz kernel

				kernelvec.push_back(
					make_rand_tensor<cpu_t,c64_t,3>(span<3>({2*xres, 2*yres, 2*zres}))
				);

				auto ratemap = make_rand_tensor<cpu_t,c64_t,3>(span<3>({xres, yres, zres}));
	
				// Ratemap diagonal * Phase offset diagonal
				kerneldiagvec.push_back(
					phase_offset.template get<cpu_t>() * ratemap
				);
			}

			kernels = cache_tensor(
				stack<0>(kernelvec),
				std::hash<std::string>{}("kernels")
			);

			kerneldiags = cache_tensor(
				stack<0>(kerneldiagvec),
				std::hash<std::string>{}("kerneldiags")
			);

		}

		cache_tensor<c64_t, 2> coilweights = cache_tensor(
			make_rand_tensor<cpu_t,c64_t,2>(span<2>({ncoils, ncoils})),
			std::hash<std::string>{}("coilweights")
		);

		cache_tensor<c64_t, 3> image = cache_tensor(
			make_rand_tensor<cpu_t,c64_t,3>(span<3>({xres, yres, zres})),
			std::hash<std::string>{}("image")
		);

		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_T_T1_OP<cuda_t, c64_t, 3> normal_sense1(
				kernel.copy(),
				smaps.copy(),
				{2}
			);
			normal_sense1(image.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NT_T1_OP: " << duration.count() << " seconds" << std::endl;
		}
		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_T_T2_OP<cuda_t, c64_t, 3> normal_sense2(
				kernel.copy(),
				image.copy(),
				{2}
			);
			normal_sense2(smaps.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NT_T2_OP: " << duration.count() << " seconds" << std::endl;
		}

		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_IDT_T1_OP<cuda_t,c64_t,3> normal_sense1(
				kernels.copy(),
				kerneldiags.copy(),
				smaps.copy(),
				{2}
			);
			normal_sense1(image.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NIDT_T1_OP: " << duration.count() << " seconds" << std::endl;
		}
		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_IDT_T2_OP<cuda_t,c64_t,3> normal_sense2(
				kernels.copy(),
				kerneldiags.copy(),
				image.copy(),
				{2}
			);
			normal_sense2(smaps.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NIDT_T2_OP: " << duration.count() << " seconds" << std::endl;
		}

		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_IDTW_T1_OP<cuda_t,c64_t,3> normal_sense1(
				kernels.copy(),
				kerneldiags.copy(),
				smaps.copy(),
				coilweights.copy(),
				{2}
			);
			normal_sense1(image.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NIDTW_T1_OP: " << duration.count() << " seconds" << std::endl;
		}
		{
			auto start = std::chrono::high_resolution_clock::now();

			NORMAL_IDTW_T2_OP<cuda_t,c64_t,3> normal_sense2(
				kernels.copy(),
				kerneldiags.copy(),
				image.copy(),
				coilweights.copy(),
				{2}
			);
			normal_sense2(smaps.template get<cuda_t>(didx));

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> duration = end - start;
			std::cout << "NIDTW_T2_OP: " << duration.count() << " seconds" << std::endl;
		}

		return at::rand({1,1,1});
	}


	void test_prototype_stuff() {

		using namespace hasty;

		using T1 = trace::tensor_prototype<cpu_t, f32_t, 1>;
		using T2 = trace::tensor_prototype<cuda_t, c64_t, 2>;
		using T3 = trace::tensor_prototype_vector<cuda_t, f32_t, 3>;

		T1 t1("t1");
		T2 t2("t2");
		T3 t3("t3");

		using OUT_T1 = trace::tensor_prototype<cuda_t, f32_t,1>;
		using OUT_T2 = trace::tensor_prototype_vector<cpu_t, c64_t, 2>;

		auto builder = trace::trace_function_builder_factory<OUT_T1, OUT_T2>::make(
							"func",
							R"ts(
def somecomp(self, t):
	return torch.sin(t) + torch.sin(2*t)

FORWARD_ENTRYPOINT(self, t1, t2, t3):
	a = []
	for t in t3:
		temp = (t[:,:,0] + t2) + t1[:,None].to(t2.device)
		a.append(self.somecomp(temp).cpu())

	return (a[0][:,0].to(t2.device),a)
		)ts", t1, t2, t3);

		builder.compile();

		auto trace_func = builder.build_trace_function();

		auto runner = trace_func.get_runnable();

		std::cout << "Uncompiled:" << builder.uncompiled_str() << std::endl;

		std::cout << "Compiled:" << builder.compiled_str() << std::endl;

		//std::vector<hasty::tensor<hasty::empty_strong_typedef<hasty::cuda_>, hasty::strong_typedef<float, hasty::f32_>, 3>> &

		auto in1 = make_rand_tensor<cpu_t,f32_t,1>(span<1>({10}));
		auto in2 = make_rand_tensor<cuda_t,c64_t,2>(span<2>({10,10}), device_idx::CUDA0);
		auto in3 = std::vector<tensor<cuda_t,f32_t,3>>{
			make_rand_tensor<cuda_t,f32_t,3>(span<3>({10,10,10})),
			make_rand_tensor<cuda_t,f32_t,3>(span<3>({10,10,10})),
			make_rand_tensor<cuda_t,f32_t,3>(span<3>({10,10,10}))
		};

		std::cout << in1.str() << std::endl;
		std::cout << in2.str() << std::endl;
		std::cout << in3[0].str() << std::endl;
		std::cout << in3[1].str() << std::endl;
		std::cout << in3[2].str() << std::endl;

		auto a = runner.run(std::move(in1), std::move(in2), std::move(in3));

		std::cout << std::get<0>(a).str() << std::endl;
		std::cout << std::get<1>(a)[0].str() << std::endl;
		std::cout << std::get<1>(a)[1].str() << std::endl;
		std::cout << std::get<1>(a)[2].str() << std::endl;
	}


}



module;

#include "pch.hpp"

export module mri:forward;

import util;
import tensor;
import nufft;
import threading;

namespace hasty {


	/**
	@brief
	Performs the operation
	\f[ Ax \f] with the offresonance approximation
	\f[ e^{z_jt_i} \approx \sum_{l=1}^L b_{il}c_{lj} \f]
	
	The forward operation creates one 

	@param coords 
	@param image
	@param smaps
	@param bil
	@param cjl
	@param thread_pool

	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam TTC Precision used for NUFFT
	@tparam DIM Dimension.
	*/
	template<is_device D, is_fp_complex_tensor_type TT, is_fp_real_tensor_type TTC, size_t DIM>
	auto sense_forward_offresonance(
		std::array<cache_tensor<TTC,1>,DIM>& coords,
		cache_tensor<TT,DIM>& image,
		cache_tensor<TT,DIM+1>& smaps,
		cache_tensor<TT,2>& bil,
		cache_tensor<TT,DIM+1>& cjl,
		storage_thread_pool& thread_pool,
		bool free_nufft_plan = true) -> tensor<cpu_t,TT,2>
	{
		auto spatial_dim = image.shape();

		int64_t number_of_datapts = coords[0].get().template shape<0>();

		auto output = make_empty_tensor<cpu_t,TT,2>(span<2>{smaps.template shape<0>(), coords[0].template shape<0>()});

		auto run_lambda = [&output, &coords, &image, &smaps, &bil, &cjl, number_of_datapts](
			storage& store, i32 data_idx)
		{
			device_idx didx = store.get_ref_throw<device_idx>("device_idx");

			if (!store.exist("nufft_plan")) {
				nufft_opts<cuda_t,TTC,DIM> options;
				options.device_idx = i32(didx);
				options.nmodes = spatial_dim;
				options.ntransf = 1;
				options.method = nufft_method_cuda::DEFAULT;
				options.upsamp = nufft_upsamp_cuda::DEFAULT;
				if (std::is_same_v<TT, f32_t>) {
					options.tol = 1e-6;
				} else {
					options.tol = 1e-14;
				}

				std::array<tensor<cuda_t,TTC,1>,DIM> cuda_coords;
				for_sequence<DIM>([&cuda_coords, &coords, didx](auto i) {
					cuda_coords[i] = coords[i].get(didx);
				});

				auto plan = nufft_plan<cuda_t,TTC,DIM,nufft_type::FORWARD>::make(options);
				plan->setpts(cuda_coords);

				store.add<nufft_plan<cuda_t,TTC,DIM,nufft_type::FORWARD>>("nufft_plan", std::make_shared(plan));
			}

			if (!store.exist("image")) {
				auto cuda_image = image.get(device_idx::CPU).template to<cuda_t>(didx);
				store.add<tensor<cuda_t,TT,DIM>>("image", cuda_image);
			}

			if (!store.exist("bil")) {
				auto cubil = bil.get(device_idx::CPU).template to<cuda_t>(didx);
				store.add<tensor<cuda_t,TT,DIM>>("bil", cubil);
			}

			if (!store.exist("cjl")) {
				auto cucjl = cjl.get(device_idx::CPU).template to<cuda_t>(didx);
				store.add<tensor<cuda_t,TT,DIM+1>>("cjl", cucjl);
			}

			auto smap = smaps.get(device_idx::CPU)[data_idx,Ellipsis{}].template to<cuda_t>(didx);
			smap *= store.get_ref_throw<tensor<cuda_t,TT,DIM>>("image");
			
			auto temp_img = make_empty_tensor_like(smap);
			
			auto output_slice = make_zero_tensor<cuda_t,TT,2>(span<2>({1, number_of_datapts}));
			auto temp_output = make_empty_tensor<cuda_t,complex_t<TTC>,2>(span<2>({1, number_of_datapts}));

			auto& nufft_plan = store.get_ref_throw<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan");
			auto& cuda_bil = store.get_ref_throw<tensor<cuda_t,TT,2>>("bil");
			auto& cuda_cjl = store.get_ref_throw<tensor<cuda_t,TT,DIM+1>>("cjl");


			for (int i = 0; i < interp_steps; ++i) {
				
				temp_img = (smap * cuda_cjl[i, Ellipsis{}]).unsqueeze(0);

				hasty::synchronize(didx);
				if constexpr(std::is_same_v<TT, complex_t<TTC>>) {
					nufft_plan->execute(temp_img, temp_output);
					output_slice += temp_output * cuda_bil[i,Ellipsis{}];
				} else {
					nufft_plan->execute(temp_img.template to<complex_t<TTC>>(), temp_output);
					output_slice += temp_output.template to<TT>() * cuda_bil[i,Ellipsis{}];
				}
				hasty::synchronize(didx);

			}

			output[data_idx, Ellipsis{}] = output_slice.template to<cpu_t>(didx);
		};

		std::vector<std::future<void>> futures;
		futures.reserve(smaps.shape<0>());

		for (int i = 0; i < smaps.shape<0>(); i++) {
			auto runner = [i, &run_lambda](storage& store) {
				run_lambda(store, i);
			};
			auto fut = thread_pool.enqueue(run_lambda, i);
			futures.push_back(std::move(fut));
		}

		for (auto& fut : futures) {
			util::future_catcher(fut);
		}

		const auto& storages = thread_pool.get_storages();
		for (auto& store : storages) {
			store.clear("image");
			store.clear("bil");
			store.clear("cjl");
			if (free_nufft_plan) {
				store.clear("nufft_plan");
			}
		}

	}

}


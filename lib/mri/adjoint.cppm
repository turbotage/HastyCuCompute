module;

#include "pch.hpp"

export module forward;

import util;
import tensor;

namespace hasty {

	/**
	@brief
	Performs the operation
	\f[ A^Hb \f] with the offresonance approximation
	\f[ e^{z_jt_i} \approx \sum_{l=1}^L b_{il}c_{lj} \f]
	
	@param coords 
	@param kdata
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
	tensor<cpu_t,TT,DIM+1> sense_adjoint_offresonance(
		std::array<cache_tensor<TTC,1>,DIM>& coords, 
		cache_tensor<TT,2>& kdata,
		cache_tensor<TT,DIM+1>& smaps, 
		cache_tensor<TT,2>& bil,
		cache_tensor<TT,DIM+1>& cjl,
		storage_thread_pool& thread_pool) 
	{
		auto spatial_dim = span<DIM>(smaps.shape(), 1).to_arr();

		auto output = make_empty_like(smaps);

		int L = bil.shape<0>();

		auto run_lambda = [&output, &coords, &kdata, &smaps, &bil, &cjl](storage& store, i32 data_idx) 
		{
			device_idx didx = store.get_ref_throw<device_idx>("device_idx");

			if (!store.exists("nufft_plan")) {
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
				for_sequence<DIM>([&cuda_coords](auto i) {
					cuda_coords[i] = coords.get(didx);
				});

				hasty::synchronize(didx);
				auto plan = nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>::make(options);
				plan->setpts(cuda_coords);
				hasty::synchronize(didx);

				store.add<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan", std::make_shared(std::move(plan)));
			}

			if (!store.exists("bil")) {
				auto cubil = bil.get(device_idx::CPU).template to<cuda_t>(didx);
				store.add<tensor<cuda_t,TT,DIM>>("bil", cubil);
			}

			if (!store.exists("cjl")) {
				auto cucjl = cjl.get(device_idx::CPU).template to<cuda_t>(didx);
				store.add<tensor<cuda_t,TT,DIM+1>>("cjl", cucjl);
			}

			auto mul = smaps.get(device_idx::CPU)[data_idx,Ellipsis{}].template to<cuda_t>(didx).conj();

			auto cuda_nufft_output = make_empty_tensor<cuda_t,complex_t<TTC>,DIM>(spatial_dim);
			auto cuda_output = make_zeros_tensor_like(cuda_nufft_output);

			auto& nufft_plan = store.get_ref_throw<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan");
			auto& cuda_bil = store.get_ref_throw<tensor<cuda_t,TT,2>>("bil");
			auto& cuda_cjl = store.get_ref_throw<tensor<cuda_t,TT,DIM+1>>("cjl");

			if constexpr(std::is_same_v<TT,complex_t<TTC>>) {
				auto cuda_kdata = kdata[data_idx, Ellipsis{}].unsqueeze(0).template to<cuda_t>(didx);
				auto temp = make_empty_tensor_like(cuda_kdata);

				for (int i = 0; i < L; ++i) {
					temp = cuda_kdata * cuda_bil[i, Ellipsis{}]

					hasty::synchronize(didx);
					nufft_plan->execute(temp, cuda_nufft_output);
					hasty::synchronize(didx);

					cuda_nufft_output *= cuda_cjl[i, Ellipsis{}];
					cuda_output += cuda_nufft_output;
				}

				output[data_idx, Ellipsis{}] = cuda_output.template to<cpu_t>(didx);

			} else {
				auto cuda_kdata = input[cuda_kdata, Ellipsis{}].unsqueeze(0).template to<cuda_t>(
					coords[0].get_device_idx()).template to<complex_t<TTC>>();
				auto temp = make_empty_tensor_like(cuda_kdata);

				for (int i = 0; i < L; ++i) {
					temp = cuda_kdata * cuda_bil[i, Ellipsis{}].conj();

					hasty::synchronize(didx);
					nufft_plan->execute(temp, cuda_nufft_output);
					hasty::synchronize(didx);

					cuda_nufft_output *= cuda_cjl[i, Ellipsis{}].conj();
					cuda_output += cuda_nufft_output;
				}

				output[data_idx, Ellipsis{}] = cuda_output.template to<TT>().template to<cpu_t>(didx);

			}
			
		};

		
		std::vector<std::future<void>> futures;
		futures.reserve(smaps.shape<0>());

		for (int i = 0; i < smaps.shape<0>(); ++i) {
			auto runner = [i, &run_lambda](storage& store) {
				run_lambda(store, i);
			};

			auto fut = thread_pool.enqueue(runner, {});
			futures.push_back(std::move(fut));
		}
		
		for (auto& fut : futures) {
			util::future_catcher(fut);
		}

	}


}

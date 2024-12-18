module;

#include "pch.hpp"

export module forward;

import util;
import tensor;

namespace hasty {

	template<is_device D, is_fp_complex_tensor_type TT, is_fp_real_tensor_type TTC, size_t DIM>
	tensor<cpu_t,TT,DIM+1> sense_adjoint_offresonance(
		std::array<cache_tensor<TTC,1>,DIM>& coords, 
		cache_tensor<TT,2>& kdata,
		cache_tensor<TT,DIM+1>& smaps, 
		tensor<cpu_t,TT,DIM>& offrensonance, 
		std::vector<std::pair<
			base_t<real_t<TT>>,
			base_t<TT>
		>>&& interpt_interpcoeff,
		storage_thread_pool& thread_pool) 
	{
		auto spatial_dim = span<DIM>(smaps.shape(), 1).to_arr();

		auto offres = make_zeros_like(offrensonance);

		for (i32 i = 0; i < interp_steps; ++i) {
			auto temp = offrensonance;
			temp *= interpt_interpcoeff[i].first;
			temp.exp_();
			temp *= std::conj(interpt_interpcoeff[i].second);

			offres += temp;
		}

		auto output = make_empty_like(smaps);

		auto run_lambda = [&output, &coords, &kdata, &smaps, &offres](storage& store, i32 data_idx) 
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

				store.add<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan", std::make_shared(nufft_plan));
			}

			auto mul = smaps.get(device_idx::CPU)[data_idx,Ellipsis{}].template to<cuda_t>(didx).conj();
			mul *= offres.get(device_idx::CPU).template to<cuda_t>(didx);

			auto cuda_nufft_output = make_empty_tensor<cuda_t,complex_t<TTC>,DIM>(spatial_dim);

			auto& nufft_plan = store.get_ref_throw<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan");

			if constexpr(std::is_same_v<TT,complex_t<TTC>>) {
				auto cuda_kdata = kdata[data_idx, Ellipsis{}].unsqueeze(0).template to<cuda_t>(didx);

				hasty::synchronize(didx);
				nufft_plan->execute(cuda_kdata, cuda_nufft_output);
				hasty::synchronize(didx);
				cuda_nufft_output *= mul;

				output[data_idx, Ellipsis{}] = cuda_nufft_output.template to<cpu_t>(didx);
			} else {
				auto cuda_kdata = input[cuda_kdata, Ellipsis{}].unsqueeze(0).template to<cuda_t>(
					coords[0].get_device_idx()).template to<complex_t<TTC>>();

				hasty::synchronize(didx);
				nufft_plan->execute(cuda_kdata, cuda_nufft_output);
				hasty::synchronize(didx);
				cuda_nufft_output *= mul;

				output[data_idx, Ellipsis{}] = cuda_nufft_output.template to<TT>().template to<cpu_t>(didx);
			}
			
		};

		
		std::vector<std::future<void>> futures;
		futures.reserve(smaps.shape<0>());

		for (int i = 0; i < smaps.shape<0>(); ++i) {
			auto runner = [i, &run_lambda](storage& store) {
				run_lambda(store, i);
			};

			auto fut = thread_pool.enqueue(runner, {});
		}
		
		for (auto& fut : futures) {
			util::future_catcher(fut);
		}

	}


}

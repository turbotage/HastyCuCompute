module;

#include "pch.hpp"

export module forward;

import util;
import tensor;

namespace hasty {

	template<is_deice D, is_fp_complex_tensor_type TT, is_fp_real_tensor_type TTC, size_t DIM>
	tensor<cpu_t,TT,2> sense_forward_offresonance(
		std::array<cache_tensor<TTC,1>,DIM>& coords,
		cache_tensor<TT,DIM>& image,
		cache_tensor<TT,DIM+1>& smaps,
		tensor<cpu_t,TT,DIM>& offresonance,
		std::vector<std::pair<
			base_t<real_t<TT>>,
			base_t<TT>
		>>&& interpt_interpcoeff,
		storage_thread_pool& thread_pool)
	{
		auto spatial_dim = image.shape();

		int64_t number_of_datapts = coords[0].get().shape<0>();

		auto output = make_empty_tensor<cpu_t,TT,2>(span<2>{smaps.shape<0>(), coords[0].shape<0>()});

		auto run_lambda = [&output, &coords, &image, &smaps, &offresonance, &interpt_interpcoeff, number_of_datapts](
			storage& store, i32 data_idx)
		{
			device_idx didx = store.get_ref_throw<device_idx>("device_idx");

			i32 interp_steps = interpt_interpcoeff.size();

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
					cuda_coords[i] = coords[i].get(didx);
				});

				auto plan = nufft_plan<cuda_t,TTC,DIM,nufft_type::FORWARD>::make(options);
				plan->setpts(cuda_coords);

				store.add<nufft_plan<cuda_t,TTC,DIM,nufft_type::FORWARD>>("nufft_plan", std::make_shared(plan));
			}

			auto smap = smaps.get(device_idx::CPU)[data_idx,Ellipsis{}].template to<cuda_t>(didx);
			auto offres = offresonance.get(device_idx::CPU).template to<cuda_t>(didx);
			auto input = image.get(device_idx::CPU).template to<cuda_t>(didx);
			
			auto output_slice = make_zero_tensor<cuda_t,TT,2>(span<2>{1, number_of_datapts});
			auto temp_output = make_empty_tensor<cuda_t,complex_t<TTC>,2>(span<2>{1, number_of_datapts});

			auto& nufft_plan = store.get_ref_throw<nufft_plan<cuda_t,TTC,DIM,nufft_type::BACKWARD>>("nufft_plan");

			for (int i = 0; i < interp_steps; ++i) {
				auto temp = offres;
				temp *= -interpt_interpcoeff[i].first;
				temp.exp_();
				temp *= interpt_interpcoeff[i].second;
				temp *= smap;
				temp *= input;
				
				hasty::synchronize(didx);
				if constexpr(std::is_same_v<TT, complex_t<TTC>>) {
					nufft_plan->execute(temp.unsqueeze(0), temp_output);
					output_slice += temp_output;
				} else {
					nufft_plan->execute(temp.unsqueeze(0).template to<complex_t<TTC>>(), temp_output);
					output_slice += temp_output.template to<TT>();
				}
				hasty::synchronize(didx);

			}

			output[data_idx, Ellipsis{}] = output_slice.template to<cpu_t>(didx);
		};

	}

}


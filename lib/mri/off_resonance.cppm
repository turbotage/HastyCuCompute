module;

#include <stdexcept>
#include <type_traits>
export module mri:off_resonance;

import tensor;
import trace;
import util;

namespace hasty {

	export template<is_device D, size_t R>
	requires (R >= 2) && (R <= 3)
	struct off_res_interpolators {
		tensor<D, c64_t, 2> B;
		tensor<D, c64_t, R+1> Ct;
	};

	export template<is_device D, size_t R>
	requires (R >= 2) && (R <= 3)
	off_res_interpolators<D,R> make_offresonance_interpolators(
		const tensor<D,c64_t,R>& fieldmap,
		const tensor<D,f32_t,1>& times,
		i32 L,
		i32 nbin,
		bool b0_only = false,
		bool autocorrelate = false)
	{
		if (autocorrelate && (method != off_res_method::TimeSegmentation)) {
			throw std::invalid_argument("Autocorrelation can only be used with TimeSegmentation");
		}

		bool if_trace_exists = false;
		if (if_trace_exists) {
			throw std::runtime_error("Trace with this signature already exists!");
		} else {
			using FIELDMAP_PROTO_T = trace::tensor_prototype<D,c64_t,R>;
			using TIMES_PROTO_T = trace::tensor_prototype<D,f32_t,1>;
			using L_PROTO_T = trace::tensor_prototype<D,i32_t,0>;
			using NBIN_PROTO_T = trace::tensor_prototype<D,i32_t,0>;
			using B0_ONLY_PROTO_T = trace::tensor_prototype<D,b8_t,0>;
			using AUTOCORR_PROTO_T = trace::tensor_prototype<D,b8_t,0>;
	
			using B_PROTO_T = trace::tensor_prototype<D,c64_t,2>;
			using CT_PROTO_T = trace::tensor_prototype<D,c64_t,R+1>;

			using TRACE_FUNC_T = trace::trace_function<
				std::tuple<B_PROTO_T, CT_PROTO_T>,
				std::tuple<
					FIELDMAP_PROTO_T, 
					TIMES_PROTO_T, 
					L_PROTO_T, 
					NBIN_PROTO_T,
					B0_ONLY_PROTO_T,
					AUTOCORR_PROTO_T
				>
			>;

			FIELDMAP_PROTO_T fieldmap_proto("fieldmap");
			TIMES_PROTO_T times_proto("times");
			L_PROTO_T l_proto("L");
			NBIN_PROTO_T nbin_proto("nbin");
			B0_ONLY_PROTO_T b0_only_proto("b0_only");
			AUTOCORR_PROTO_T autocorr_proto("autocorr");
			
			
			TRACE_FUNC_T runner = trace::trace_function_factory<B_PROTO_T, CT_PROTO_T>::make(
				"make_offresonance_interpolators", 
				fieldmap_proto, times_proto, l_proto, nbin_proto, b0_only_proto, autocorr_proto
			);

			runner.add_lines(R"ts(
FORWARD_ENTRYPOINT(self, fieldmap, times, L, nbin, b0_only, autocorr):
	device = fieldmap.device
	dtype = fieldmap.dtype
	shape = fieldmap.shape
	fmap_flat = fieldmap.flatten()

	if b0_only:
		fmap_data = fmap_flat.imag
	else:
		fmap_data = fmap_flat

	if not b0_only:
		re = fmap_data.real
		im = fmap_data.imag

		re_min, re_max = re.min(), re.max()
		im_min, im_max = im.min(), im.max()
		re_idx = ((re - re_min)/(re_max - re_min + 1e-12)*(nbins-1)).long()
		im_idx = ((im - im_min)/(im_max - im_min + 1e-12)*(nbins-1)).long()
		hist = torch.zeros(nbins, nbins, device=device, dtype=dtype)
		hist.index_put_((re_idx, im_idx), torch.ones_like(re_idx, dtype=dtype), accumulate=True)
	else:
		vmin, vmax = fmap_data.min(), fmap_data.max()
		idx = ((fmap_data - vmin) / (vmax - vmin + 1e-12) * (nbins - 1)).long()
		hist = torch.zeros(nbins, device=device, dtype=dtype)
		hist.index_put_(idx, torch.ones_like(idx, dtype=dtype), accumulate=True)

	if autocorr:
		if b0_only:
			hist = hist.view(1,1,-1)
			hist = torch.nn.functional.conv1d(hist, hist.flip(-1), padding=nbins-1).squeeze()
		else:
			hist = hist.view(1,1,nbins,nbins)
			hist = torch.nn.functional.conv2d(hist, hist.flip(-2,-1), padding=nbins-1).squeeze()

	tl = torch.linspace(T.min(), T.max(), L, dtype=torch.float64)

	if b0_only:
		zk = torch.linspace(vmin, vmax, hist.numel(), dtype=torch.float64, device=device)
		w = hist.sqrt()
		ch = torch.exp(-1j*tl.view(L,1)*zk.view(1,-1))
		B = torch.linalg.pinv(w*ch.T) @ (w.unsqueeze(0) * torch.exp(-1j*zk.view(-1,1)*T.view(1,-1)))
		B = B.to(dtype)
	else:
		zk_re = torch.linspace(re_min, re_max, nbins, device=device, type=torch.float64)
		zk_im = torch.linspace(im_min, im_max, nbins, device=device, type=torch.float64)
		zk1, zk2 = torch.meshgrid(zk_re, zk_im, indexing='ij')
		zk_flat = (zk1 + 1j*zk2).flatten()
		w = hist.flatten().sqrt()
		ch = torch.exp(-1j * tl.view(L,1) * zk_flat.view(1,-1))
		B = torch.linalg.pinv(w*ch.T) @ (w.unsqueeze(0) * torch.exp(-1j*zk_flat.view(-1,1)*T.view(1,-1)))
		B = B.to(dtype)

	Ct = torch.exp(-1j*fieldmap.unsqueeze(0)*tl.view(L,*([1]*len(shape)))

	return B, Ct
)ts");

			runner.compile();

		}



		return {std::move(fieldmap_proto), std::move(times_proto), std::move(l_proto), std::move(nbin_proto), std::move(autocorr_proto)};
	}

};



	}

};




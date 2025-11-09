module;

#include "pch.hpp"

export module mri:misc;

import util;
import script;
import tensor;

namespace hasty {

export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
auto magnitude_reg_gradient_stepper() 
	-> std::function<tensor<D,TT,DIM>(const tensor<D,TT,DIM>&, const tensor<D,b8_t,DIM>&, const tensor<D,TT,DIM>&)>
{
	script::tensor_prototype<D,TT,DIM> xproto("x");
	script::tensor_prototype<D,b8_t,DIM> maskproto("mask");
	script::tensor_prototype<D,TT,DIM> xmeanproto("xmean");
	script::tensor_prototype<D,TT,DIM> output("output");

	auto builder = script::compiled_script_builder<decltype(output)>(
		"magnitude_reg_gradient_stepper",
		std::format(R"ts(
FORWARD_ENTRYPOINT(self, x, mask, xmean):
	output = torch.zeros_like(x)
	mx = x[mask]
	mmx = xmean[mask]

	s = mx.abs().square()
	ms = mmx.abs().square()

	g = mx * (1 - (ms / s))

	output[mask] = g

	return output	
	)ts"),
		xproto, maskproto, xmeanproto
	);

	builder.compile();
	auto runner = builder.decay_to_runnable_script();

	using INP1 = const tensor<D,TT,DIM>&;
	using INP2 = const tensor<D,b8_t,DIM>&;
	using INP3 = const tensor<D,TT,DIM>&;
	using OUT = tensor<D,TT,DIM>;

	std::function<OUT(INP1,INP2, INP3)> runner_func = [run = std::move(runner)](
		const tensor<D,TT,DIM>& x, 
		const tensor<D,b8_t,DIM>& mask, 
		const tensor<D,TT,DIM>& xmean) 
	{
		return run.run(x, mask, xmean);
	};

	return runner_func;
}

}



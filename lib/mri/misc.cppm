module;

#include "pch.hpp"

export module misc;

namespace hasty {

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	auto magnitude_reg_gradient_stepper() 
		-> std::function<tensor<D,TT,DIM>(const tensor<D,TT,DIM>&, const tensor<D,b8_t,DIM>&, const tensor<D,TT,DIM>&)>
	{
		trace::tensor_prototype<D,TT,DIM> xproto("x");
		trace::tensor_prototype<D,b8_t,DIM> maskproto("mask");
		trace::tensor_prototype<D,TT,DIM> xmeanproto("xmean");
		trace::tensor_prototype<D,TT,DIM> output("output");

		auto runner = trace::trace_function_factory<decltype(output)>::make(
						"magnitude_reg_gradient_stepper", xproto, maskproto, xmeanproto);

		runner.add_lines(std::format(R"ts(
	with torch.inference_mode():
		output = torch.zeros_like(x)

		mx = x[mask]
		mmx = xmean[mask]

		s = mx.abs().square()
		ms = mmx.abs().square()

		g = mx * (1 - (ms / s))

		output[mask] = g

		return output
)ts", 2));

		runner.compile();

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



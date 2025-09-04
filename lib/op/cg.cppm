module;

#include "pch.hpp"

export module cg;

//import pch;

import util;
import op;
import tensor;
import trace;

namespace hasty {

	export template<is_hom_square_tensor_operator Op, is_hom_square_tensor_operator PrecondOp>
	requires    std::same_as<typename Op::device_type_t, typename PrecondOp::device_type_t> &&
				std::same_as<typename Op::output_tensor_type_t, typename PrecondOp::input_tensor_type_t> &&
				std::same_as<typename Op::output_tensor_rank_t, typename PrecondOp::input_tensor_rank_t>
	auto conjugate_gradient(sptr<Op> A, sptr<PrecondOp> P, i32 max_inner_iter, i32 max_outer_iter, double tol = 1e-6) 
	{
		using D = typename Op::device_type_t;
		using TT = typename Op::input_tensor_type_t;
		constexpr size_t R = Op::input_rank_t();

		trace::tensor_prototype<D,TT,R> x("x");
		trace::tensor_prototype<D,TT,R> p("p");
		trace::tensor_prototype<D,TT,R> Ap("Ap");
		trace::tensor_prototype<D,TT,R> r("r");
		trace::tensor_prototype<D,TT,R> z("z");
		trace::tensor_prototype<D,real_t<TT>,0> rzold("rzold");
		trace::tensor_prototype<D,b8_t,0> restart("restart");

		auto cg1_builder = trace::trace_function_builder_factory<decltype(x), decltype(r)>::make(
						"cg_step1", 
						std::format(R"ts(
FORWARD_ENTRYPOINT(self, x, p, Ap, r):
	pAp = torch.real(torch.vdot(p.flatten(),Ap.flatten()))
	alpha = rzold / pAp
	x += p * alpha
	r -= Ap * alpha
	return (x, r)
		)ts"), x,p,Ap,r);

		cg1_builder.compile();
		auto cg1 = cg1_builder.build_trace_function();

		auto cg2_builder = trace::trace_function_builder_factory<decltype(p), decltype(rzold)>::make(
						"cg_step2", 
						std::format(R"ts(
FORWARD_ENTRYPOINT(self, z, r, p, rzold, restart):
	rznew = torch.real(torch.vdot(r.flatten(),z.flatten()))
	if restart.item():
		p = z
	else:
		p = z + p * (rznew / rzold)
	return (p, rznew)
)ts"), z,r,p,rzold,restart);

		cg2_builder.compile();
		auto cg2 = cg2_builder.build_trace_function();


		auto cgl = [cg1=std::move(cg1), cg2=std::move(cg2), A=std::move(A), P=std::move(P), 
					max_inner_iter, max_outer_iter, tol]
					(tensor<D,TT,R>& x, const tensor<D,TT,R>& b, 
					std::optional<std::function<bool(tensor<D,TT,R>&)>> should_restart_callback = std::nullopt,
					std::optional<std::function<void(tensor<D,TT,R>&)>> after_restart_callback = std::nullopt)
		{

			auto r = b - A(x);
			auto z = P(r);
			auto p = z.clone();

			auto rzold = vdot(r, z).real();
			double resid = std::sqrt(rzold.item());

			for (i32 outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {

				for (i32 inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {

					if (resid < tol) {
						break;
					}

					auto Ap = A(p);

					std::tie(x,r) = cg1.run(std::move(x), p, std::move(Ap), std::move(r));

					z = P(r);

					bool should_restart = false;
					if (should_restart_callback.has_value()) {
						should_restart = should_restart_callback.value()(x);
					}

					std::tie(p,rzold) = cg2.run(std::move(z), r, std::move(p), std::move(rzold), tensor_factory<D,b8_t,0>::make_scalar(should_restart));

					resid = std::sqrt(rzold.item());

					if (should_restart) {
						break;
					}
				}

				if (after_restart_callback.has_value()) {
					after_restart_callback.value()(x);
				}

			}

			return x;
		};

		return cgl;
	}


	export template<is_hom_square_tensor_operator Op>
	auto conjugate_gradient(sptr<Op> A, i32 max_iter = 0, double tol = 1e-6) 
	{
		using D = typename Op::device_type_t;
		using TT = typename Op::input_tensor_type_t;
		constexpr size_t R = Op::input_rank_t();


		trace::tensor_prototype<D,TT,R> x("x");
		trace::tensor_prototype<D,TT,R> p("p");
		trace::tensor_prototype<D,TT,R> Ap("Ap");
		trace::tensor_prototype<D,TT,R> r("r");
		trace::tensor_prototype<D,real_t<TT>,0> rzold("rzold");

		auto cg_builder = trace::trace_function_builder_factory<decltype(x), decltype(r), decltype(p), decltype(rzold)>::make(
					"cg_step", 
					std::format(R"ts(
FORWARD_ENTRYPOINT(self, x, p, Ap, r):
	pAp = torch.real(torch.vdot(p.flatten(),Ap.flatten()))
	alpha = rzold / pAp
	x += p * alpha
	r -= Ap * alpha

	rznew = torch.real(torch.vdot(r.flatten(),r.flatten()))
	p = r + p * (rznew / rzold)
	
	return x, r, p, rznew
)ts"), x,p,Ap,r);
		cg_builder.compile();
		auto cg = cg_builder.build_trace_function();

		auto cgl = [cg = std::move(cg), A = std::move(A), max_iter, tol]
						(tensor<D,TT,R>& x, const tensor<D,TT,R>& b) 
		{
			auto r = b - A(x);
			auto p = r.clone();

			auto rzold = vdot(r, r).real();
			double resid = std::sqrt(rzold.item());

			for (i32 iter = 0; iter < max_iter; ++iter) {

				if (resid < tol) {
					break;
				}

				auto Ap = A(p);

				std::tie(x,r,p,rzold) = cg.run(std::move(x), std::move(p), std::move(Ap), std::move(r));

				resid = std::sqrt(rzold.item());

			}

			return x;
		};

		return cgl;
	}


	

}


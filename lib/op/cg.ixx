module cg;

#include "../pch.hpp"

export module cg;

import op;
import tensor;
import trace;

namespace hasty {

    template<is_hom_square_tensor_operator Op, is_hom_square_tensor_operator PrecondOp>
    requires    std::same_as<typename Op::device_type_t, typename PrecondOp::device_type_t> &&
                std::same_as<typename Op::output_tensor_type_t, typename PrecondOp::input_tensor_type_t> &&
                std::same_as<typename Op::output_tensor_rank_t, typename PrecondOp::input_tensor_rank_t>
    auto conjugate_gradient(sptr<Op> A, sptr<PrecondOp> P, i32 max_iter = 0, double tol) 
    {
        using D = Op::device_type_t;
        using TT == Op::input_tensor_type_t;
        using R = Op::input_rank_t;

        auto out = A(x);

        auto r = b - A(x);
        auto z = P.has_value() ? P.value()(r) : r;

        auto p = z;

        tensor_prototype<D,TT,R> x("x");
        tensor_prototype<D,TT,R> p("p");
        tensor_prototype<D,TT,R> Ap("Ap")
        tensor_prototype<D,TT,R> r("r");
        tensor_prototype<D,TT,R> z("z");
        tensor_prototype<D,real_t<TT>,0> rzold("rzold");

        auto cg1 = trace_function_factory<decltype(x), decltype(r)>::make("cg_step1", x,p,Ap,r);

        cg1.add_lines(
std::format(R"ts(
    pAp = torch.real(torch.vdot(p.flatten(),Ap.flatten()))
    alpha = rzold / pAp
    x += p * alpha
    r -= Ap * alpha
    return (x, r)
)ts"));

        cg1.compile();

        auto cg2 = trace_function<decltype(p), decltype(rzold)>::make("cg_step2", z,r,p,rzold);

        cg2.add_lines(
std::format(R"ts(
    rznew = torch.real(torch.vdot(r.flatten(),z.flatten()))
    p = z + p * (rznew / rzold)
    
    return p, rznew
)ts"));

        cg2.compile();

        auto cgl = [cg1 = std::move(cg1), cg2 = std::move(cg2), A = std::move(A), P = std::move(P),
                            max_iter, tol](tensor<D,TT,R>& x, const tensor<D,TT,R>& b) {

            auto r = b - A(x);
            auto z = P(r);
            auto p = z.clone();

            auto rzold = vdot(r, z).real();
            double resid = std::sqrt(rzold.item());

            for (i32 iter = 0; iter < max_iter; ++iter) {

                if (resid < tol) {
                    break;
                }

                auto Ap = A(p);

                std::tie(x,r) = cg1.run(std::move(x), p, std::move(Ap), std::move(r));

                z = P(r);

                std::tie(p,rzold) = cg2.run(std::move(z), r, std::move(p), std::move(rzold));

                resid = std::sqrt(rzold.item());

            }

            return x;
        };

        return cgl;
    }

}


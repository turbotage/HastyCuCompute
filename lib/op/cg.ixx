module cg;

#include "../pch.hpp"

export module cg;

import op;
import tensor;
import trace;

namespace hasty {

    template<is_square_tensor_operator Op, is_square_tensor_operator PrecondOp>
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
        tensor_prototype<D,TT,R> r("r");
        tensor_prototype<D,TT,R> z("z");

        auto cg1 = trace_function_factory<decltype(x), decltype(r)>::make("cg_step1", x,p,r);

        cg1.add_lines(
std::format(R"ts(
    pAp = torch.real(torch.vdot(p.flatten(),Ap.flatten()))
    alpha = rzold / pAp
    x += p * alpha
    r -= Ap * alpha
    return (x, r)
)ts"));

        cg1.compile();

        auto cg2 = trace_function<decltype(p)>::make("cg_step2", z, r, )
        


    }

}


module cg;

#include "../pch.hpp"

export module cg;

import op;
import tensor;
import trace;

namespace hasty {

    template<is_square_tensor_operator Op, is_square_tensor_operator PrecondOp>
    auto conjugate_gradient(
        Op A, 
        const tensor<Op::device_type_t, Op::output_tensor_type_t, Op::output_rank_t::value> &b,
        tensor<Op::device_type_t, Op::output_tensor_type_t, Op::output_rank_t::value> &x,
        opt<PrecondOp> P, i32 max_iter = 0, double tol) 
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


        auto cg = trace_function_factory<decltype(x)>::make("cg_step", x,p,r,z);

        cg.add_lines(
std::format(R"ts(
    

)ts"))

        for (i32 i = 0; i < max_iter; i++) {
            auto Ap = A(p);

        }
    }

}


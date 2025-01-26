module;

#include "pch.hpp"

export module alg;

import op;
import tensor;

namespace hasty {

    template<is_hom_square_tensor_operator Op>
    requires is_fp_tensor_type<typename Op::input_tensor_type_t>
    auto max_eig(
        Op A, 
        tensor_matching_input_op_t<Op>&& x0, 
        i32 max_iter = 10
    ) -> std::tuple<real_t<Op::input_tensor_type_t>, tensor_matching_input_op_t<Op>>
    {
        tensor_matching_input_op_t<Op> y = empty_tensor_like(x0);
        for (int i = 0; i < max_iter; ++i) {
            y = A(x0);
            real_t<Op::input_tensor_type_t> maxeig = y.norm();
            x0 = y / base_t<Op::input_tensor_type_t>(maxeig);
        }

        return std::make_tuple(maxeig, x0);
    }

}
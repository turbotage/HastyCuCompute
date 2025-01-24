module;

#include "pch.hpp"

export module alg;

import op;

namespace hasty {

    template<is_tensor_operator Op, is_device D, is_tensor_type TT, size_t R>
    auto max_eig(Op A, tensor<D,TT,R>&& x0, i32 max_iter = 10, double tol = 1e-6) 
    {
        for (int i = 0; i < max_iter; ++i) {
            auto y = A(x0);
            auto maxeig = y.norm();
            x0 = y / maxeig;
        }
    }
}
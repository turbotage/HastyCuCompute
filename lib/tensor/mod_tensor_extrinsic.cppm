module;

#include "pch.hpp"

export module tensor:extrinsic;

import util;
import :intrinsic;

namespace hasty {

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim, std::optional<fft_norm> norm);

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim, std::optional<fft_norm> norm);

    export template<is_device D1, is_tensor_type TT1, size_t R>
    tensor<D1,TT1,0> vdot(const tensor<D1,TT1,R>& lhs, const tensor<D1,TT1,R>& rhs);

}

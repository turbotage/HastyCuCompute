module;

#include "pch.hpp"

export module tensor:impl_extrinsic_operator;

import util;
import :intrinsic;

namespace hasty {


    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator+(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.add(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator-(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs)
    {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.sub(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator*(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.mul(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape; 

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator/(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.div(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }



}


module;

#include "pch.hpp"

export module tensor:impl_extrinsic_operator;
//module tensor:impl_extrinsic_operator;

import util;
import :intrinsic;

namespace hasty {

    // ADDITION
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

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator+(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = lhs._pimpl->underlying_tensor.add(rhs);
        return tensor<D1,TT1,R1>(lhs.get_shape(), std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator+(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = rhs._pimpl->underlying_tensor.add(lhs);
        return tensor<D1,TT1,R1>(rhs.get_shape(), std::move(newtensor));
    }

    // SUBTRACTION
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
    
    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator-(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = lhs._pimpl->underlying_tensor.sub(rhs);
        return tensor<D1,TT1,R1>(lhs.get_shape(), std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator-(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = rhs._pimpl->underlying_tensor.sub(lhs);
        return tensor<D1,TT1,R1>(rhs.get_shape(), std::move(newtensor));
    }

    // MULTIPLICATION
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

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator*(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = lhs._pimpl->underlying_tensor.mul(rhs);
        return tensor<D1,TT1,R1>(lhs.get_shape(), std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator*(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = rhs._pimpl->underlying_tensor.mul(lhs);
        return tensor<D1,TT1,R1>(rhs.get_shape(), std::move(newtensor));
    }

    // DIVISION
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

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator/(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = lhs._pimpl->underlying_tensor.div(rhs);
        return tensor<D1,TT1,R1>(lhs.get_shape(), std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R1>
    auto operator/(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1> {
        at::Tensor newtensor = rhs._pimpl->underlying_tensor.div(lhs);
        return tensor<D1,TT1,R1>(rhs.get_shape(), std::move(newtensor));
    }

}


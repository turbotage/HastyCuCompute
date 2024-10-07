module;

#include "pch.hpp"

export module tensor:impl_intrinsic_math;

import util;
import :intrinsic;

namespace hasty {

    template<is_device D, is_tensor_type TT, size_t RANK>
    auto tensor<D,TT,RANK>::norm() const -> std::conditional_t<is_fp32_tensor_type<TT>, float, double> 
    requires is_fp_tensor_type<TT> 
    {
        if constexpr(is_fp32_tensor_type<TT>) {
            return _pimpl->underlying_tensor.norm().template item<float>();
        }
        else if constexpr(is_fp64_tensor_type<TT>) {
            return _pimpl->underlying_tensor.norm().template item<double>();
        }
        else {
            static_assert(false, "tensor::norm: unknown precission");
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    auto tensor<D,TT,RANK>::abs() const {
        if constexpr(is_fp32_tensor_type<TT>) {
            auto ret = _pimpl->underlying_tensor.abs().to(at::ScalarType::Float);
            return tensor<D, f32_t, RANK>(_pimpl->shape, std::move(ret));
        }
        else if constexpr(is_fp64_tensor_type<TT>) {
            auto ret = _pimpl->underlying_tensor.abs().to(at::ScalarType::Double);
            return tensor<D, f64_t, RANK>(_pimpl->shape, std::move(ret));
        }
        else if constexpr(is_int_tensor_type<TT>) {
            auto ret = _pimpl->underlying_tensor.abs().to(scalar_type_func<TT>());
            return tensor<D, TT, RANK>(_pimpl->shape, std::move(ret));
        }
        else {
            static_assert(false, "tensor::abs: unknown precission");
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    auto tensor<D,TT,RANK>::max() const {
        return _pimpl->underlying_tensor.max().template item<base_t<TT>>();
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    auto tensor<D,TT,RANK>::min() const {
        return _pimpl->underlying_tensor.min().template item<base_t<TT>>();
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    auto tensor<D,TT,RANK>::real() const  {
        if constexpr(is_fp_complex_tensor_type<TT>) {
            auto ret = _pimpl->underlying_tensor.real();
            return tensor<D, real_t<TT>, RANK>(_pimpl->shape, std::move(ret));
        }
        else {
            return clone();
        }
    }


}


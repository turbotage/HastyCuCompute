module;

#include "pch.hpp"

export module tensor:impl_extrinsic_math;

import util;
import :intrinsic;

namespace hasty {

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t,
        span<R1> s,
        span<R2> dim,
        opt<fft_norm> norm)
    {
        auto normstr = [&norm]() -> at::optional<c10::string_view> {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return at::optional<c10::string_view>("forward");
                case fft_norm::BACKWARD:
                    return at::optional<c10::string_view>("backward");
                case fft_norm::ORTHO:
                    return at::optional<c10::string_view>("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
            return at::nullopt;
        };
        

        at::Tensor newtensor = torch::fft::fftn(t._pimpl->underlying_tensor,
            s.to_opt_arr_ref(),
            dim.to_opt_arr_ref(),
            normstr()
        );
        return tensor<D1,TT1,R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t,
        span<R1> s,
        span<R2> dim,
        opt<fft_norm> norm)
    {
        auto normstr = [&norm]() -> at::optional<c10::string_view> {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return at::optional<c10::string_view>("forward");
                case fft_norm::BACKWARD:
                    return at::optional<c10::string_view>("backward");
                case fft_norm::ORTHO:
                    return at::optional<c10::string_view>("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
            return at::nullopt;
        };

        at::Tensor newtensor = torch::fft::ifftn(t._pimpl->underlying_tensor,
            s.to_opt_arr_ref(),
            dim.to_opt_arr_ref(),
            normstr()
        );
        return tensor<D1,TT1,R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R>
    tensor<D1,TT1,0> vdot(const tensor<D1,TT1,R>& lhs, const tensor<D1,TT1,R>& rhs) 
    {
        at::Tensor newtensor = at::vdot(lhs._pimpl->underlying_tensor.flatten(), rhs._pimpl->underlying_tensor.flatten());
        return tensor<D1,TT1,0>({}, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, size_t R>
    tensor<D1,TT1,R> exp(const tensor<D1,TT1,R>& t)
    {
        at::Tensor newtensor = torch::exp(t._pimpl->underlying_tensor);
        return tensor<D1,TT1,R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

}
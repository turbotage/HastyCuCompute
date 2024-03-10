module;

#include "pch.hpp"

module tensor;

import util;

using namespace hasty;

template<device_fp F1, device_fp F2, size_t R1, size_t R2>
requires std::same_as<device_type_t<F1>, device_type_t<F2>>
auto hasty::operator+(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
    constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

    at::Tensor newtensor = lhs._pimpl->underlying_tensor + rhs._pimpl->underlying_tensor;
    std::array<int64_t, RETRANK> new_shape;

    assert(newtensor.ndimension() == RETRANK);

    for_sequence<RETRANK>([&](auto i) {
        new_shape[i] = newtensor.size(i);
    });

    return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
}


template<device_fp F1, device_fp F2, size_t R1, size_t R2>
requires std::same_as<device_type_t<F1>, device_type_t<F2>>
auto hasty::operator-(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs)
{
    constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

    at::Tensor newtensor = lhs._pimpl->underlying_tensor - rhs._pimpl->underlying_tensor;
    std::array<int64_t, RETRANK> new_shape;

    assert(newtensor.ndimension() == RETRANK);

    for_sequence<RETRANK>([&](auto i) {
        new_shape[i] = newtensor.size(i);
    });

    return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
}

template<device_fp F1, device_fp F2, size_t R1, size_t R2>
requires std::same_as<device_type_t<F1>, device_type_t<F2>>
auto hasty::operator*(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
    constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

    at::Tensor newtensor = lhs._pimpl->underlying_tensor * rhs._pimpl->underlying_tensor;
    std::array<int64_t, RETRANK> new_shape;

    assert(newtensor.ndimension() == RETRANK);

    for_sequence<RETRANK>([&](auto i) {
        new_shape[i] = newtensor.size(i);
    });

    return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
}

template<device_fp F1, device_fp F2, size_t R1, size_t R2>
requires std::same_as<device_type_t<F1>, device_type_t<F2>>
auto hasty::operator/(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
    constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

    at::Tensor newtensor = lhs._pimpl->underlying_tensor / rhs._pimpl->underlying_tensor;
    std::array<int64_t, RETRANK> new_shape;

    assert(newtensor.ndimension() == RETRANK);

    for_sequence<RETRANK>([&](auto i) {
        new_shape[i] = newtensor.size(i);
    });

    return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
}


template<device_fp F, size_t R, size_t R1, size_t R2>
requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R>
tensor<F, R> hasty::fftn(tensor<F, R> t,
    span<R1> s,
    span<R2> dim,
    std::optional<fft_norm> norm)
{
    auto normstr = [&norm]() {
        if (norm.has_value()) {
            switch (norm.value()) {
            case fft_norm::FORWARD:
                return std::string("forward");
            case fft_norm::BACKWARD:
                return std::string("backward");
            case fft_norm::ORTHO:
                return std::string("ortho");
            default:
                throw std::runtime_error("Invalid fft_norm value");
            }
        }
    };

    at::Tensor newtensor = at::fft_fftn(t._pimpl->underlying_tensor,
        s.to_torch_arr(),
        dim.to_torch_arr(),
        norm.has_value() ? normstr() : at::nullopt
    );
    return tensor<F, R>(span<R>(newtensor.sizes()), std::move(newtensor));
}   
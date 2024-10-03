module;

#include "pch.hpp"

module tensor:tensor_impl;

import util;

template<is_device D, is_tensor_type TT, size_t RANK>
hasty::tensor_impl<D,TT,RANK>::tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
    : shape(input_shape), underlying_tensor(std::move(input))
{
    //debug::print_memory_usage("tensor_impl::tensor_impl: (1)");
}
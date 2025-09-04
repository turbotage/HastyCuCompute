#pragma once

#include "interface_includes.hpp"

namespace ffi {

    LIB_EXPORT std::vector<at::Tensor> test_simple_invert();

    LIB_EXPORT at::Tensor test_offresonance_operator();

    LIB_EXPORT at::Tensor test_whitten_offresonance_operator();

    LIB_EXPORT at::Tensor test_normal_operators();

    LIB_EXPORT void test_prototype_stuff();

    LIB_EXPORT void test_trace_function();

}


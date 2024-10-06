#pragma once

#include "interface_includes.hpp"

namespace ffi {

    LIB_EXPORT std::vector<at::Tensor> test_simple_invert();

    LIB_EXPORT std::vector<at::Tensor> test_offresonance_operator();

}


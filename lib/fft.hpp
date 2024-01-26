#pragma once

#include <cufft.h>
#include <cufftXt.h>

#include "util.hpp"


inline int cufft_check_errors(cufftResult error)
{
    if (error != cufftResult::CUFFT_SUCCESS) {
        throw std::runtime_error("");
    }
    return 0; 
}

namespace hasty {




}








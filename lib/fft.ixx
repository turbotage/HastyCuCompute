module;

#include <cufft.h>
#include <cufftXt.h>

#include "util.hpp"

export module fft;

inline int cufft_check_errors(cufftResult error)
{
    if (error != cufftResult::CUFFT_SUCCESS) {
        throw std::runtime_error("");
    }
    return 0; 
}

namespace hasty {




}








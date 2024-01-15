module;

#include <cufft.h>
#include <cufftXt.h>

#include "util.hpp"

export module fft;

inline int cufft_check_errors(cufftResult error)
{
    if (error != cudaSuccess) {
        throw std::runtime_error("");
    }
}

namespace hasty {




}








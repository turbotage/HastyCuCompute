module;

#include "pch.hpp"

module vkfft;

namespace hasty {
    namespace fft {

        std::array<VkFFT_Cache, (size_t)device_idx::MAX_CUDA_DEVICES> global_vkfft_cache = {};

    }
}
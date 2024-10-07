module;

#include "pch.hpp"

export module tensor:base;

import util;

namespace hasty {

    export using TensorBackend = at::Tensor;
    export using TensorBackendDevice = at::Device;


    export enum struct device_idx {
        CPU = -1,
        CUDA0 = 0,
        CUDA1 = 1,
        CUDA2 = 2,
        CUDA3 = 3,
        CUDA4 = 4,
        CUDA5 = 5,
        CUDA6 = 6,
        CUDA7 = 7,
        CUDA8 = 8,
        CUDA9 = 9,
        CUDA10 = 10,
        CUDA11 = 11,
        CUDA12 = 12,
        CUDA13 = 13,
        CUDA14 = 14,
        CUDA15 = 15,
        MAX_CUDA_DEVICES = 16
    };

    export TensorBackendDevice get_backend_device(device_idx idx) {
        if (idx == device_idx::CPU) {
            return at::Device(at::DeviceType::CPU);
        }
        else {
            return at::Device(at::DeviceType::CUDA, at::DeviceIndex(idx));
        }
    }

    bool is_cuda(device_idx idx) {
        return idx != device_idx::CPU;
    }

    template<is_device D>
    constexpr bool device_match(device_idx idx) {
        if constexpr(std::is_same_v<D, cuda_t>) {
            return is_cuda(idx);
        }
        else if constexpr(std::is_same_v<D, cpu_t>) {
            return !is_cuda(idx);
        }
        else {
            static_assert(false, "device_match: unknown device type");
        }
    }

    export enum struct fft_norm {
        FORWARD,
        BACKWARD,
        ORTHO
    };


}
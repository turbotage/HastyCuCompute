module;

#include "pch.hpp"

export module tensor:base;

import util;

namespace hasty {

    export using TensorBackend = at::Tensor;
    export using TensorBackendDevice = at::Device;

    export using TensorBackendDeviceType = at::DeviceType;

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
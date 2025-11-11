module;

#include "pch.hpp"

export module tensor:base;

import torch_base;
import util;

namespace hasty {

export using TensorBackend = hat::Tensor;
export using TensorBackendDevice = hat::Device;

export using TensorBackendDeviceType = hat::DeviceType;

export TensorBackendDevice get_backend_device(device_idx idx) {
	if (idx == device_idx::CPU) {
		return hat::Device(hat::DeviceType::CPU);
	}
	else {
		return hat::Device(hat::DeviceType::CUDA, hat::DeviceIndex(idx));
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
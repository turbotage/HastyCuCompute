module;

#include "pch.hpp"

export module tensor:tensor_impl;

//import pch;

import util;

namespace hasty {

    export template<is_device D, is_tensor_type TT, size_t R>
    class tensor;






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

    export at::Device get_torch_device(device_idx idx) {
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



    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor_impl {
    private:

        template<is_device DO1, is_tensor_type TTO1, size_t RO1>
        friend class tensor;

    public:

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input);

        tensor_impl(span<RANK> input_shape, at::Tensor input)
            : shape(input_shape.to_arr()), underlying_tensor(std::move(input))
        {
            //debug::print_memory_usage("tensor_impl::tensor_impl: (2)");
        }

        ~tensor_impl() {
            //debug::print_memory_usage("~tensor_impl: ");
        }

        base_t<TT>* mutable_data_ptr() { 
            return static_cast<base_t<TT>*>(underlying_tensor.data_ptr()); 
        }

        const base_t<TT>* const_data_ptr() const { 
            return static_cast<base_t<TT>*>(underlying_tensor.data_ptr()); 
        }

        device_idx get_device_idx() const {
            return static_cast<device_idx>(underlying_tensor.device().index());
        }

        std::array<int64_t, RANK> shape;
        at::Tensor underlying_tensor;
        //std::vector<TensorIndex> indices;
    };

}
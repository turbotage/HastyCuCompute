module;

#include "pch.hpp"

export module tensor:factory;

import util;
import :intrinsic;

namespace hasty {

    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor_factory {
    public:

        static tensor<D,TT,RANK> make(const std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            return tensor<D,TT,RANK>(input_shape, std::move(input));
        }

        static tensor<D,TT,RANK> make(span<RANK> input_shape, at::Tensor input) {
            return tensor<D,TT,RANK>(input_shape, std::move(input));
        }

        static uptr<tensor<D,TT,RANK>> make_unique(const std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            return std::make_unique<tensor<D,TT,RANK>>(input_shape, std::move(input));
        }

        static uptr<tensor<D,TT,RANK>> make_unique(span<RANK> input_shape, at::Tensor input) {
            return std::make_unique<tensor<D,TT,RANK>>(input_shape, std::move(input));
        }

        static tensor<D,TT,0> make_scalar(base_t<TT> val) requires (RANK == 0) {
            return tensor<D,TT,0>({}, at::scalar_tensor(val));
        }

    };

    export enum struct tensor_make_opts {
        EMPTY,
        ONES,
        ZEROS,
        RAND_UNIFORM
    };

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_empty_tensor(span<RANK> shape, device_idx didx = device_idx::CPU) {
        if constexpr (std::is_same_v<D, cuda_t>) {
            if (didx == device_idx::CPU) {
                throw std::runtime_error("make_empty_tensor: cannot make empty tensor on CPU with CUDA device type");
            }
        } else {
            if (didx != device_idx::CPU) {
                throw std::runtime_error("make_empty_tensor: cannot make empty tensor on non-CPU device with non-CUDA device type");
            }
        }
        at::TensorOptions opts = at::TensorOptions(scalar_type_func<TT>(
                )).device(c10::Device(device_type_func<D>(), i32(didx)));
        return tensor<D,TT,RANK>(shape, std::move(at::empty(shape.to_arr_ref(), opts)));
    }   

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_ones_tensor(span<RANK> shape, device_idx didx = device_idx::CPU) {
        if constexpr (std::is_same_v<D, cuda_t>) {
            if (didx == device_idx::CPU) {
                throw std::runtime_error("make_ones_tensor: cannot make ones tensor on CPU with CUDA device type");
            }
        } else {
            if (didx != device_idx::CPU) {
                throw std::runtime_error("make_ones_tensor: cannot make ones tensor on non-CPU device with non-CUDA device type");
            }
        }
        at::TensorOptions opts = at::TensorOptions(scalar_type_func<TT>(
                )).device(c10::Device(device_type_func<D>(), i32(didx)));
        return tensor<D,TT,RANK>(shape, std::move(at::ones(shape.to_arr_ref(), opts)));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_zeros_tensor(span<RANK> shape, device_idx didx = device_idx::CPU) {
        if constexpr (std::is_same_v<D, cuda_t>) {
            if (didx == device_idx::CPU) {
                throw std::runtime_error("make_zeros_tensor: cannot make zeros tensor on CPU with CUDA device type");
            }
        } else {
            if (didx != device_idx::CPU) {
                throw std::runtime_error("make_zeros_tensor: cannot make zeros tensor on non-CPU device with non-CUDA device type");
            }
        }
        at::TensorOptions opts = at::TensorOptions(scalar_type_func<TT>(
                )).device(c10::Device(device_type_func<D>(), i32(didx)));
        return tensor<D,TT,RANK>(shape, std::move(at::zeros(shape.to_arr_ref(), opts)));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_rand_tensor(span<RANK> shape, device_idx didx = device_idx::CPU) {
        if constexpr (std::is_same_v<D, cuda_t>) {
            if (didx == device_idx::CPU) {
                throw std::runtime_error("make_rand_tensor: cannot make rand tensor on CPU with CUDA device type");
            }
        } else {
            if (didx != device_idx::CPU) {
                throw std::runtime_error("make_rand_tensor: cannot make rand tensor on non-CPU device with non-CUDA device type");
            }
        }
        at::TensorOptions opts = at::TensorOptions(scalar_type_func<TT>(
                )).device(c10::Device(device_type_func<D>(), i32(didx)));
        return tensor<D,TT,RANK>(shape, std::move(at::rand(shape.to_arr_ref(), opts)));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_tensor(span<RANK> shape, 
        device_idx didx = device_idx::CPU, tensor_make_opts make_opts=tensor_make_opts::EMPTY)
    {
        switch (make_opts) {
            case tensor_make_opts::EMPTY:
                return make_empty_tensor<D,TT,RANK>(shape, didx);
            case tensor_make_opts::ONES:
                return make_ones_tensor<D,TT,RANK>(shape, didx);
            case tensor_make_opts::ZEROS:
                return make_zeros_tensor<D,TT,RANK>(shape, didx);
            case tensor_make_opts::RAND_UNIFORM:
                return make_rand_tensor<D,TT,RANK>(shape, didx);
            default:
                throw std::runtime_error("Unknown tensor_make_opts option");
        }
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_empty_tensor_like(const tensor<D,TT,RANK>& other)
    {
        return tensor<D,TT,RANK>(other.shape(), std::move(at::empty_like(other.get_tensor())));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_ones_tensor_like(const tensor<D,TT,RANK>& other)
    {
        return tensor<D,TT,RANK>(other.shape(), std::move(at::ones_like(other.get_tensor())));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_zeros_tensor_like(const tensor<D,TT,RANK>& other)
    {
        return tensor<D,TT,RANK>(other.shape(), std::move(at::zeros_like(other.get_tensor())));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_rand_tensor_like(const tensor<D,TT,RANK>& other)
    {
        return tensor<D,TT,RANK>(other.shape(), std::move(at::rand_like(other.get_tensor())));
    }

    export template<is_device D, is_tensor_type TT, size_t RANK>
    auto make_tensor_unique(TensorBackend tensorin) -> uptr<tensor<D,TT,RANK>>
    {
        if (tensorin.ndimension() != RANK)
            throw std::runtime_error("make_tensor: tensor.ndimension() did not match RANK");

        if (tensorin.dtype().toScalarType() != scalar_type_func<TT>())
            throw std::runtime_error("make_tensor: tensor.dtype() did not match templated any_fp FP");

        struct creator : tensor<D,TT,RANK> {
            creator(std::initializer_list<int64_t> a, TensorBackend b)
                : tensor<D,TT,RANK>(a, std::move(b)) {}
        };

        return std::make_unique<creator>(tensorin.sizes(), std::move(tensorin));
    }



}


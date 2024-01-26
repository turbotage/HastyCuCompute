#pragma once

#include "util.hpp"


namespace hasty {

    export template<any_fp FP, size_t RANK>
    struct tensor_impl {

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
            : shape(input_shape), underlying_tensor(std::move(input))
        {   
            data_ptr = reinterpret_cast<underlying_type<FP>*>(underlying_tensor.mutable_data_ptr()); 
        }

        std::array<int64_t, RANK> shape;
        at::Tensor underlying_tensor;
        underlying_type<FP>* data_ptr = nullptr;
        std::shared_ptr<trace>  tracectx = nullptr;
    };

    export template<any_fp FP, size_t RANK>
    struct tensor {
        
        tensor() = default;

        tensor(const std::array<int64_t, RANK>& input_shape, at::Tensor input) 
            : _pimpl(std::make_shared<tensor_impl<FP, RANK>>(input_shape, std::move(input)))
        {}

        template<size_t R>
        requires less_than<R, RANK>
        int64_t shape() const { return _pimpl->shape[R]; }

        underlying_type<FP>* mutable_data() { return _pimpl->data_ptr; }
        const underlying_type<FP>* immutable_data() const { return _pimpl->data_ptr; }
        underlying_type<FP>* const_cast_data() const { return const_cast<underlying_type<FP>*>(_pimpl->data_ptr); }

    private:
        std::shared_ptr<tensor_impl<FP,RANK>> _pimpl;
    };

    export template<any_fp FP, size_t RANK>
    tensor<FP, RANK> make_tensor(const std::array<int64_t, RANK>& shape, const std::string& device_str)
    {
        at::TensorOptions opts = at::TensorOptions(static_type_to_scalar_type<FP>()).device(device_str);

        return tensor<FP,RANK>(shape, std::move(at::empty(shape, opts)));
    }

    export enum struct tensor_make_opts {
        EMPTY,
        ONES,
        ZEROS,
        RAND_NORMAL,
        RAND_UNIFORM
    };

    export template<any_fp FP, size_t RANK>
    std::unique_ptr<tensor<FP, RANK>> make_tensor(at::Tensor tensorin)
    {
        if (tensorin.ndimension() != RANK)
            throw std::runtime_error("make_tensor: tensor.ndimension() did not match RANK");

        if (tensorin.dtype().toScalarType() != static_type_to_scalar_type<FP>())
            throw std::runtime_error("make_tensor: tensor.dtype() did not match templated any_fp FP");

        struct creator : tensor<FP, RANK> {
            creator(std::initializer_list<int64_t> a, at::Tensor b)
                : tensor<FP, RANK>(a, std::move(b)) {}
        };


        return std::make_unique<creator>(tensorin.sizes(), std::move(tensorin));
    }


}
module;

#include "util.hpp"

export module tensor;


namespace hasty {

    export template<any_fp FP, size_t RANK>
    struct tensor {
        
        tensor() {}

    protected:

        tensor(std::initializer_list<int64_t> shape, at::Tensor tensor)
            : _shape(std::move(shape)), _tensor(std::move(tensor))
        {   _data_ptr = reinterpret_cast<underlying_type<FP>>(_tensor.data_ptr()); }

        template<size_t R>
        requires less_than<R, RANK>
        int32_t shape() { return _shape[R]; }

        underlying_type<FP>* data() { return _data_ptr; }
        const underlying_type<FP>* data() const { return _data_ptr; }

    private:
        std::array<int32_t, RANK> _shape;
        at::Tensor _tensor;
        underlying_type<FP>* _data_ptr = nullptr;
    };

    export template<any_fp FP, size_t RANK>
    std::unique_ptr<tensor<FP, RANK>> make_tensor(std::initializer_list<int64_t> shape, const std::string& device_str)
    {
        at::TensorOptions opts = at::TensorOptions(static_type_to_scalar_type<FP>()).device(device_str);

        struct creator : tensor<FP, RANK> {
            creator(std::initializer_list<int64_t> a, at::Tensor b)
                : tensor<FP, RANK>(a, std::move(b)) {}
        };

        return std::make_unique<creator>(std::move(shape), std::move(at::empty(shape, opts)));
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
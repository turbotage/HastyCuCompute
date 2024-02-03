module;

#include "pch.hpp"

export module tensor;

import util;
import trace;

namespace hasty {

    
    export struct None {};

    export struct Ellipsis {};

    export struct Slice {
        
        std::optional<int64_t> start;
        std::optional<int64_t> end;
        std::optional<int64_t> step;
    };

    export template<typename T>
    concept index_type =   std::is_same_v<T, None> 
                        || std::is_same_v<T, Ellipsis> 
                        || std::is_same_v<T, Slice>
                        || std::is_integral_v<T>;

    export template<index_type... Itx, size_t R>
    constexpr int64_t get_slice_rank(Itx... idxs)
    {
        int none;
        int ints;
        int ellipsis;

        //auto idxss = std::make_tuple<Itx...>(std::forward<Itx...>(idxs...));
        auto idxss = std::make_tuple(idxs...);

        for_sequence<sizeof...(Itx)>([&](auto i) constexpr {
            if constexpr(std::is_same_v<decltype(idxss.template get<i>()), None>) {
                ++none;
            } 
            else if constexpr(std::is_integral_v<decltype(idxss.template get<i>())>) {
                ++ints;
            }
            /*
            else if constexpr(std::is_same_v<decltype(idxss.template get<i>()), Ellipsis>) {
                ++ellipsis;
            } 
            */
        });

        return R - ints + none;
    }

    template<typename T>
    c10::optional<T> torch_optional(const std::optional<T>& opt)
    {
        if (opt.has_value()) {
            return c10::optional(opt.value());
        }
        return c10::nullopt;
    }

    template<index_type Idx>
    at::indexing::TensorIndex torchidx(Idx idx) {
        if constexpr(std::is_same_v<Idx, None>) {
            return at::indexing::None;
        } 
        else if constexpr(std::is_same_v<Idx, Ellipsis>) {
            return at::indexing::Ellipsis;
        }
        else if constexpr(std::is_same_v<Idx, Slice>) {
            return at::indexing::Slice(
                torch_optional(idx.start),
                torch_optional(idx.end),
                torch_optional(idx.step));
        } else if constexpr(std::is_integral_v<Idx>) {
            return idx;
        }
    }


    export template<device_fp FPT, size_t RANK>
    struct tensor_impl {

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
            : shape(input_shape), underlying_tensor(std::move(input))
        {}

        underlying_type<FPT>* mutable_data_ptr() { return underlying_tensor.mutable_data_ptr<underlying_type<FPT>>(); }
        const underlying_type<FPT>* const_data_ptr() { return underlying_tensor.const_data_ptr<underlying_type<FPT>>(); }

        std::array<int64_t, RANK> shape;
        at::Tensor underlying_tensor;
        std::shared_ptr<trace>  tracectx = nullptr;
    };

    export template<device_fp FPT, size_t RANK>
    struct tensor_view {
        

    private:
        
    }

    export template<device_fp FPT, size_t RANK>
    struct tensor {
        
        tensor() = default;

        tensor(const std::array<int64_t, RANK>& input_shape, at::Tensor input) 
            : _pimpl(std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input)))
        {}

        template<size_t R>
        requires less_than<R, RANK>
        int64_t shape() const { return _pimpl->shape[R]; }

        underlying_type<FPT>* mutable_data_ptr() { return _pimpl->mutable_data_ptr(); }
        const underlying_type<FPT>* const_data_ptr() { return _pimpl->const_data_ptr();}

        template<cpu_fp F>
        void fill(F val) { _pimpl->underlying_tensor.fill_(val); }

        template<index_type ...Idx>
        auto operator[](Idx... indices) {
            constexpr auto RETRANK = get_slice_rank(indices...);

            at::Tensor tensorview = _pimpl->underlying_tensor.index(tensoridx(indices)...);

            

        }

    private:
        std::shared_ptr<tensor_impl<FPT,RANK>> _pimpl;
    };


    export enum struct tensor_make_opts {
        EMPTY,
        ONES,
        ZEROS,
        RAND_NORMAL,
        RAND_UNIFORM
    };

    export template<device_fp FP, size_t RANK>
    tensor<FP, RANK> make_tensor(const std::array<int64_t, RANK>& shape, 
        const std::string& device_str="cpu", tensor_make_opts make_opts=tensor_make_opts::EMPTY)
    {
        at::TensorOptions opts = at::TensorOptions(static_type_to_scalar_type<FP>()).device(device_str);

        switch (make_opts) {
            case tensor_make_opts::EMPTY:
                return tensor<FP,RANK>(shape, std::move(at::empty(shape, opts)));
            case tensor_make_opts::ONES:
                return tensor<FP,RANK>(shape, std::move(at::ones(shape, opts)));
            case tensor_make_opts::ZEROS:
                return tensor<FP,RANK>(shape, std::move(at::zeros(shape, opts)));
            case tensor_make_opts::RAND_NORMAL:
                return tensor<FP,RANK>(shape, std::move(at::normal(shape, opts)));
            case tensor_make_opts::RAND_UNIFORM:
                return tensor<FP,RANK>(shape, std::move(at::rand(shape, opts)));
            default:
                throw std::runtime_error("Unknown tensor_make_opts option");
        }
    }

    export template<device_fp FP, size_t RANK>
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
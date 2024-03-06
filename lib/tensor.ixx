module;

#include "pch.hpp"

export module tensor;

import util;

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

    using TensorIndex = std::variant<None, Ellipsis, Slice, int64_t>;

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

    template<index_type Idx>
    std::string torchidxstr(Idx idx) {
        if constexpr(std::is_same_v<Idx, None>) {
            return "None";
        } 
        else if constexpr(std::is_same_v<Idx, Ellipsis>) {
            return "...";
        }
        else if constexpr(std::is_same_v<Idx, Slice>) {
            // If the Slice has start, end, and step values, format them as "start:end:step"
            if (idx.start.has_value() && idx.end.has_value() && idx.step.has_value()) {
                return std::format("{}:{}:{}", idx.start.value(), idx.end.value(), idx.step.value());
            } 
            // If the Slice has only start and end values, format them as "start:end"
            else if (idx.start.has_value() && idx.end.has_value()) {
                return std::format("{}:{}", idx.start.value(), idx.end.value());
            } 
            // If the Slice has only start and step values, format them as "start::step"
            else if (idx.start.has_value() && idx.step.has_value()) {
                return std::format("{}::{}", idx.start.value(), idx.step.value());
            } 
            // If the Slice has only end and step values, format them as ":end:step"
            else if (idx.end.has_value() && idx.step.has_value()) {
                return std::format(":{}:{}", idx.end.value(), idx.step.value());
            } 
            // If the Slice has only a start value, format it as "start:"
            else if (idx.start.has_value()) {
                return std::format("{}:", idx.start.value());
            } 
            // If the Slice has only an end value, format it as ":end"
            else if (idx.end.has_value()) {
                return std::format(":{}", idx.end.value());
            } 
            // If the Slice has only a step value, format it as "::step"
            else if (idx.step.has_value()) {
                return std::format("::{}", idx.step.value());
            }
        } 
        else if constexpr(std::is_integral_v<Idx>) {
            return std::to_string(idx);
        }
    }


    export template<device_fp FPT, size_t RANK>
    struct tensor_impl {

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
            : shape(input_shape), underlying_tensor(std::move(input))
        {}

        template<std::integral I>
        tensor_impl(span<I, RANK> input_shape, at::Tensor input)
            : shape(input_shape.to_arr()), underlying_tensor(std::move(input))
        {}

        underlying_type<FPT>* mutable_data_ptr() { return underlying_tensor.mutable_data_ptr<underlying_type<FPT>>(); }
        const underlying_type<FPT>* const_data_ptr() { return underlying_tensor.const_data_ptr<underlying_type<FPT>>(); }

        std::array<int64_t, RANK> shape;
        at::Tensor underlying_tensor;
        //std::vector<TensorIndex> indices;
    };

    export template<device_fp FPT, size_t RANK>
    struct tensor {
    private:
        std::shared_ptr<tensor_impl<FPT,RANK>> _pimpl;
    public:

        tensor() = default;

        tensor(const std::array<int64_t, RANK>& input_shape, at::Tensor input) 
            : _pimpl(std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input)))
        {}

        template<std::integral I>
        tensor(span<I, RANK> input_shape, at::Tensor input)
            : _pimpl(std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input)))
        {}

        template<size_t R>
        requires less_than<R, RANK>
        int64_t shape() const { 
            assert(_pimpl->shape[R] == _pimpl->underlying_tensor.size(R));    
            return _pimpl->shape[R]; 
        }

        constexpr int64_t ndim() const { 
            assert(RANK == _pimpl->underlying_tensor.ndimension()); 
            return RANK; 
        }

        int64_t nelem() const { 
            int64_t nelem = 0; 
            for_sequence<RANK>([&](auto i) { nelem *= _pimpl->shape[i]; });
            assert(nelem == _pimpl->underlying_tensor.numel());
            return nelem;
        }

        std::string devicestr() const {
            return _pimpl->underlying_tensor.device().str();
        
        }

        underlying_type<FPT>* mutable_data_ptr() { return _pimpl->mutable_data_ptr(); }
        const underlying_type<FPT>* const_data_ptr() { return _pimpl->const_data_ptr();}

        template<cpu_fp F>
        void fill_(F val) { _pimpl->underlying_tensor.fill_(val); }



        template<std::integral I, size_t R>
        tensor<FPT, R> view(span<I,R> shape) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(shape.to_torch_arr());
            return tensor<FPT, R>(shape, std::move(tensorview));
        }

        tensor<FPT, 1> flatview() {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            return tensor<FPT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        tensor<FPT, 1> flatslice(index_type auto& idx) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            tensorview = tensorview.index(torchidx(idx));
            return tensor<FPT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        template<index_type ...Idx>
        auto operator[](Idx... indices) {
            constexpr auto RETRANK = get_slice_rank(indices...);

            at::Tensor tensorview = _pimpl->underlying_tensor.index(tensoridx(indices)...);

            if (!tensorview.is_view())
                throw std::runtime_error("tensor::operator[]: tensorview is not a view");

            if (tensorview.ndimension() != RETRANK)
                throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");

            std::array<int64_t, RETRANK> new_shape;
            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = tensorview.size(i);
            });

            return tensor<FPT, RETRANK>(new_shape, std::move(tensorview));
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.copy_(other._pimpl->underlying_tensor);
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator+=(const tensor<FPT, R>& other) const {
            _pimpl->underlying_tensor.add_(other._pimpl->underlying_tensor);
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator-=(const tensor<FPT, R>& other) const {
            _pimpl->underlying_tensor.sub_(other._pimpl->underlying_tensor);
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator*=(const tensor<FPT, R>& other) const {
            _pimpl->underlying_tensor.mul_(other._pimpl->underlying_tensor);
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator/=(const tensor<FPT, R>& other) const {
            _pimpl->underlying_tensor.div_(other._pimpl->underlying_tensor);
        }

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires std::same_as<device_type_t<F1>, device_type_t<F2>>
        friend auto operator+(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
            constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

            at::Tensor newtensor = lhs._pimpl->underlying_tensor + rhs._pimpl->underlying_tensor;
            std::array<int64_t, RETRANK> new_shape;

            assert(newtensor.ndimension() == RETRANK);

            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = newtensor.size(i);
            });

            return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
        }

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires std::same_as<device_type_t<F1>, device_type_t<F2>>
        friend auto operator-(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
            constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

            at::Tensor newtensor = lhs._pimpl->underlying_tensor - rhs._pimpl->underlying_tensor;
            std::array<int64_t, RETRANK> new_shape;

            assert(newtensor.ndimension() == RETRANK);

            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = newtensor.size(i);
            });

            return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
        }

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires std::same_as<device_type_t<F1>, device_type_t<F2>>
        friend auto operator*(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
            constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

            at::Tensor newtensor = lhs._pimpl->underlying_tensor * rhs._pimpl->underlying_tensor;
            std::array<int64_t, RETRANK> new_shape;

            assert(newtensor.ndimension() == RETRANK);

            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = newtensor.size(i);
            });

            return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
        }

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires std::same_as<device_type_t<F1>, device_type_t<F2>>
        friend auto operator/(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
            constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

            at::Tensor newtensor = lhs._pimpl->underlying_tensor / rhs._pimpl->underlying_tensor;
            std::array<int64_t, RETRANK> new_shape;

            assert(newtensor.ndimension() == RETRANK);

            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = newtensor.size(i);
            });

            return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
        }

        export template<device_fp F, size_t R, size_t R1, size_t R2, std::integral I1, std::integral I2>
        friend tensor<F, size_t R> fftn(tensor<F, size_t R> t,
            ospan<I1,R1> s,
            ospan<I2,R2> dim,
            std::optional<fft_norm> norm = std::nullopt)
        {
            auto normstr = [&norm]() {
                if (norm.has_value()) {
                    switch (norm.value()) {
                    case fft_norm::FORWARD:
                        return std::string("forward");
                    case fft_norm::BACKWARD:
                        return std::string("backward");
                    case fft_norm::ORTHO:
                        return std::string("ortho");
                    default:
                        throw std::runtime_error("Invalid fft_norm value");
                    }
                }
            };

            at::Tensor newtensor = at::fft::fftn(t._pimpl->underlying_tensor,
                s.to_torch_arr(),
                dim.to_torch_arr(),
                norm.has_value() ? normstr() : at::nullopt
            );
            return tensor<F, R>(span<R>(newtensor.sizes()), std::move(newtensor));
        }   

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
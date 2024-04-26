module;

#include "pch.hpp"

export module tensor;

import util;

namespace hasty {

    export template<device_fp F, size_t R>
    class tensor;

    template<device_fp FPT, size_t RANK>
    class tensor_impl {
    public:

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
            : shape(input_shape), underlying_tensor(std::move(input))
        {}

        tensor_impl(span<RANK> input_shape, at::Tensor input)
            : shape(input_shape.to_arr()), underlying_tensor(std::move(input))
        {}

        underlying_type<FPT>* mutable_data_ptr() { return static_cast<underlying_type<FPT>*>(underlying_tensor.data_ptr()); }
        const underlying_type<FPT>* const_data_ptr() const { return static_cast<underlying_type<FPT>*>(underlying_tensor.data_ptr()); }

        std::array<int64_t, RANK> shape;
        at::Tensor underlying_tensor;
        //std::vector<TensorIndex> indices;
    };

    export enum struct fft_norm {
        FORWARD,
        BACKWARD,
        ORTHO
    };

    template<device_fp F, size_t R, size_t R1, size_t R2>
    tensor<F, R> fftn(const tensor<F, R>& t, span<R1> s, span<R2> dim,
        std::optional<fft_norm> norm);

    template<device_fp F, size_t R, size_t R1, size_t R2>
    tensor<F, R> ifftn(const tensor<F, R>& t, span<R1> s, span<R2> dim,
        std::optional<fft_norm> norm);


    export template<device_fp FPT, size_t RANK>
    class tensor_factory;

    export template<device_fp FPT, size_t RANK>
    class tensor {
    private:
        std::shared_ptr<tensor_impl<FPT,RANK>> _pimpl;
        friend class tensor_factory<FPT, RANK>;



        tensor(const std::array<int64_t, RANK>& input_shape, at::Tensor input) 
            : _pimpl(std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input)))
        {}

        tensor(span<RANK> input_shape, at::Tensor input)
            : _pimpl(std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input)))
        {}

        void assign(std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            _pimpl = std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input));
        }

        void assign(span<RANK> input_shape, at::Tensor input) {
            _pimpl = std::make_shared<tensor_impl<FPT, RANK>>(input_shape, std::move(input));
        }

    public:

        using tensor_device_fp = FPT;
        static constexpr std::integral_constant<size_t, RANK> size = {};

        auto& get_pimpl() { return _pimpl; }
        const auto& get_pimpl() const { return _pimpl; }

        auto& get_tensor() { return _pimpl->underlying_tensor; }
        const auto& get_tensor() const { return _pimpl->underlying_tensor; }

        tensor() = default;

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
        const underlying_type<FPT>* const_data_ptr() const { return _pimpl->const_data_ptr();}

        // I know what I am doing...
        underlying_type<FPT>* unconsted_data_ptr() const { return const_cast<underlying_type<FPT>*>(_pimpl->const_data_ptr()); }

        template<cpu_fp F>
        void fill_(F val) { _pimpl->underlying_tensor.fill_(val); }

        tensor<FPT, RANK> clone() const {
            at::Tensor newtensor = _pimpl->underlying_tensor.clone();
            return tensor<FPT, RANK>(_pimpl->shape, std::move(newtensor));
        }

        template<size_t R>
        tensor<FPT, R> view(span<R> shape) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(shape.to_torch_arr());
            return tensor<FPT, R>(shape, std::move(tensorview));
        }

        tensor<FPT, 1> flatview() {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            return tensor<FPT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        tensor<FPT, 1> flatslice(index_type auto& idx) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            tensorview = tensorview.index(tch::torchidx(idx));
            return tensor<FPT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        tensor<FPT, RANK> contiguous() {
            return tensor<FPT, RANK>(_pimpl->shape, _pimpl->underlying_tensor.contiguous());
        }

        void contiguous_() {
            _pimpl->underlying_tensor.contiguous_();
        }

        template<size_t N>
        requires less_than_or_equal<N, RANK>
        auto operator[](const std::array<Slice, N>& slices) {
            
            at::Tensor tensorview = _pimpl->underlying_tensor.index(tch::torchidx(std::tuple_cat(slices)));

            if (!tensorview.is_view())
                throw std::runtime_error("tensor::operator[]: tensorview is not a view");

            if (tensorview.ndimension() != RANK)
                throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");

            std::array<int64_t, RANK> new_shape;
            for_sequence<RANK>([&](auto i) {
                new_shape[i] = tensorview.size(i);
            });

            //return tensor_factory<FPT, RANK>::make(new_shape, std::move(tensorview));
            return tensor<FPT, RANK>(new_shape, std::move(tensorview));
        }

        template<index_type ...Idx>
        auto operator[](std::tuple<Idx...> indices) {
            constexpr auto RETRANK = get_slice_rank<RANK, Idx...>();

            at::Tensor tensorview = _pimpl->underlying_tensor.index(tch::torchidx(indices));

            if (!tensorview.is_view())
                throw std::runtime_error("tensor::operator[]: tensorview is not a view");

            if (tensorview.ndimension() != RETRANK)
                throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");

            std::array<int64_t, RETRANK> new_shape;
            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = tensorview.size(i);
            });

            return tensor_factory<FPT, RETRANK>::make(new_shape, std::move(tensorview));
        }

        template<index_type ...Idx>
        auto operator[](Idx... indices) {
            constexpr auto RETRANK = get_slice_rank<RANK, Idx...>();

            at::Tensor tensorview = _pimpl->underlying_tensor.index({tch::torchidx(indices)...});

            if (!tensorview.is_view())
                throw std::runtime_error("tensor::operator[]: tensorview is not a view");

            if (tensorview.ndimension() != RETRANK)
                throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");

            std::array<int64_t, RETRANK> new_shape;
            for_sequence<RETRANK>([&](auto i) {
                new_shape[i] = tensorview.size(i);
            });

            return tensor_factory<FPT, RETRANK>::make(new_shape, std::move(tensorview));
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.copy_(other.get_tensor());
        }

        void operator=(const tensor<FPT, RANK>& other) {
            _pimpl->underlying_tensor.copy_(other.get_tensor());
        }

        void set_(const tensor<FPT, RANK>& other) {
            if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
                throw std::runtime_error("tensor::set_: other tensor shape did not match this tensor shape");
            _pimpl->underlying_tensor.set_(other._pimpl->underlying_tensor);
        }

        void assign_(tensor<FPT, RANK>&& other) {
            if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
                throw std::runtime_error("tensor::assign_: other tensor shape did not match this tensor shape");
            _pimpl->underlying_tensor = std::move(other._pimpl->underlying_tensor);
        }

        auto norm() const {
            if constexpr(is_32bit_precission<FPT>()) {
                return _pimpl->underlying_tensor.norm().template item<float>();
            }
            else if constexpr(is_64bit_precission<FPT>()) {
                return _pimpl->underlying_tensor.norm().template item<double>();
            }
            else {
                throw std::runtime_error("tensor::norm: unknown precission");
            }
        }

        auto abs() const {
            if constexpr(is_32bit_precission<FPT>()) {
                auto ret = _pimpl->underlying_tensor.abs().to(::torch::kFloat);
                return tensor<cuda_f32, RANK>(_pimpl->shape, std::move(ret));
            }
            else if constexpr(is_64bit_precission<FPT>()) {
                auto ret = _pimpl->underlying_tensor.abs().to(::torch::kDouble);
                return tensor<cuda_f64, RANK>(_pimpl->shape, std::move(ret));
            }
            else {
                throw std::runtime_error("tensor::abs: unknown precission");
            }
        }

        auto max() const {
            if constexpr(is_32bit_precission<FPT>())
                return _pimpl->underlying_tensor.max().template item<float>();
            else if constexpr(is_64bit_precission<FPT>())
                return _pimpl->underlying_tensor.max().template item<double>();
            else
                throw std::runtime_error("tensor::max: unknown precission");
        }

        auto min() const {
            if constexpr(is_32bit_precission<FPT>())
                return _pimpl->underlying_tensor.min().template item<float>();
            else if constexpr(is_64bit_precission<FPT>())
                return _pimpl->underlying_tensor.min().template item<double>();
            else
                throw std::runtime_error("tensor::max: unknown precission");
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator+=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.add_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<FPT,RANK>& add_(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.add_(other.get_tensor());
            return *this;
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator+=(T val) {
            _pimpl->underlying_tensor.add_(val);
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<FPT,RANK>& add_(T val) {
            _pimpl->underlying_tensor.add_(val);
            return *this;
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator-=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.sub_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<FPT,RANK>& sub_(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.sub_(other.get_tensor());
            return *this;
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator-=(T val) {
            _pimpl->underlying_tensor.sub_(val);
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<FPT,RANK>& sub_(T val) {
            _pimpl->underlying_tensor.sub_(val);
            return *this;
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator*=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.mul_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<FPT,RANK>& mul_(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.mul_(other.get_tensor());
            return *this;
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator*=(T val) {
            _pimpl->underlying_tensor.mul_(val);
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<FPT,RANK>& mul_(T val) {
            _pimpl->underlying_tensor.mul_(val);
            return *this;
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator/=(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.div_(other.get_tensor());
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        tensor<FPT,RANK>& div_(const tensor<FPT, R>& other) {
            _pimpl->underlying_tensor.div_(other.get_tensor());
            return *this;
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator/=(T val) {
            _pimpl->underlying_tensor.div_(val);
        }

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<FPT,RANK>& div_(T val) {
            _pimpl->underlying_tensor.div_(val);
            return *this;
        }

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires (device_type_func<F1>() == device_type_func<F2>())
        friend auto operator+(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs);

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires (device_type_func<F1>() == device_type_func<F2>())
        friend auto operator-(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs);

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires (device_type_func<F1>() == device_type_func<F2>())
        friend auto operator*(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs);

        template<device_fp F1, device_fp F2, size_t R1, size_t R2>
        requires (device_type_func<F1>() == device_type_func<F2>())
        friend auto operator/(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs);
        
        template<device_fp F, size_t R, size_t R1, size_t R2>
        requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
        friend tensor<F, R> fftn(const tensor<F, R>& t, span<R1> s, span<R2> dim,
            std::optional<fft_norm> norm);

        template<device_fp F, size_t R, size_t R1, size_t R2>
        requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
        friend tensor<F, R> ifftn(const tensor<F, R>& t, span<R1> s, span<R2> dim,
            std::optional<fft_norm> norm);  

    };

    
    export template<device_fp FPT, size_t RANK>
    class tensor_factory {
    public:

        static tensor<FPT,RANK> make(const std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            return tensor<FPT,RANK>(input_shape, std::move(input));
        }

        static tensor<FPT,RANK> make(span<RANK> input_shape, at::Tensor input) {
            return tensor<FPT,RANK>(input_shape, std::move(input));
        }

        static std::unique_ptr<tensor<FPT, RANK>> make_unique(const std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            return std::make_unique<tensor<FPT, RANK>>(input_shape, std::move(input));
        }

        static std::unique_ptr<tensor<FPT, RANK>> make_unique(span<RANK> input_shape, at::Tensor input) {
            return std::make_unique<tensor<FPT, RANK>>(input_shape, std::move(input));
        }

        static void assign(tensor<FPT, RANK>& t, at::Tensor input) {
            
            if (RANK != input.ndimension())
                throw std::runtime_error("tensor_factory::assign: input.ndimension() did not match RANK");

            if (static_type_to_scalar_type<FPT>() != input.dtype().toScalarType())
                throw std::runtime_error("tensor_factory::assign: input.dtype() did not match FPT");

            if (input.device().is_cuda() && (device_type_func<FPT>() != device_type::CUDA)) {
                throw std::runtime_error("tensor_factory::assign: input.device() was cuda but FPT was not CUDA");
            } else if (input.device().is_cpu() && (device_type_func<FPT>() != device_type::CPU)) {
                throw std::runtime_error("tensor_factory::assign: input.device() was cpu but FPT was not CPU");
            }

            std::array<int64_t, RANK> shape;

            for_sequence<RANK>([&](auto i) {
                shape[i] = input.size(i);
            });

            t.assign(shape, std::move(input));
        }

    };
    
    
    export template<typename T>
    concept is_tensor = requires(T t) {
        []<device_fp FPT, size_t RANK>(tensor<FPT,RANK>&){}(t);
    };


    export enum struct tensor_make_opts {
        EMPTY,
        ONES,
        ZEROS,
        RAND_UNIFORM
    };

    export template<device_fp FP, size_t RANK>
    tensor<FP, RANK> make_tensor(span<RANK> shape, 
        const std::string& device_str="cpu", tensor_make_opts make_opts=tensor_make_opts::EMPTY)
    {
        at::TensorOptions opts = at::TensorOptions(static_type_to_scalar_type<FP>()).device(device_str);

        using TF = tensor_factory<FP, RANK>;

        switch (make_opts) {
            case tensor_make_opts::EMPTY:
                return TF::make(shape, std::move(at::empty(shape.to_arr_ref(), opts)));
            case tensor_make_opts::ONES:
                return TF::make(shape, std::move(at::ones(shape.to_arr_ref(), opts)));
            case tensor_make_opts::ZEROS:
                return TF::make(shape, std::move(at::zeros(shape.to_arr_ref(), opts)));
            case tensor_make_opts::RAND_UNIFORM:
                return TF::make(shape, std::move(at::rand(shape.to_arr_ref(), opts)));
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
 


    export template<device_fp F1, device_fp F2, size_t R1, size_t R2>
    requires (device_type_func<F1>() == device_type_func<F2>())
    auto operator+(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.add(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
    }

    export template<device_fp F1, device_fp F2, size_t R1, size_t R2>
    requires (device_type_func<F1>() == device_type_func<F2>())
    auto operator-(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs)
    {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.sub(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
    }

    export template<device_fp F1, device_fp F2, size_t R1, size_t R2>
    requires (device_type_func<F1>() == device_type_func<F2>())
    auto operator*(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.mul(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
    }

    export template<device_fp F1, device_fp F2, size_t R1, size_t R2>
    requires (device_type_func<F1>() == device_type_func<F2>())
    auto operator/(const tensor<F1, R1>& lhs, const tensor<F2, R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.div(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<nonloss_type_t<F1,F2>, RETRANK>(new_shape, std::move(newtensor));
    }

    export template<device_fp F, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<F, R> fftn(const tensor<F, R>& t,
        span<R1> s,
        span<R2> dim,
        std::optional<fft_norm> norm)
    {
        auto normstr = [&norm]() -> at::optional<c10::string_view> {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return at::optional<c10::string_view>("forward");
                case fft_norm::BACKWARD:
                    return at::optional<c10::string_view>("backward");
                case fft_norm::ORTHO:
                    return at::optional<c10::string_view>("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
            return at::nullopt;
        };
        

        at::Tensor newtensor = torch::fft::fftn(t._pimpl->underlying_tensor,
            s.to_opt_arr_ref(),
            dim.to_opt_arr_ref(),
            normstr()
        );
        return tensor<F, R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

    export template<device_fp F, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<F, R> ifftn(const tensor<F, R>& t,
        span<R1> s,
        span<R2> dim,
        std::optional<fft_norm> norm)
    {
        auto normstr = [&norm]() -> at::optional<c10::string_view> {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return at::optional<c10::string_view>("forward");
                case fft_norm::BACKWARD:
                    return at::optional<c10::string_view>("backward");
                case fft_norm::ORTHO:
                    return at::optional<c10::string_view>("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
            return at::nullopt;
        };

        at::Tensor newtensor = torch::fft::ifftn(t._pimpl->underlying_tensor,
            s.to_opt_arr_ref(),
            dim.to_opt_arr_ref(),
            normstr()
        );
        return tensor<F, R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

}

module;

#include "pch.hpp"

export module tensor;

import util;

namespace hasty {

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
    };

    at::Device get_torch_device(device_idx idx) {
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

    export template<is_device D, is_tensor_type TT, size_t R>
    class tensor;

    template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor_impl {
    public:

        tensor_impl(const std::array<int64_t, RANK>& input_shape, at::Tensor input)
            : shape(input_shape), underlying_tensor(std::move(input))
        {}

        tensor_impl(span<RANK> input_shape, at::Tensor input)
            : shape(input_shape.to_arr()), underlying_tensor(std::move(input))
        {}

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

    export enum struct fft_norm {
        FORWARD,
        BACKWARD,
        ORTHO
    };

    template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
        std::optional<fft_norm> norm);

    template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
        std::optional<fft_norm> norm);


    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor_factory;

    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor {
    private:
        std::shared_ptr<tensor_impl<D,TT,RANK>> _pimpl;
        friend class tensor_factory<D,TT,RANK>;

        device_idx get_device_idx() const {
            return _pimpl->get_device_idx();
        }

        tensor(const std::array<int64_t, RANK>& input_shape, at::Tensor input) 
            : _pimpl(std::make_shared<tensor_impl<D,TT,RANK>>(input_shape, std::move(input)))
        {}

        tensor(span<RANK> input_shape, at::Tensor input)
            : _pimpl(std::make_shared<tensor_impl<D,TT,RANK>>(input_shape, std::move(input)))
        {}

        void assign(std::array<int64_t, RANK>& input_shape, at::Tensor input) {
            _pimpl = std::make_shared<tensor_impl<D,TT,RANK>>(input_shape, std::move(input));
        }

        void assign(span<RANK> input_shape, at::Tensor input) {
            _pimpl = std::make_shared<tensor_impl<D,TT,RANK>>(input_shape, std::move(input));
        }

    public:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, RANK> size = {};

        auto& get_pimpl() { return _pimpl; }
        const auto& get_pimpl() const { return _pimpl; }

        auto& get_tensor() { return _pimpl->underlying_tensor; }
        const auto& get_tensor() const { return _pimpl->underlying_tensor; }

        auto decay_to_tensor() { 
            at::Tensor ten = std::move(_pimpl->underlying_tensor); 
            _pimpl = nullptr;
            return std::move(ten);
        }

        auto ninstances() const {
            return _pimpl.use_count();
        }

        tensor() = default;

        template<is_device DN, is_tensor_type TTN>
        requires (!std::is_same_v<DN, D> || !std::is_same_v<TTN, TT>)
        tensor(tensor<DN, TTN, RANK>&& other, device_idx idx = device_idx::CPU) {
            if (!device_match<D>(idx))
                throw std::runtime_error("tensor::tensor: device mismatch");

            _pimpl = std::move(other._pimpl);
            if (!(std::is_same_v<D, cpu_t> && std::is_same_v<DN, cpu_t>))
                _pimpl->underlying_tensor = _pimpl->underlying_tensor.to(get_torch_device(idx));

            if (!std::is_same_v<TT, TTN>)
                _pimpl->underlying_tensor = _pimpl->underlying_tensor.to(scalar_type_func<TT>());
        }

        base_t<TT> item() const requires (RANK == 0){
            return _pimpl->underlying_tensor.template item<base_t<TT>>();
        }

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

        base_t<TT>* mutable_data_ptr() { return _pimpl->mutable_data_ptr(); }
        const base_t<TT>* const_data_ptr() const { return _pimpl->const_data_ptr();}

        // I know what I am doing...
        base_t<TT>* unconsted_data_ptr() const { return const_cast<base_t<TT>*>(_pimpl->const_data_ptr()); }

        void fill_(base_t<TT> val) {
            if constexpr(!is_fp_complex_tensor_type<TT>) {
                _pimpl->underlying_tensor.fill_(val);
            }
            else {
                auto cten = at::complex(at::scalar_tensor(val.real()), at::scalar_tensor(val.imag())); 
                _pimpl->underlying_tensor.fill_(cten);
            }
        }

        tensor<D, TT, RANK> clone() const {
            at::Tensor newtensor = _pimpl->underlying_tensor.clone();
            return tensor<D, TT, RANK>(_pimpl->shape, std::move(newtensor));
        }

        template<size_t R>
        tensor<D, TT, R> view(span<R> shape) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(shape.to_torch_arr());
            return tensor<D, TT, R>(shape, std::move(tensorview));
        }

        tensor<D, TT, 1> flatview() {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            return tensor<D, TT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        tensor<D, TT, 1> flatslice(index_type auto& idx) {
            at::Tensor tensorview = _pimpl->underlying_tensor.view(-1);
            tensorview = tensorview.index(tch::torchidx(idx));
            return tensor<D, TT, 1>({tensorview.size(0)}, std::move(tensorview));
        }

        tensor<D, TT, RANK> contiguous() {
            return tensor<D, TT, RANK>(_pimpl->shape, _pimpl->underlying_tensor.contiguous());
        }

        void contiguous_() {
            _pimpl->underlying_tensor.contiguous_();
        }

        void assign(at::Tensor in)
        {
            if (RANK != in.ndimension())
                throw std::runtime_error("tensor_factory::assign: input.ndimension() did not match RANK");

            if (scalar_type_func<TT>() != in.dtype().toScalarType())
                throw std::runtime_error("tensor_factory::assign: input.dtype() did not match TT");

            if (in.device().is_cuda() && !std::is_same_v<D, cuda_t>) {
                throw std::runtime_error("tensor_factory::assign: input.device() was cuda but TT was not CUDA");
            } else if (in.device().is_cpu() && !std::is_same_v<D, cpu_t>) {
                throw std::runtime_error("tensor_factory::assign: input.device() was cpu but TT was not CPU");
            }

            std::array<int64_t, RANK> shape;

            for_sequence<RANK>([&](auto i) {
                shape[i] = in.size(i);
            });

            if (_pimpl == nullptr) {
                _pimpl = std::make_shared<tensor_impl<D,TT,RANK>>(shape, std::move(in));
            }
            else {
                _pimpl->shape = shape;
                _pimpl->underlying_tensor = std::move(in);
            }
        }

        template<is_device DN>
        tensor<D, TT, RANK> to(device_idx idx) {
            static_assert(!(std::is_same_v<D, cpu_t> && std::is_same_v<DN, cpu_t>), "don't move from cpu to cpu");

            if (idx == get_device_idx())
                throw std::runtime_error("tensor::to: tensor already on device");

            return tensor<DN,TT,RANK>(_pimpl->underlying_tensor.shape, _pimpl->underlying_tensor.to(get_torch_device(idx)));
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
            return tensor<D, TT, RANK>(new_shape, std::move(tensorview));
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

            return tensor_factory<D, TT, RETRANK>::make(new_shape, std::move(tensorview));
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

            return tensor_factory<D, TT, RETRANK>::make(new_shape, std::move(tensorview));
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator=(const tensor<D, TT, R>& other) {
            _pimpl->underlying_tensor.copy_(other.get_tensor());
        }

        void operator=(const tensor<D, TT, RANK>& other) {
            _pimpl->underlying_tensor.copy_(other.get_tensor());
        }

        void set_(const tensor<D, TT, RANK>& other) {
            if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
                throw std::runtime_error("tensor::set_: other tensor shape did not match this tensor shape");
            _pimpl->underlying_tensor.set_(other._pimpl->underlying_tensor);
        }

        void assign_(tensor<D, TT, RANK>&& other) {
            if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
                throw std::runtime_error("tensor::assign_: other tensor shape did not match this tensor shape");
            _pimpl->underlying_tensor = std::move(other._pimpl->underlying_tensor);
        }

        auto norm() const -> std::enable_if<is_fp_tensor_type<TT>, 
            std::conditional_t<is_fp32_tensor_type<TT>, float, double>>::type {
            
            if constexpr(is_fp32_tensor_type<TT>) {
                return _pimpl->underlying_tensor.norm().template item<float>();
            }
            else if constexpr(is_fp64_tensor_type<TT>) {
                return _pimpl->underlying_tensor.norm().template item<double>();
            }
            else {
                static_assert(false, "tensor::norm: unknown precission");
            }
        }

        auto abs() const {
            if constexpr(is_fp32_tensor_type<TT>) {
                auto ret = _pimpl->underlying_tensor.abs().to(at::ScalarType::Float);
                return tensor<D, f32_t, RANK>(_pimpl->shape, std::move(ret));
            }
            else if constexpr(is_fp64_tensor_type<TT>) {
                auto ret = _pimpl->underlying_tensor.abs().to(at::ScalarType::Double);
                return tensor<D, f64_t, RANK>(_pimpl->shape, std::move(ret));
            }
            else if constexpr(is_int_tensor_type<TT>) {
                auto ret = _pimpl->underlying_tensor.abs().to(scalar_type_func<TT>());
                return tensor<D, TT, RANK>(_pimpl->shape, std::move(ret));
            }
            else {
                static_assert(false, "tensor::abs: unknown precission");
            }
        }

        auto max() const {
            return _pimpl->underlying_tensor.max().template item<base_t<TT>>();
        }

        auto min() const {
            return _pimpl->underlying_tensor.min().template item<base_t<TT>>();
        }

        auto real() const  {
            if constexpr(is_fp_complex_tensor_type<TT>) {
                auto ret = _pimpl->underlying_tensor.real();
                return tensor<D, real_t<TT>, RANK>(_pimpl->shape, std::move(ret));
            }
            else {
                return clone();
            }
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator+=(const tensor<D,TT,R>& other) {
            _pimpl->underlying_tensor.add_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& add_(const tensor<D,TT,R>& other) {
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
        tensor<D,TT,RANK>& add_(T val) {
            _pimpl->underlying_tensor.add_(val);
            return *this;
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator-=(const tensor<D,TT,R>& other) {
            _pimpl->underlying_tensor.sub_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& sub_(const tensor<D,TT,R>& other) {
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
        tensor<D,TT,RANK>& sub_(T val) {
            _pimpl->underlying_tensor.sub_(val);
            return *this;
        }

        template<size_t R>
        requires less_than<R, RANK>
        void operator*=(const tensor<D,TT,R>& other) {
            _pimpl->underlying_tensor.mul_(other.get_tensor());
        }

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& mul_(const tensor<D,TT,R>& other) {
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
        tensor<D,TT,RANK>& mul_(T val) {
            _pimpl->underlying_tensor.mul_(val);
            return *this;
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator/=(const tensor<D,TT,R>& other) {
            _pimpl->underlying_tensor.div_(other.get_tensor());
        }

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        tensor<D,TT,RANK>& div_(const tensor<D,TT,R>& other) {
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
        tensor<D,TT,RANK>& div_(T val) {
            _pimpl->underlying_tensor.div_(val);
            return *this;
        }

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator+(const tensor<D1,TT1,R1>& lhs, const tensor<D,TT2,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator-(const tensor<D1,TT,R1>& lhs, const tensor<D,TT,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator*(const tensor<D1,TT,R1>& lhs, const tensor<D,TT,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator/(const tensor<D1,TT,R1>& lhs, const tensor<D,TT,R2>& rhs);
        
        template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
        requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
        friend tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
            opt<fft_norm> norm);

        template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
        requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
        friend tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
            opt<fft_norm> norm);  

        template<is_device D1, is_tensor_type TT1, size_t R>
        friend tensor<D,TT1,0> vdot(const tensor<D1,TT1,R>& lhs, const tensor<D1,TT1,R>& rhs);

    };
    
    
    export template<typename T>
    concept is_tensor = requires(T t) {
        []<is_device D2, is_tensor_type TT2, size_t RANK2>(tensor<D2,TT2,RANK2>&){}(t);
    };


    export enum struct tensor_make_opts {
        EMPTY,
        ONES,
        ZEROS,
        RAND_UNIFORM
    };

    export template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK> make_tensor(span<RANK> shape, 
        const std::string& device_str="cpu", tensor_make_opts make_opts=tensor_make_opts::EMPTY)
    {
        at::TensorOptions opts = at::TensorOptions(scalar_type_func<TT>()).device(device_str);

        using TF = tensor_factory<D,TT,RANK>;

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

    export template<is_device D, is_tensor_type TT, size_t RANK>
    std::unique_ptr<tensor<D,TT,RANK>> make_tensor(at::Tensor tensorin)
    {
        if (tensorin.ndimension() != RANK)
            throw std::runtime_error("make_tensor: tensor.ndimension() did not match RANK");

        if (tensorin.dtype().toScalarType() != scalar_type_func<TT>())
            throw std::runtime_error("make_tensor: tensor.dtype() did not match templated any_fp FP");

        struct creator : tensor<D,TT,RANK> {
            creator(std::initializer_list<int64_t> a, at::Tensor b)
                : tensor<D,TT,RANK>(a, std::move(b)) {}
        };

        return std::make_unique<creator>(tensorin.sizes(), std::move(tensorin));
    }
 


    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator+(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.add(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator-(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs)
    {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.sub(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator*(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.mul(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
    auto operator/(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs) {
        constexpr size_t RETRANK = R1 > R2 ? R1 : R2;

        at::Tensor newtensor = lhs._pimpl->underlying_tensor.div(rhs._pimpl->underlying_tensor);
        std::array<int64_t, RETRANK> new_shape;

        assert(newtensor.ndimension() == RETRANK);

        for_sequence<RETRANK>([&](auto i) {
            new_shape[i] = newtensor.size(i);
        });

        return tensor<D1,nonloss_type_t<TT1,TT2>,RETRANK>(new_shape, std::move(newtensor));
    }

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t,
        span<R1> s,
        span<R2> dim,
        opt<fft_norm> norm)
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
        return tensor<D1,TT1,R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }

    export template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
    requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
    tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t,
        span<R1> s,
        span<R2> dim,
        opt<fft_norm> norm)
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
        return tensor<D1,TT1,R>(span<R>(newtensor.sizes()), std::move(newtensor));
    }


    template<is_device D1, is_tensor_type TT1, size_t R>
    tensor<D1,TT1,0> vdot(const tensor<D1,TT1,R>& lhs, const tensor<D1,TT1,R>& rhs) 
    {
        at::Tensor newtensor = at::vdot(lhs._pimpl->underlying_tensor.flatten(), rhs._pimpl->underlying_tensor.flatten());
        return tensor<D1,TT1,0>({}, std::move(newtensor));
    }







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

        /*
        template<is_device DN>
        static auto move_device(uptr<tensor<D,TT,RANK>> t, device_idx idx, bool block=true) -> 
            uptr<tensor<DN, RANK>> {

            if (idx == device_idx::CPU) {
                if (device_type_func<FPT>() == device_type::CPU) {
                    return std::move(t);
                }
                else {
                    auto newtensor = t->get_tensor().to(at::kCPU);
                    t = nullptr;
                    return std::make_unique<tensor<swap_device_t<FPT>, RANK>>(t->get_pimpl()->shape, std::move(newtensor));
                }
            }
            else {
                if (device_type_func<FPT>() == device_type::CUDA) {
                    auto newtensor = t->get_tensor().to(at::kCUDA, idx);
                    t = nullptr;
                    return std::make_unique<tensor<swap_device_t<FPT>, RANK>>(t->get_pimpl()->shape, std::move(newtensor));
                }
                else {
                    throw std::runtime_error("tensor_factory::move_to: unknown device");
                }
            }
        }
        */

    };












    export template<is_device D, is_tensor_type TT, size_t RANK>
    class cache_tensor {
    private:

        struct tensor_holder_cuda {
            std::unique_ptr<tensor<cuda_t,TT,RANK>> tensor_cuda;
            std::unique_ptr<tensor<cpu_t,TT,RANK>> tensor_cpu;
        };

        struct tensor_holder_cpu {
            std::unique_ptr<tensor<cpu_t,TT,RANK>> tensor_cpu;
        };

        std::conditional_t<std::is_same_v<D,cuda_t>, 
            tensor_holder_cuda, tensor_holder_cpu> _tensor_holder;

        size_t hashidx;
        static std::filesystem::path cache_dir;
        std::array<i64, RANK> shape;


        auto get_cuda_ptr(device_idx idx) -> std::enable_if<std::is_same_v<D,cuda_t>, uptr<tensor<cuda_t,TT,RANK>>>::type& {
            if (!_tensor_holder.tensor_cuda) {
                _tensor_holder.tensor_cuda = std::make_unique<tensor<cuda_t,TT,RANK>>(std::move(_tensor_holder.get_cpu_ptr()), idx);
                return _tensor_holder.tensor_cuda;
            }
            else if (_tensor_holder.tensor_cuda.get_device_idx() != idx) {
                _tensor_holder.tensor_cuda = std::make_unique<tensor<cuda_t,TT,RANK>>(std::move(_tensor_holder.tensor_cuda), idx);
                return _tensor_holder.tensor_cuda;
            } else {
                return _tensor_holder.tensor_cuda;
            }
        }

        auto get_cpu_ptr() -> uptr<tensor<cpu_t,TT,RANK>>&  {
            if (_tensor_holder.tensor_cpu)
                return _tensor_holder.tensor_cpu;

            if constexpr(std::is_same_v<D,cuda_t>) {
                if (_tensor_holder.tensor_cuda) {
                    _tensor_holder.tensor_cpu = std::make_unique<cpu_t,TT,RANK>(std::move(_tensor_holder.tensor_cuda));
                    return _tensor_holder.tensor_cpu;
                }
            }

            _tensor_holder.tensor_cpu = load_from_disk();

            return _tensor_holder.tensor_cpu;
        }

        void cache_disk() {
            if constexpr(std::is_same_v<D,cuda_t>) {
                if (_tensor_holder.tensor_cuda) {
                    _tensor_holder.tensor_cpu = std::make_unique<cpu_t,TT,RANK>(std::move(_tensor_holder.tensor_cuda));
                    return _tensor_holder.tensor_cpu;
                }
            }

            if (!_tensor_holder.tensor_cpu)
                throw std::runtime_error("cache_disk: no tensor to cache");

            auto tt = _tensor_holder.tensor_cpu.get_tensor().contiguous();


            try {
                namespace fs = std::filesystem;
                std::ofstream ofs(cache_dir / fs::path(std::to_string(hashidx) + ".htc"), std::ios::binary | std::ios::out);
                if (!ofs.is_open())
                    throw std::runtime_error("cache_disk: could not open file for writing");

                ofs.write(reinterpret_cast<char*>(tt.data_ptr()), tt.numel() * sizeof(base_t<TT>));
                ofs.close();
            } catch (std::ifstream::failure e) {
                std::cerr << "cache_disk: Exception opening/reading/closing file\n";
                throw;
            } catch (...) {
                throw;
            }
        }

        uptr<tensor<cpu_t,TT,RANK>> load_from_disk() {
            namespace fs = std::filesystem;
            auto tt = tensor_factory<cpu_t,TT,RANK>::make(shape, at::empty(span(shape), at::kCPU));
            
            try {
                std::ifstream ifs(cache_dir / fs::path(std::to_string(hashidx) + ".htc"), std::ios::binary | std::ios::in);
                if (!ifs.is_open())
                    throw std::runtime_error("load_from_disk: could not open file for reading");
                    
                ifs.read(reinterpret_cast<char*>(tt->mutable_data_ptr()), tt->nelem() * sizeof(base_t<TT>));
                ifs.close();
            } catch (std::ostream::failure e) {
                std::cerr << "load_from_disk: Exception opening/writing/closing file\n";
                throw;
            } catch (...) {
                throw;
            }

            return tt;
        }


    public:
        
        std::enable_if<std::is_same_v<D,cuda_t>, tensor<cuda_t,TT,RANK>>::type& get_cuda(device_idx idx) {
            return *get_cuda_ptr(idx);
        }

        tensor<cpu_t,TT,RANK>& get_cpu() {
            return *get_cpu_ptr();
        }

        void cache() {
            cache_disk();
        }

        void uncache() {
            _tensor_holder.tensor_cpu = load_from_disk();
        }

    };


}

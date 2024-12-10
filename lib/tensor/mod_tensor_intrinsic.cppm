module;

#include "pch.hpp"

export module tensor:intrinsic;

//import pch;

import util;
export import :base;

namespace hasty {


    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor {
    private:

        struct tensor_base {

            tensor_base(const std::array<int64_t, RANK>& input_shape, TensorBackend input);

            tensor_base(span<RANK> input_shape, TensorBackend input);

            ~tensor_base();

            base_t<TT>* mutable_data_ptr();

            const base_t<TT>* const_data_ptr() const;

            device_idx get_device_idx() const;

            std::array<int64_t, RANK> shape;
            TensorBackend underlying_tensor;
        };

        std::shared_ptr<tensor_base> _pimpl;

        template<is_device DO1, is_tensor_type TTO1, size_t RO1>
        friend class tensor;

    public:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, RANK> size = {};

        auto& get_pimpl();
        const auto& get_pimpl() const;

        auto get_tensor() -> TensorBackend&;
        auto get_tensor() const -> const TensorBackend&;

        auto decay_to_tensor();

        size_t ninstances() const;

        std::string str() const;

        base_t<TT> item() const requires (RANK == 0);

        template<size_t R>
        requires less_than<R, RANK>
        int64_t shape() const;

        std::array<int64_t, RANK> shape() const;

        constexpr int64_t ndim() const;

        int64_t numel() const;

        std::string devicestr() const;

        device_idx get_device_idx() const;

        base_t<TT>* mutable_data_ptr();
        const base_t<TT>* const_data_ptr() const;

        // I know what I am doing...
        base_t<TT>* unconsted_data_ptr() const;

        tensor<D, TT, RANK> clone() const;

        tensor<D, TT, RANK+1> unsqueeze(int64_t dim);

        template<size_t R>
        tensor<D, TT, R> view(span<R> shape);

        tensor<D, TT, 1> flatview();

        tensor<D, TT, 1> flatslice(index_type auto& idx);

        tensor<D, TT, RANK> contiguous();

        template<is_device DN>
        tensor<DN, TT, RANK> to(device_idx idx);

        template<is_tensor_type TTN>
        tensor<D,TTN,RANK> to();

        tensor<D, TT, 1> masked_select(const tensor<D,b8_t,RANK>& mask) const;

        void assign(std::array<int64_t, RANK>& input_shape, TensorBackend input);

        void assign(span<RANK> input_shape, TensorBackend input);



        tensor();

        //tensor(const tensor&) = delete;
        tensor(const tensor& other);

        //tensor(tensor&&) = delete;
        tensor(tensor&& other);

        template<is_tensor_type TTN>
        tensor(tensor<D, TTN, RANK>&& other);

        template<is_device DN, is_tensor_type TTN>
        requires (!std::is_same_v<DN, D> || !std::is_same_v<TTN, TT>)
        tensor(tensor<DN, TTN, RANK>&& other, device_idx idx);

        tensor(const std::array<int64_t, RANK>& input_shape, TensorBackend input);

        tensor(span<RANK> input_shape, TensorBackend input);




        ~tensor();

        // <============= OPERATORS =============>

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D, TT, RANK>& operator=(const tensor<D, TT, R>& other);

        tensor<D, TT, RANK>& operator=(const tensor<D, TT, RANK>& other);

        tensor<D,TT,RANK>& operator=(move<tensor<D,TT,RANK>>&& other);

        tensor<D, TT, RANK>& operator=(tensor<D,TT,RANK>&& other);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator+(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator-(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator*(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

        template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
        friend auto operator/(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator+=(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator+=(T val);

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator-=(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator-=(T val);

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator*=(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator*=(T val);

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        void operator/=(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        void operator/=(T val);

        /* must return a view */
        template<size_t N>
        requires less_than_or_equal<N, RANK> && less_than<0,RANK>
        auto operator[](const std::array<Slice, N>& slices) const -> tensor<D, TT, RANK>;

        /* must return a view */
        template<index_type ...Idx>
        requires less_than<0,RANK>
        auto operator[](std::tuple<Idx...> indices) const;

        /* must return a view */
        template<index_type ...Idx>
        requires less_than<0,RANK>
        auto operator[](Idx... indices) const;

        /* will not return a view */
        auto operator[](const tensor<D,b8_t,RANK>& mask) const -> tensor<D,TT,1>;


        // <=============== INPLACE OPERATIONS ================>


        template<is_device DN>
        tensor<D, TT, RANK>& copy_(tensor<DN,TT,RANK>& other, bool non_blocking=false);

        template<is_device DN>
        tensor<D, TT, RANK>& copy_(tensor<DN,TT,RANK>&& other, bool non_blocking=false);

        void fill_(base_t<TT> val);

        void contiguous_();

        void set_(const tensor<D, TT, RANK>& other);

        void assign_(tensor<D, TT, RANK>&& other);

        void masked_scatter_(const tensor<D,b8_t,RANK>& mask, const tensor<D, TT, 1>& src);

        void masked_fill_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val);

        void masked_add_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val);

        void masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src);

        void masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src, base_t<TT> alpha);

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& add_(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<D,TT,RANK>& add_(T val);

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& sub_(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<D,TT,RANK>& sub_(T val);

        template<size_t R>
        requires less_than<R, RANK>
        tensor<D,TT,RANK>& mul_(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<D,TT,RANK>& mul_(T val);

        template<size_t R>
        requires less_than_or_equal<R, RANK>
        tensor<D,TT,RANK>& div_(const tensor<D,TT,R>& other);

        template<typename T>
        requires std::integral<T> || std::floating_point<T>
        tensor<D,TT,RANK>& div_(T val);
        
        // <================= MATH OPERATIONS ===============>

        auto norm() const -> std::conditional_t<is_fp32_tensor_type<TT>, float, double> 
        requires is_fp_tensor_type<TT>;

        auto abs() const;

        auto max() const;

        auto min() const;

        auto real() const;


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

    /*
    export template<typename T>
    concept is_tensor = requires(T t) {
        typename T::device_type_t;
        typename T::tensor_type_t;
        { T::size() } -> std::convertible_to<size_t>;
        requires std::is_same_v<T,
            tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()>
        >;
    };
    */

    export template<is_tensor_type TT>
    class scalar {
    public:

        scalar(base_t<TT> val) : _val(val) {}

        template<is_device D>
        tensor<D, TT, 0> to_tensor() {
            return tensor<D, TT, 0>({}, at::scalar_tensor(_val));
        }

    private:
        base_t<TT> _val;
    };
 
    // Define the is_vector_of_tensors concept
    export template<typename T>
    concept is_vector_of_tensors = requires(T t) {
        typename T::value_type;
        requires is_std_vector_v<T>;
        requires is_tensor<typename T::value_type>;
    };

    export template<typename T>
    concept is_tensor_or_vector_of_tensors = is_tensor<T> || is_vector_of_tensors<T>;


}




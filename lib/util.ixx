module;

#include "pch.hpp"

export module util;

namespace hasty {

    template <typename T, typename = void>
    struct has_strong_value : std::false_type{};

    template <typename T>
    struct has_strong_value<T, decltype((void)T::strong_value, void())> : std::true_type {};

    struct strong_typedef_base {};

    template<typename T>
    concept is_strong_type = std::is_base_of_v<strong_typedef_base, T> && has_strong_value<T>::value;

    // export from here
    export {

        using i16 = int16_t;
        using i32 = int32_t;
        using i64 = int64_t;
        using u16 = uint16_t;
        using u32 = uint32_t;
        using u64 = uint64_t;
        using f32 = float;
        using f64 = double;
        using c64 = std::complex<float>;
        using c128 = std::complex<double>;


        template<size_t T1, size_t T2>
        concept less_than = T1 < T2;

        template<size_t T1, size_t T2>
        concept less_than_or_equal = T1 <= T2;

        template<size_t T1, size_t T2>
        concept equal_or_one_zero = !((T1 != 0) && (T2 != 0) && (T1 != T2));

        template<size_t T1, size_t T2>
        concept equal_or_right_zero = (T1 == T2) || (T2 == 0);

        template<size_t T1, size_t T2>
        concept equal_or_left_zero = (T1 == T2) || (T1 == 0);

        template<std::integral T, size_t N>
        struct arbspan {

            //nullspan
            arbspan() : _data(nullptr) {};

            arbspan(T const (&list)[N]) 
                : _data(list) {}

            arbspan(const T* listptr)
                : _data(listptr) {}

            arbspan(at::ArrayRef<T> arr)
                : _data(arr.data())
            {}

            arbspan(const std::array<T, N>& arr)
                : _data(arr.data())
            {}

            /*
            span(std::span<const T, N> span) 
                : _data(span.data()) {}
            */
            at::ArrayRef<T> to_arr_ref() {
                return at::ArrayRef<T>(_data, N);
            }

            std::array<T, N> to_arr() {
                std::array<T, N> arr;
                for (size_t i = 0; i < N; i++) {
                    arr[i] = _data[i];
                }
                return arr;
            }

            arbspan(std::nullopt_t)
                : _data(nullptr) {}

            const T& operator[](size_t index) const {
                if (index >= N) {
                    throw std::out_of_range("Index out of range");
                }
                return _data[index];
            }

            template<size_t I>
            requires less_than<I,N>
            const T& get() {
                return _data[I];
            }

            constexpr size_t size() const { return N; }

            bool has_value() const {
                return _data != nullptr;
            }

        private:
            const T* _data;
        };
        
        template<std::integral I, size_t R>
        constexpr std::string span_to_str(arbspan<I,R> arr, bool as_tuple = true) {
            std::string retstr = as_tuple ? "(" : "[";
            
            for_sequence<R>([&](auto i) {
                retstr += std::to_string(arr.template get<i>());
                if constexpr(i < R - 1) {
                    retstr += ",";
                }
            });
            retstr += as_tuple ? ")" : "]";
            return retstr;
        }

        template<size_t N>
        struct span {

            //nullspan
            span() : _data(nullptr) {};

            span(i64 const (&list)[N]) 
                : _data(list) {}

            span(const i64* listptr)
                : _data(listptr) {}

            span(at::ArrayRef<i64> arr)
                : _data(arr.data())
            {}

            span(const std::array<i64, N>& arr)
                : _data(arr.data())
            {}

            at::ArrayRef<i64> to_arr_ref() {
                return at::ArrayRef<i64>(_data, N);
            }

            std::array<i64, N> to_arr() {
                std::array<i64, N> arr;
                for (size_t i = 0; i < N; i++) {
                    arr[i] = _data[i];
                }
                return arr;
            }

            span(std::nullopt_t)
                : _data(nullptr) {}

            const i64& operator[](size_t index) const {
                if (index >= N) {
                    throw std::out_of_range("Index out of range");
                }
                return _data[index];
            }

            template<size_t I>
            requires less_than<I,N>
            const i64& get() {
                return _data[I];
            }

            constexpr size_t size() const { return N; }

            bool has_value() const {
                return _data != nullptr;
            }

        private:
            const i64* _data;
        };

        using nullspan = span<0>;

        template<size_t R>
        constexpr std::string span_to_str(span<R> arr, bool as_tuple = true) {
            std::string retstr = as_tuple ? "(" : "[";
            
            for_sequence<R>([&](auto i) {
                retstr += std::to_string(arr.template get<i>());
                if constexpr(i < R - 1) {
                    retstr += ",";
                }
            });
            retstr += as_tuple ? ")" : "]";
            return retstr;
        }


        enum struct device_type {
            CPU,
            CUDA
        };

        enum struct precission {
            F32,
            F64
        };

        enum struct complexity {
            REAL,
            COMPLEX
        };

        

        template<typename T, typename U>
        struct strong_typedef : public strong_typedef_base {

            strong_typedef() = default;

            T strong_value;
        };

        template<is_strong_type T>
        using underlying_type = decltype(T::strong_value);
        

        using cpu_f32 = strong_typedef<float, struct cpu_f32_>;
        using cpu_f64 = strong_typedef<double, struct cpu_f64_>;
        using cpu_c64 = strong_typedef<std::complex<float>, struct cpu_c64_>;
        using cpu_c128 = strong_typedef<std::complex<double>, struct cpu_c128_>;

        template<typename T>
        concept cpu_real_fp = std::is_same_v<cpu_f32, T> || std::is_same_v<cpu_f64, T>;

        template<typename T>
        concept cpu_complex_fp = std::is_same_v<cpu_c64, T> || std::is_same_v<cpu_c128, T>;

        template<typename T>
        concept cpu_fp = requires {cpu_complex_fp<T> || cpu_real_fp<T>;};

        
        using cuda_f32 = strong_typedef<float, struct cuda_f32_>;
        using cuda_f64 = strong_typedef<double, struct cuda_f64_>;
        using cuda_c64 = strong_typedef<cuFloatComplex, struct cuda_c64_>;
        using cuda_c128 = strong_typedef<cuDoubleComplex, struct cuda_c128_>;

        template<typename T>
        concept cuda_real_fp = std::is_same_v<cuda_f32, T> || std::is_same_v<cuda_f64, T>;

        template<typename T>
        concept cuda_complex_fp = std::is_same_v<cuda_c64, T> || std::is_same_v<cuda_c128, T>;

        template<typename T>
        concept cuda_fp = requires {cuda_complex_fp<T> || cuda_real_fp<T>;};



        template<typename T>
        concept device_real_fp = requires { requires cpu_real_fp<T> || cuda_real_fp<T>;};

        template<typename T>
        concept device_complex_fp = requires { cuda_complex_fp<T> || cpu_complex_fp<T>;};

        template<typename T>
        concept device_fp = requires { cuda_fp<T> || cpu_fp<T>; };

        template<device_fp F>
        constexpr auto swap_device_type_func() {
            if constexpr (std::is_same_v<F, cpu_f32>) {
                return cuda_f32();
            } else if constexpr(std::is_same_v<F,cpu_f64>) {
                return cuda_f64();
            } else if constexpr(std::is_same_v<F,cpu_c64>) {
                return cuda_c64();
            } else if constexpr(std::is_same_v<F,cpu_c128>) {
                return cuda_c128();
            } else if constexpr(std::is_same_v<F,cuda_f32>) {
                return cpu_f32();
            } else if constexpr(std::is_same_v<F,cuda_f64>) {
                return cpu_f64();
            } else if constexpr(std::is_same_v<F,cuda_c64>) {
                return cpu_c64();
            } else if constexpr(std::is_same_v<F, cuda_c128>) {
                return cpu_c128();
            }
        }

        template<cuda_fp F>
        using cpu_t = decltype(swap_device_type_func<F>());

        template<cpu_fp F>
        using cuda_t = decltype(swap_device_type_func<F>());

        template<device_fp F>
        using swap_device_t = decltype(swap_device_type_func<F>());


        template<device_fp F>
        constexpr auto swap_complex_real_type_func() {
            if constexpr (std::is_same_v<F, cpu_f32>) {
                return cpu_c64();
            } else if constexpr(std::is_same_v<F,cpu_f64>) {
                return cpu_c128();
            } else if constexpr(std::is_same_v<F,cpu_c64>) {
                return cpu_f32();
            } else if constexpr(std::is_same_v<F,cpu_c128>) {
                return cpu_f64();
            } else if constexpr(std::is_same_v<F,cuda_f32>) {
                return cuda_c64();
            } else if constexpr(std::is_same_v<F,cuda_f64>) {
                return cuda_c128();
            } else if constexpr(std::is_same_v<F,cuda_c64>) {
                return cuda_f32();
            } else if constexpr(std::is_same_v<F, cuda_c128>) {
                return cpu_f64();
            }
        }

        template<device_real_fp F>
        using complex_t = decltype(swap_complex_real_type_func<F>());

        template<device_complex_fp F>
        using real_t = decltype(swap_complex_real_type_func<F>());

        template<device_fp F>
        constexpr bool is_cuda() 
        { 
            return 
                std::is_same_v<F, cuda_f32> || 
                std::is_same_v<F, cuda_f64> || 
                std::is_same_v<F, cuda_c64> ||
                std::is_same_v<F, cuda_c128>;
        }

        template<device_fp F>
        constexpr bool is_cpu()
        {
            return 
                std::is_same_v<F, cpu_f32> || 
                std::is_same_v<F, cpu_f64> || 
                std::is_same_v<F, cpu_c64> ||
                std::is_same_v<F, cpu_c128>;
        }

        template<device_fp F>
        constexpr bool is_real()
        {
            return
                std::is_same_v<F, cpu_f32> ||
                std::is_same_v<F, cpu_f64> ||
                std::is_same_v<F, cuda_f32> ||
                std::is_same_v<F, cuda_f64>;
        }

        template<device_fp F>
        constexpr bool is_complex()
        {
            return
                std::is_same_v<F, cpu_c64> ||
                std::is_same_v<F, cpu_c128> ||
                std::is_same_v<F, cuda_c64> ||
                std::is_same_v<F, cuda_c128>;
        }

        template<device_fp F>
        constexpr bool is_32bit_precission()
        {
            return
                std::is_same_v<F, cpu_f32> ||
                std::is_same_v<F, cuda_f32> ||
                std::is_same_v<F, cpu_c64> ||
                std::is_same_v<F, cuda_c64>;
        }

        template<device_fp F>
        constexpr bool is_64bit_precission()
        {
            return
                std::is_same_v<F, cpu_f64> ||
                std::is_same_v<F, cuda_f64> ||
                std::is_same_v<F, cpu_c128> ||
                std::is_same_v<F, cuda_c128>;
        }

        template<device_fp FP>
        constexpr at::ScalarType static_type_to_scalar_type()
        {
            if constexpr(std::is_same_v<FP, cpu_f32> || std::is_same_v<FP, cuda_f32>) {
                return at::ScalarType::Float;
            }
            else if constexpr(std::is_same_v<FP, cpu_f64> || std::is_same_v<FP, cuda_f64>) {
                return at::ScalarType::Double;
            }
            else if constexpr(std::is_same_v<FP, cpu_c64> || std::is_same_v<FP, cuda_c64>) {
                return at::ScalarType::ComplexFloat;
            }
            else if constexpr(std::is_same_v<FP, cpu_c128> || std::is_same_v<FP, cuda_c128>) {
                return at::ScalarType::ComplexDouble;
            }
        }

        template<device_fp FP>
        constexpr device_type device_type_func()
        {
            if constexpr(is_cuda<FP>()) {
                return device_type::CUDA;
            } else if constexpr(is_cpu<FP>()) {
                return device_type::CPU;
            } else {
                static_assert(false, "Invalid device type");
            }
        }

        template<device_fp F1, device_fp F2>
        constexpr auto nonloss_type_func() {
            auto types = []<device_fp FP1, device_fp FP2>() -> bool {
                return std::is_same_v<FP1, F1> && std::is_same_v<FP2, F2> ||
                    std::is_same_v<FP1, F2> && std::is_same_v<FP2, F1>;
            };

            static_assert(device_type_func<F1>() == device_type_func<F2>(), "Types must have same device type");

            if constexpr(std::is_same_v<F1,F2>) {
                return F1();
            } else if constexpr(types.template operator()<cpu_f32, cpu_f64>()) {
                return cpu_f64();
            } else if constexpr(types.template operator()<cpu_f32, cpu_c128>()) {
                return cpu_c128();
            } else if constexpr(types.template operator()<cpu_f64, cpu_c64>()) {
                return cpu_c128();
            } else if constexpr(types.template operator()<cpu_f64, cpu_c128>()) {
                return cpu_c128();
            } else if constexpr(types.template operator()<cpu_c64, cpu_c128>()) {
                return cpu_c128();
            } else if constexpr(types.template operator()<cuda_f32, cuda_f64>()) {
                return cuda_f64();
            } else if constexpr(types.template operator()<cuda_f32, cuda_c64>()) {
                return cuda_c64();
            } else if constexpr(types.template operator()<cuda_f32, cuda_c128>()) {
                return cuda_c128();
            } else if constexpr(types.template operator()<cuda_f64, cuda_c64>()) {
                return cuda_c128();
            } else if constexpr(types.template operator()<cuda_f64, cuda_c128>()) {
                return cuda_c128();
            } else if constexpr(types.template operator()<cuda_c64, cuda_c128>()) {
                return cuda_c128();
            } else {
                static_assert(false, "Invalid types");
            }
        }

        template<device_fp F1, device_fp F2>
        using nonloss_type_t = decltype(nonloss_type_func<F1, F2>());

        static_assert(alignof(cuda_f32) == alignof(float));
        static_assert(sizeof(cuda_f32) == sizeof(float));
        static_assert(alignof(cuda_f64) == alignof(double));
        static_assert(sizeof(cuda_f64) == sizeof(double)); 
        static_assert(alignof(cuda_c64) == alignof(cuFloatComplex));
        static_assert(sizeof(cuda_c64) == sizeof(cuFloatComplex));
        static_assert(alignof(cuda_c128) == alignof(cuDoubleComplex));
        static_assert(sizeof(cuda_c128) == sizeof(cuDoubleComplex));


        template<typename T>
        using optrefw = std::optional<std::reference_wrapper<T>>;

        template <typename T, T... S, typename F>
        constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
            (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
        }

        template<auto n, typename F>
        constexpr void for_sequence(F f) {
            for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
        }

        template<auto n, typename F, typename V>
        constexpr V for_sequence(F f, const V& t) {
            V tcopy = t;
            for_sequence<n>([&tcopy, &f](auto i) {
                f(i, tcopy);
            });
            return tcopy;
        }

    }


    
    export struct None {};

    export struct Ellipsis {};

    export struct Slice {

        Slice() = default;

        template<std::integral I>
        Slice(I ival) : start(ival) {}

        template<std::integral I1, std::integral I2>
        Slice(I1 ival1, I2 ival2) : start(ival1), end(ival2) {}

        std::optional<int64_t> start;
        std::optional<int64_t> end;
        std::optional<int64_t> step;
    };

    export template<typename T>
    concept index_type =   std::is_same_v<T, None> 
                        || std::is_same_v<T, Ellipsis> 
                        || std::is_same_v<T, Slice>
                        || std::is_integral_v<T>;

    export using TensorIndex = std::variant<None, Ellipsis, Slice, int64_t>;

    export template<size_t R, index_type... Idx>
    constexpr size_t get_slice_rank()
    {
        int none = 0;
        int ints = 0;
        int ellipsis = 0;

        ((std::is_same_v<Idx, None> ? ++none : 
        std::is_integral_v<Idx> ? ++ints : 
        std::is_same_v<Idx, Ellipsis> ? ++ellipsis : 0), ...);

        return R - ints + none;
    }

    export template<size_t R, index_type... Idx>
    constexpr size_t get_slice_rank(std::tuple<Idx...> idxs)
    {
        int none;
        int ints;
        int ellipsis;

        for_sequence<std::tuple_size_v<decltype(idxs)>>([&](auto i) constexpr {
            //if constexpr(std::is_same_v<decltype(idxs.template get<i>()), None>) {
            if constexpr(std::is_same_v<decltype(std::get<i>(idxs)), None>) {
                ++none;
            } 
            //else if constexpr(std::is_integral_v<decltype(idxs.template get<i>())>) {
            else if constexpr(std::is_integral_v<decltype(std::get<i>(idxs))>) {
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

    export template<size_t R, index_type... Itx>
    constexpr size_t get_slice_rank(Itx... idxs)
    {
        return get_slice_rank<R>(std::make_tuple(idxs...));
    }

    export template<typename T>
    c10::optional<T> torch_optional(const std::optional<T>& opt)
    {
        if (opt.has_value()) {
            return c10::optional(opt.value());
        }
        return c10::nullopt;
    }

    export template<typename R, typename T>
    c10::optional<R> torch_optional(const std::optional<T>& opt)
    {
        if (opt.has_value()) {
            return c10::optional<R>(opt.value());
        }
        return c10::nullopt;
    }

    export template<index_type Idx>
    at::indexing::TensorIndex torchidx(Idx idx) {
        if constexpr(std::is_same_v<Idx, None>) {
            return at::indexing::None;
        } 
        else if constexpr(std::is_same_v<Idx, Ellipsis>) {
            return at::indexing::Ellipsis;
        }
        else if constexpr(std::is_same_v<Idx, Slice>) {
            return at::indexing::Slice(
                torch_optional<c10::SymInt>(idx.start),
                torch_optional<c10::SymInt>(idx.end),
                torch_optional<c10::SymInt>(idx.step));
        } else if constexpr(std::is_integral_v<Idx>) {
            return idx;
        } else {
            static_assert(false);
        }
    }

    template<index_type... Idx, size_t... Is>
    auto torchidx_impl(std::tuple<Idx...> idxs, std::index_sequence<Is...>) {
        return std::array<at::indexing::TensorIndex, sizeof...(Idx)>{torchidx(std::get<Is>(idxs))...};
    }

    export template<index_type... Idx>
    auto torchidx(std::tuple<Idx...> idxs) {
        return torchidx_impl(idxs, std::make_index_sequence<sizeof...(Idx)>{});
    }

    export template<index_type Idx>
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



}
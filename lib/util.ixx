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

    export {
        // export from here
        template<size_t T1, size_t T2>
        concept less_than = T1 < T2;
        
        enum struct device_type {
            CPU,
            CUDA
        };

        template<typename T, typename U>
        struct strong_typedef : public strong_typedef_base {
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

        template<cpu_real_fp T>
        using complex_cpu_t = std::conditional_t<std::is_same_v<T,cpu_f32>, cpu_c64, cpu_c128>;

        
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

        template<cuda_real_fp T>
        using complex_cuda_t = std::conditional_t<std::is_same_v<T,cuda_f32>, cuda_c64, cuda_c128>;


        template<typename T>
        concept device_real_fp = requires { requires cpu_real_fp<T> || cuda_real_fp<T>;};

        template<typename T>
        concept device_complex_fp = requires { cuda_complex_fp<T> || cpu_complex_fp<T>;};

        template<typename T>
        concept device_fp = requires { cuda_fp<T> || cpu_fp<T>; };


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

        template<device_real_fp F>
        using complex_t = std::conditional_t<is_cuda<F>(), 
            complex_cuda_t<F>, complex_cpu_t<F>>;
                            


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

    }

}
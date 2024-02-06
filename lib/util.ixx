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

        template<device_fp F>
        using swap_complex_real_t = decltype(swap_complex_real_type_func<F>());

        template<device_real_fp F>
        using complex_t = decltype(smap_complex_real_type_func<F>());

        template<device_complex_fp F>
        using real_t = decltype(smap_complex_real_type_func<F>());

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
                throw std::runtime_error("Invalid device type");
            }
        }

        template<device_fp FP>
        using device_type_t = decltype(device_type_func<FP>());


        template<device_fp F1, device_fp F2>
        constexpr auto nonloss_type_func() {
            auto types = []<device_fp FP1, device_fp FP2>() {
                return std::is_same_v<FP1, F1> && std::is_same_v<FP2, F2> ||
                    std::is_same_v<FP1, F2> && std::is_same_v<FP2, F1>;
            };

            if        constexpr(std::is_same_v<device_type_t<F1>,device_type_t<F2>>) {
                throw std::runtime_error("Types must have same device in nonless_type_func");
            } else if constexpr(std::is_same_v<F1,F2>) {
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
                throw std::runtime_error("Invalid device type");
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

        template<auto n, typename F, typename T>
        constexpr T for_sequence(F f, const T& t) {
            T tcopy = t;
            for_sequence<n>([&tcopy, &f](auto i) {
                f(i, tcopy);
            });
            return tcopy;
        }

    }

}
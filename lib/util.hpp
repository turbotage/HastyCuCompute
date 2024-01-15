#pragma once
#include <cuComplex.h>
#include <torch/torch.h>


namespace hasty {

    template<size_t T1, size_t T2>
    concept less_than = T1 < T2;

    template<typename T>
    concept floating_point = std::is_same_v<float, T> || std::is_same_v<double, T>;

    template <typename T, typename = void>
    struct has_strong_value : std::false_type{};

    template <typename T>
    struct has_strong_value<T, decltype((void)T::strong_value, void())> : std::true_type {};

    struct strong_typedef_base {};

    template<typename T>
    concept is_strong_type = std::is_base_of_v<strong_typedef_base, T> && has_strong_value<T>::value;

    // export from here

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
    concept cpu_any_fp = requires {cpu_complex_fp<T> || cpu_real_fp<T>;};

    template<typename T>
    using complexify_cpu_type = std::conditional_t<std::is_same_v<T,cpu_f32>, cpu_c64, cpu_c128>;

    
    using cuda_f32 = strong_typedef<float, struct cuda_f32_>;
    using cuda_f64 = strong_typedef<double, struct cuda_f64_>;
    using cuda_c64 = strong_typedef<cuFloatComplex, struct cuda_c64_>;
    using cuda_c128 = strong_typedef<cuDoubleComplex, struct cuda_c128_>;

    template<typename T>
    concept cuda_real_fp = std::is_same_v<cuda_f32, T> || std::is_same_v<cuda_f64, T>;

    template<typename T>
    concept cuda_complex_fp = std::is_same_v<cuda_c64, T> || std::is_same_v<cuda_c128, T>;

    template<typename T>
    concept cuda_any_fp = requires {cuda_complex_fp<T> || cuda_real_fp<T>;};

    
    template<typename T>
    using complexify_cuda_type = std::conditional_t<std::is_same_v<T,cuda_f32>, cuda_c64, cuda_c128>;



    template<typename T>
    concept any_real_fp = requires { requires cpu_real_fp<T> || cuda_real_fp<T>;};

    template<typename T>
    concept any_complex_fp = requires { cuda_complex_fp<T> || cpu_complex_fp<T>;};

    template<typename T>
    concept any_fp = requires { cuda_any_fp<T> || cpu_any_fp<T>; };


    template<any_real_fp T>
    using complexify_type = std::conditional_t<(cuda_real_fp<T>), complexify_cuda_type<T>, complexify_cpu_type<T>>;

    template<any_fp FP>
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


    


}
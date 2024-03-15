module;

#include "../pch.hpp"

export module util:typing;

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

    }

    export {

		class multiple_inheritable_shared_from_this : public std::enable_shared_from_this<multiple_inheritable_shared_from_this> {
		public:
			virtual ~multiple_inheritable_shared_from_this(){}
		};

		template<class T>
		class inheritable_enable_shared_from_this : virtual public multiple_inheritable_shared_from_this {
		public:
			std::shared_ptr<T> shared_from_this() {
				return std::dynamic_pointer_cast<T>(multiple_inheritable_shared_from_this::shared_from_this());
			}

			std::shared_ptr<const T> shared_from_this() const {
				return std::dynamic_pointer_cast<const T>(multiple_inheritable_shared_from_this::shared_from_this());
			}

			template<class U>
			std::shared_ptr<U> downcast_shared_from_this() {
				return std::dynamic_pointer_cast<U>(multiple_inheritable_shared_from_this::shared_from_this());
			}

			template<class U>
			std::shared_ptr<const U> downcast_shared_from_this() const {
				return std::dynamic_pointer_cast<const U>(multiple_inheritable_shared_from_this::shared_from_this());
			}

		};

		template <typename T>
		struct reversion_wrapper { T& iterable; };

		template <typename T>
		auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

		template <typename T>
		auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

		template <typename T>
		reversion_wrapper<T> reverse(T&& iterable) { return { iterable }; }

	}

    export class NotImplementedError : public std::runtime_error {
	public:

		NotImplementedError()
			: std::runtime_error("Not Implemented Yet")
		{}

	};


}

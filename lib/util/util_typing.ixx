module;

#include "pch.hpp"

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

        template<typename T>
        using uptr = std::unique_ptr<T>;

        template<typename T>
        using sptr = std::shared_ptr<T>;

        template<typename T>
        using opt = std::optional<T>;

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

        template<typename T, typename U>
        struct strong_typedef : public strong_typedef_base {

            strong_typedef() = default;

            T strong_value;
        };

        template<typename T>
        struct empty_strong_typedef : public strong_typedef_base {
            empty_strong_typedef() = default;
        };


        using cuda_t = empty_strong_typedef<struct cuda_>;
        using cpu_t = empty_strong_typedef<struct cpu_>;

        template<typename T>
        concept is_device = std::is_same_v<T, cuda_t> || std::is_same_v<T, cpu_t>;

        using f32_t     = strong_typedef<f32, struct f32_>;
        using f64_t     = strong_typedef<f64, struct f64_>;
        using c64_t     = strong_typedef<c64, struct c64_>;
        using c128_t    = strong_typedef<c128, struct c128_>;
        using i16_t     = strong_typedef<i16, struct i16_>;
        using i32_t     = strong_typedef<i32, struct i32_>;
        using i64_t     = strong_typedef<i64, struct i64_>;
        using b8_t      = strong_typedef<bool, struct b8_>;
        
        template<typename T>
        concept is_tensor_type =    std::is_same_v<T, f32_t> || std::is_same_v<T, f64_t> || 
                                    std::is_same_v<T, c64_t> || std::is_same_v<T, c128_t> || 
                                    std::is_same_v<T, i32_t> || std::is_same_v<T, i64_t> ||
                                    std::is_same_v<T, i16_t> || std::is_same_v<T, b8_t>;

        template<typename T>
        concept is_fp_tensor_type = std::is_same_v<T, f32_t> || std::is_same_v<T, f64_t> || 
                                    std::is_same_v<T, c64_t> || std::is_same_v<T, c128_t>;

        template<typename T>
        concept is_fp32_tensor_type = std::is_same_v<T, f32_t> || std::is_same_v<T, c64_t>;

        template<typename T>
        concept is_fp64_tensor_type = std::is_same_v<T, f64_t> || std::is_same_v<T, c128_t>;

        template<typename T>
        concept is_fp_real_tensor_type = std::is_same_v<T, f32_t> || std::is_same_v<T, f64_t>;

        template<typename T>
        concept is_fp_complex_tensor_type = std::is_same_v<T, c64_t> || std::is_same_v<T, c128_t>;

        template<typename T>
        concept is_int_tensor_type =    std::is_same_v<T, i32_t> || std::is_same_v<T, i64_t> || 
                                        std::is_same_v<T, i16_t>;

        template<typename T>
        concept is_bool_tensor_type = std::is_same_v<T, b8_t>;

        template<is_tensor_type TT>
        constexpr at::ScalarType scalar_type_func() {
            if constexpr(std::is_same_v<TT, f32_t>) {
                return at::ScalarType::Float;
            } else if constexpr(std::is_same_v<TT, f64_t>) {
                return at::ScalarType::Double;
            } else if constexpr(std::is_same_v<TT, c64_t>) {
                return at::ScalarType::ComplexFloat;
            } else if constexpr(std::is_same_v<TT, c128_t>) {
                return at::ScalarType::ComplexDouble;
            } else if constexpr(std::is_same_v<TT, i32_t>) {
                return at::ScalarType::Int;
            } else if constexpr(std::is_same_v<TT, i64_t>) {
                return at::ScalarType::Long;
            } else if constexpr(std::is_same_v<TT, i16_t>) {
                return at::ScalarType::Short;
            } else if constexpr(std::is_same_v<TT, b8_t>) {
                return at::ScalarType::Bool;
            }
        }

        template<is_device D>
        constexpr at::DeviceType device_type_func() {
            if constexpr(std::is_same_v<D, cuda_t>) {
                return at::DeviceType::CUDA;
            } else if constexpr(std::is_same_v<D, cpu_t>) {
                return at::DeviceType::CPU;
            }
        }

        template<is_tensor_type TT1, is_tensor_type TT2>
        struct PairedTypeCheckStruct {

            template<is_tensor_type IN_T1, is_tensor_type IN_T2>
            constexpr static bool is_any_match() {
                return  std::is_same_v<TT1, IN_T1> && std::is_same_v<TT2, IN_T2> 
                        ||
                        std::is_same_v<TT1, IN_T2> && std::is_same_v<TT2, IN_T1>;
            }

        };


        template<is_tensor_type F1, is_tensor_type F2>
        constexpr auto nonloss_type_func() {

            using PTC = PairedTypeCheckStruct<F1,F2>;

            if constexpr(std::is_same_v<F1,F2>) {
                return F1();
            } else if constexpr(PTC::template is_any_match<f32_t, f64_t>())  {
                return f64_t();
            } else if constexpr(PTC::template is_any_match<f32_t, c128_t>()) {
                return c128_t();
            } else if constexpr(PTC::template is_any_match<f64_t, c64_t>())  {
                return c128_t();
            } else if constexpr(PTC::template is_any_match<f64_t, c128_t>()) {
                return c128_t();
            } else if constexpr(PTC::template is_any_match<c64_t, c128_t>()) {
                return c128_t();
            } else if constexpr(PTC::template is_any_match<f32_t, c64_t>())  {
                return c64_t();
            } else if constexpr(PTC::template is_any_match<i32_t, i64_t>())  {
                return i64_t();
            } else if constexpr(PTC::template is_any_match<i32_t, i16_t>())  {
                return i32_t();
            } else if constexpr(PTC::template is_any_match<i64_t, i16_t>())  {
                return i64_t();
            } else if constexpr(PTC::template is_any_match<b8_t,  i32_t>())  {
                return i32_t();
            } else if constexpr(PTC::template is_any_match<b8_t,  i64_t>())  {
                return i64_t();
            } else if constexpr(PTC::template is_any_match<b8_t,  i16_t>())  {
                return i16_t();
            } else {
                static_assert(false, "Invalid types");
            }
        }

        template<is_tensor_type F1, is_tensor_type F2>
        using nonloss_type_t = decltype(nonloss_type_func<F1, F2>());

        template<is_fp_tensor_type TT>
        constexpr auto complex_type_func() {
            if constexpr(std::is_same_v<TT, f32_t>) {
                return c64_t();
            } else if constexpr(std::is_same_v<TT, f64_t>) {
                return c128_t();
            } else {
                return TT();
            }
        }

        template<is_fp_tensor_type TT>
        using complex_t = decltype(complex_type_func<TT>());

        template<is_fp_tensor_type TT>
        constexpr auto real_Type_func() {
            if constexpr(std::is_same_v<TT, c64_t>) {
                return f32_t();
            } else if constexpr(std::is_same_v<TT, c128_t>) {
                return f64_t();
            } else {
                return TT();
            }
        }

        template<is_fp_tensor_type TT>
        using real_t = decltype(real_Type_func<TT>());

        template<is_tensor_type TT>
        constexpr auto up_precision_func() {
            if constexpr(std::is_same_v<TT, f32_t>) {
                return f64_t();
            } else if constexpr(std::is_same_v<TT, c64_t>) {
                return c128_t();
            } else if constexpr(std::is_same_v<TT, i32_t>) {
                return i64_t();
            } else if constexpr(std::is_same_v<TT, i16_t>) {
                return i32_t();
            } else {
                return TT();
            }
        }

        template<is_tensor_type TT>
        using up_precision_t = decltype(up_precision_func<TT>());

        template<is_tensor_type TT>
        constexpr auto down_precision_func() {
            if constexpr(std::is_same_v<TT, f64_t>) {
                return f32_t();
            } else if constexpr(std::is_same_v<TT, c128_t>) {
                return c64_t();
            } else if constexpr(std::is_same_v<TT, i64_t>) {
                return i32_t();
            } else if constexpr(std::is_same_v<TT, i32_t>) {
                return i16_t();
            } else {
                return TT();
            }
        }

        template<is_tensor_type TT>
        using down_precision_t = decltype(down_precision_func<TT>());



        template<is_strong_type T>
        using base_t = decltype(T::strong_value);

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

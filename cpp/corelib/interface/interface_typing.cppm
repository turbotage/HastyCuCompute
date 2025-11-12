module;

export module interface_typing;

import util;
import tensor;
import generic_value;

namespace hasty {
namespace interface {

// --- Type name helpers ---

template<typename Device>
std::string device_name() {
	if constexpr (std::is_same_v<Device, cuda_t>) return "cuda";
	else if constexpr (std::is_same_v<Device, cpu_t>) return "cpu";
	else throw std::runtime_error("Unknown device type");
}

template<typename Type>
std::string type_name() {
	if constexpr (std::is_same_v<Type, f32_t>) return "f32";
	else if constexpr (std::is_same_v<Type, f64_t>) return "f64";
	else if constexpr (std::is_same_v<Type, c64_t>) return "c64";
	else if constexpr (std::is_same_v<Type, c128_t>) return "c128";
	else if constexpr (std::is_same_v<Type, i16_t>) return "i16";
	else if constexpr (std::is_same_v<Type, i32_t>) return "i32";
	else if constexpr (std::is_same_v<Type, i64_t>) return "i64";
	else if constexpr (std::is_same_v<Type, b8_t>) return "b8";
	else throw std::runtime_error("Unknown tensor type");
}

// Type signature implementation
namespace detail {
	// Helper to extract tensor parameters
	template<typename T>
	struct tensor_traits;
	
	template<typename D, typename T, std::size_t R>
	struct tensor_traits<tensor<D, T, R>> {
		using device = D;
		using type = T;
		static constexpr std::size_t rank = R;
	};
	
	// Helper to check if T is tensor
	template<typename T>
	struct is_tensor_type : std::false_type {};
	
	template<typename D, typename T, std::size_t R>
	struct is_tensor_type<tensor<D, T, R>> : std::true_type {};
	
	// Helper to check if T is vector
	template<typename T>
	struct is_vector_type : std::false_type {};
	
	template<typename T>
	struct is_vector_type<std::vector<T>> : std::true_type {};
	
	// Helper to check if T is unordered_map
	template<typename T>
	struct is_map_type : std::false_type {};
	
	template<typename K, typename V>
	struct is_map_type<std::unordered_map<K, V>> : std::true_type {};
	
	// Helper to check if T is tuple
	template<typename T>
	struct is_tuple_type : std::false_type {};
	
	template<typename... Ts>
	struct is_tuple_type<std::tuple<Ts...>> : std::true_type {};
}

template<typename T>
std::string type_signature() {
	if constexpr (detail::is_tensor_type<T>::value) {
		// Use the typedefs from tensor class
		using Device = typename T::device_type_t;
		using Type = typename T::tensor_type_t;
		constexpr std::size_t Rank = T::size;
		return std::string("Tensor[") + device_name<Device>() + "," + 
			   type_name<Type>() + "," + std::to_string(Rank) + "]";
	}
	else if constexpr (detail::is_vector_type<T>::value) {
		using ElemType = typename T::value_type;
		return "List[" + type_signature<ElemType>() + "]";
	}
	else if constexpr (detail::is_map_type<T>::value) {
		using K = typename T::key_type;
		using V = typename T::mapped_type;
		return "Dict[" + type_name<K>() + "," + type_signature<V>() + "]";
	}
	else if constexpr (detail::is_tuple_type<T>::value) {
		return []<typename... Ts>(std::tuple<Ts...>*) {
			std::string s = "Tuple[";
			((s += type_signature<Ts>() + ","), ...);
			if constexpr (sizeof...(Ts) > 0) s.pop_back();
			s += "]";
			return s;
		}(static_cast<T*>(nullptr));
	}
	else {
		static_assert(always_false<T>, "Unsupported type for type_signature");
	}
}




}
}
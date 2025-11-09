module;

export module tensor_proto;

import std;
import torch_base;
import util;
import tensor;

namespace hasty {
namespace script {

    export template<is_device D, is_tensor_type TT, std::size_t RANK>
    class tensor_prototype {
    public:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<std::size_t, RANK> size = {};

        tensor_prototype(std::string name)
            : _tracestr(name)
        {};

        std::string str() const {
            return _tracestr;
        }

    private:
        std::string _tracestr;
    };

    export template<typename T>
    concept is_tensor_prototype = requires(T t) {
        []<is_device D, is_tensor_type TT, std::size_t RANK>(tensor_prototype<D,TT,RANK>&){}(t);
    };

    template<typename T>
    struct tensor_prototype_conversion;

    template<is_tensor_prototype T>
    struct tensor_prototype_conversion<T> {
        tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(T t);
    };

    template<is_tensor T>
    struct tensor_prototype_conversion<T> {
        tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(T t);
    };

    export template<is_tensor_prototype T>
    using tensor_prototype_conversion_t = std::invoke_result_t<tensor_prototype_conversion<T>, T>;


    // We need a tensor_prototype_container concept that can also
    // recognize nested containers of tensor_prototype types.
    // This requires a helper struct to implement the recursion.
    // We limit the recursion depth to prevent infinite recursion.

    template<typename T, int Depth = 0>
    struct is_tensor_prototype_container_impl {
        static constexpr bool value = is_tensor_prototype<T>;
    };

    // Specialization for std::vector
    template<typename T, int Depth>
    requires (Depth < 10)
    struct is_tensor_prototype_container_impl<std::vector<T>, Depth> {
        static constexpr bool value = is_tensor_prototype_container_impl<T, Depth+1>::value;
    };

    // Specialization for std::unordered_map with string or int keys
    template<typename K, typename V, int Depth>
    requires (Depth < 10) && (std::same_as<K, std::string> || std::same_as<K, std::size_t>)
    struct is_tensor_prototype_container_impl<std::unordered_map<K, V>, Depth> {
        static constexpr bool value = is_tensor_prototype_container_impl<V, Depth+1>::value;
    };

    // Specialization for std::tuple
    template<typename... Ts, int Depth>
    requires (Depth < 10)
    struct is_tensor_prototype_container_impl<std::tuple<Ts...>, Depth> {
        static constexpr bool value = (is_tensor_prototype_container_impl<Ts, Depth+1>::value && ...);
    };

    // Final concept definition
    export template<typename T>
    concept is_tensor_prototype_container = is_tensor_prototype_container_impl<T,0>::value;

    export template<typename T, int Depth>
    concept is_tensor_prototype_container_depthlimited = is_tensor_prototype_container_impl<T, Depth>::value; 

    // Now we define conversions for these containers

    template<typename T, int Depth = 0>
    struct tensor_prototype_container_conversion;

    // Case 1: T is tensor_prototype -> convert to tensor
    template<is_tensor_prototype T, int Depth>
    struct tensor_prototype_container_conversion<T, Depth> {
        using type = tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()>;
        
        type operator()(const T& t) const {
            return tensor_prototype_conversion<T>{}(t);
        }
    };

    // Case 2: T is tensor -> convert to tensor_prototype
    template<is_tensor T, int Depth>
    struct tensor_prototype_container_conversion<T, Depth> {
        using type = tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()>;
        
        type operator()(const T& t) const {
            return tensor_prototype_conversion<T>{}(t);
        }
    };

    // ====================
    // std::vector - automatically detects direction
    // ====================
    template<typename T, int Depth>
    requires (Depth < 10) && (is_tensor_prototype_container_depthlimited<T, Depth + 1> || 
                            is_tensor_container_depthlimited<T, Depth + 1>)
    struct tensor_prototype_container_conversion<std::vector<T>, Depth> {
        using inner_type = typename tensor_prototype_container_conversion<T, Depth + 1>::type;
        using type = std::vector<inner_type>;
        
        type operator()(const std::vector<T>& vec) const {
            type result;
            result.reserve(vec.size());
            tensor_prototype_container_conversion<T, Depth + 1> converter;
            for (const auto& elem : vec) {
                result.push_back(converter(elem));
            }
            return result;
        }
    };

    // ====================
    // std::unordered_map - automatically detects direction
    // ====================
    template<typename K, typename V, int Depth>
    requires (Depth < 10) && (std::same_as<K, std::string> || std::same_as<K, std::size_t>) &&
            (is_tensor_prototype_container_depthlimited<V, Depth + 1> || 
            is_tensor_container_depthlimited<V, Depth + 1>)
    struct tensor_prototype_container_conversion<std::unordered_map<K, V>, Depth> {
        using inner_type = typename tensor_prototype_container_conversion<V, Depth + 1>::type;
        using type = std::unordered_map<K, inner_type>;
        
        type operator()(const std::unordered_map<K, V>& map) const {
            type result;
            tensor_prototype_container_conversion<V, Depth + 1> converter;
            for (const auto& [key, value] : map) {
                result.emplace(key, converter(value));
            }
            return result;
        }
    };

    // ====================
    // std::tuple - automatically detects direction
    // ====================
    template<typename... Ts, int Depth>
    requires (Depth < 10) && ((is_tensor_prototype_container_depthlimited<Ts, Depth + 1> || 
                            is_tensor_container_depthlimited<Ts, Depth + 1>) && ...)
    struct tensor_prototype_container_conversion<std::tuple<Ts...>, Depth> {
        using type = std::tuple<typename tensor_prototype_container_conversion<Ts, Depth + 1>::type...>;
        
        type operator()(const std::tuple<Ts...>& tup) const {
            return convert_tuple(tup, std::index_sequence_for<Ts...>{});
        }
        
    private:
        template<std::size_t... Is>
        type convert_tuple(const std::tuple<Ts...>& tup, std::index_sequence<Is...>) const {
            return std::make_tuple(
                tensor_prototype_container_conversion<Ts, Depth + 1>{}(std::get<Is>(tup))...
            );
        }
    };

    export template<typename T>
    requires (is_tensor_prototype_container<T> || is_tensor_container<T>)
    using tensor_prototype_container_conversion_t = typename tensor_prototype_container_conversion<T>::type;

}
}
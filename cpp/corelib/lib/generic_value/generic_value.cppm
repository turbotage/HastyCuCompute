module;

//#include "pch.hpp"

export module generic_value;

import util;
import tensor;
import thread_stream;
import torch_base;
import std;

namespace hasty {

// Forward declaration
export class generic_value;

// Type tags for identifying tensor container types
// This matches the is_tensor_container concept structure
export enum class value_type : u8 {
    NONE = 0,
    TENSOR,       // tensor<D,T,R>
    VECTOR,       // std::vector<is_tensor_container>
    DICT,         // std::unordered_map<K, is_tensor_container>
    TUPLE         // std::tuple<is_tensor_container...>
};

// Serialization format for tensors
struct serialized_tensor_header {
    u8 device_type;      // 0 = CPU, 1 = CUDA
    u8 scalar_type;      // maps to at::ScalarType
    u8 ndim;             // number of dimensions
    i8 device_index;     // CUDA device index (-1 for CPU)
    i64 total_elements;   // total number of elements
    // followed by i64 shape[ndim]
    // followed by actual data
};

// Helper to get ScalarType from tensor
inline hat::ScalarType get_scalar_type(const hat::Tensor& t) {
    return t.scalar_type();
}

// Helper to get device info
inline std::pair<hat::DeviceType, i32> get_device_info(const hat::Tensor& t) {
    auto device = t.device();
    return {device.type(), device.has_index() ? device.index() : -1};
}

/**
 * @brief A generic value type for tensor containers (is_tensor_container concept)
 * 
 * This class provides runtime type erasure for statically-typed tensor containers.
 * It can hold any type satisfying is_tensor_container:
 * - tensor<D,T,R>
 * - std::vector<is_tensor_container>
 * - std::unordered_map<K, is_tensor_container>  
 * - std::tuple<is_tensor_container...>
 * 
 * Purpose: Enable dynamic dispatch to statically-typed runnable_script instances
 * 
 * Supports:
 * 1. Full serialization to binary format (for network transmission)
 * 2. Type string generation matching static type signatures
 * 3. Type + metadata string (includes shapes/sizes but no data)
 * 4. Dynamic-to-static type dispatch for script execution
 */
export class generic_value {
private:
    value_type m_type;
    
    // Storage using variant
    // Only holds types that satisfy is_tensor_container concept
    // Note: VECTOR and TUPLE both use std::vector<generic_value> storage
    // differentiated by m_type enum (same as before with LIST/TUPLE)
    std::variant<
        std::monostate,                                 // NONE
        hat::Tensor,                                    // TENSOR - tensor<D,T,R>
        std::vector<generic_value>,                     // VECTOR and TUPLE
        std::unordered_map<std::string, generic_value>  // DICT
    > m_data;

public:
    // ============================================================================
    // Constructors - Only for is_tensor_container types
    // ============================================================================
    
    // Default constructor - NONE
    generic_value() : m_type(value_type::NONE), m_data(std::monostate{}) {}
    
    // Tensor constructor - accepts hat::Tensor or tensor<D,T,R>
    explicit generic_value(hat::Tensor tensor) 
        : m_type(value_type::TENSOR), m_data(std::move(tensor)) {}
    
    // Tensor wrapper constructor - for tensor<D,T,R>
    template<is_device D, is_tensor_type T, std::size_t R>
    explicit generic_value(tensor<D,T,R> ten)
        : m_type(value_type::TENSOR) 
	{
		if (ten.ninstances() == 1) {
			m_data = std::move(ten.decay_to_tensor());
		} else {
			m_data = std::move(ten.get_tensor());
		}
	}
    
    // Vector constructor - std::vector<is_tensor_container>
    explicit generic_value(std::vector<generic_value> vec) 
        : m_type(value_type::VECTOR), m_data(std::move(vec)) {}
    
    // Dict constructor - std::unordered_map<string, is_tensor_container>
    explicit generic_value(std::unordered_map<std::string, generic_value> dict) 
        : m_type(value_type::DICT), m_data(std::move(dict)) {}
    
    // Tuple constructor - std::tuple<is_tensor_container...>
    // Note: Uses same storage as VECTOR but different enum tag
    static generic_value make_tuple(std::vector<generic_value> elements) {
        generic_value val;
        val.m_type = value_type::TUPLE;
        val.m_data = std::move(elements);
        return val;
    }
    
    // Generic constructor for any is_tensor_container type
    template<is_tensor_container T>
    static generic_value from_tensor_container(T&& container) {
        if constexpr (hasty::is_tensor<std::remove_cvref_t<T>>) {
            return generic_value(std::move(container));
        }
        else if constexpr (is_tensor_vector<std::remove_cvref_t<T>>) {
            using ElemType = typename std::remove_cvref_t<T>::value_type;
            std::vector<generic_value> vec;
            for (auto&& elem : container) {
                vec.push_back(from_tensor_container(std::forward<decltype(elem)>(elem)));
            }
            return generic_value(std::move(vec));
        }
        else if constexpr (is_tensor_dict<std::remove_cvref_t<T>>) {
            std::unordered_map<std::string, generic_value> dict;
            for (auto&& [key, value] : container) {
                dict[key] = from_tensor_container(std::forward<decltype(value)>(value));
            }
            return generic_value(std::move(dict));
        }
        else if constexpr (is_tensor_tuple<std::remove_cvref_t<T>>) {
            std::vector<generic_value> vec;
            std::apply([&vec](auto&&... args) {
                (vec.push_back(from_tensor_container(std::forward<decltype(args)>(args))), ...);
            }, std::forward<T>(container));
            return make_tuple(std::move(vec));
        }
        else {
            static_assert(always_false<T>, "Type does not satisfy is_tensor_container");
        }
    }

    // ============================================================================
    // Type queries
    // ============================================================================
    
    value_type type() const { return m_type; }
    bool is_none() const { return m_type == value_type::NONE; }
    bool is_tensor() const { return m_type == value_type::TENSOR; }
    bool is_vector() const { return m_type == value_type::VECTOR; }
    bool is_dict() const { return m_type == value_type::DICT; }
    bool is_tuple() const { return m_type == value_type::TUPLE; }

    // ============================================================================
    // Accessors (with type checking)
    // ============================================================================
    
    const hat::Tensor& to_tensor() const {
        if (!is_tensor()) throw std::runtime_error("Value is not a tensor");
        return std::get<hat::Tensor>(m_data);
    }
    
    hat::Tensor& to_tensor() {
        if (!is_tensor()) throw std::runtime_error("Value is not a tensor");
        return std::get<hat::Tensor>(m_data);
    }
    
    const std::vector<generic_value>& to_vector() const {
        if (!is_vector()) throw std::runtime_error("Value is not a vector");
        return std::get<std::vector<generic_value>>(m_data);
    }
    
    std::vector<generic_value>& to_vector() {
        if (!is_vector()) throw std::runtime_error("Value is not a vector");
        return std::get<std::vector<generic_value>>(m_data);
    }
    
    const std::unordered_map<std::string, generic_value>& to_dict() const {
        if (!is_dict()) throw std::runtime_error("Value is not a dict");
        return std::get<std::unordered_map<std::string, generic_value>>(m_data);
    }
    
    std::unordered_map<std::string, generic_value>& to_dict() {
        if (!is_dict()) throw std::runtime_error("Value is not a dict");
        return std::get<std::unordered_map<std::string, generic_value>>(m_data);
    }
    
    const std::vector<generic_value>& to_tuple() const {
        if (!is_tuple()) throw std::runtime_error("Value is not a tuple");
        return std::get<std::vector<generic_value>>(m_data);
    }
    
    std::vector<generic_value>& to_tuple() {
        if (!is_tuple()) throw std::runtime_error("Value is not a tuple");
        return std::get<std::vector<generic_value>>(m_data);
    }
    
    // ============================================================================
    // Convert back to static tensor_container type
    // This is the key function for dynamic-to-static dispatch!
    // ============================================================================
    
    template<is_tensor_container T>
    T to_static() const {
		using U = std::remove_cvref_t<T>;
        if constexpr (hasty::is_tensor<U>) {
            if (!is_tensor()) throw std::runtime_error("Cannot convert to tensor - wrong type");
            // Need to figure out device, type, and rank from T
            // For now, assume hat::Tensor conversion
            return T(to_tensor());
        }
        else if constexpr (is_specialization_of<U, std::vector>) {
            if (!is_vector()) throw std::runtime_error("Cannot convert to vector - wrong type");
            using ElemType = typename T::value_type;
            T result;
            for (const auto& elem : to_vector()) {
                result.push_back(elem.to_static<ElemType>());
            }
            return result;
        }
        else if constexpr (is_specialization_of<U, std::unordered_map>) {
            if (!is_dict()) throw std::runtime_error("Cannot convert to dict - wrong type");
            using ValueType = typename T::mapped_type;
            T result;
            for (const auto& [key, value] : to_dict()) {
                result[key] = value.to_static<ValueType>();
            }
            return result;
        }
        else if constexpr (is_specialization_of<U, std::tuple>) {
            if (!is_tuple()) throw std::runtime_error("Cannot convert to tuple - wrong type");
            return to_static_tuple<T>(std::make_index_sequence<std::tuple_size_v<T>>{});
        }
        else {
            static_assert(always_false<T>, "Type does not satisfy is_tensor_container");
        }
    }

    // ============================================================================
    // TYPE STRING (2): Just the type information matching static type signatures
    // ============================================================================
    
    /**
     * @brief Get a type string matching the static type_signature template
     * @return Type string like "Tensor[cpu,f32,3]", "List[Tensor[cuda,c64,2]]", "Dict[String,Tensor[...]]"
     */
    std::string type_string() const {
        switch (m_type) {
            case value_type::NONE:
                return "None";
            case value_type::TENSOR:
                return tensor_type_string(std::get<hat::Tensor>(m_data));
            case value_type::VECTOR: {
                const auto& vec = std::get<std::vector<generic_value>>(m_data);
                if (vec.empty()) return "List[]";
                // Assume homogeneous vector
                return "List[" + vec[0].type_string() + "]";
            }
            case value_type::TUPLE: {
                const auto& vec = std::get<std::vector<generic_value>>(m_data);
                if (vec.empty()) return "Tuple[]";
                std::string result = "Tuple[";
                for (std::size_t i = 0; i < vec.size(); ++i) {
                    if (i > 0) result += ",";
                    result += vec[i].type_string();
                }
                result += "]";
                return result;
            }
            case value_type::DICT: {
                const auto& dict = std::get<std::unordered_map<std::string, generic_value>>(m_data);
                if (dict.empty()) return "Dict[String,?]";
                // Assume homogeneous dict values
                return "Dict[String," + dict.begin()->second.type_string() + "]";
            }
        }
        return "Unknown";
    }

    // ============================================================================
    // METADATA STRING (3): Type + metadata (shapes, sizes) but no actual data
    // ============================================================================
    
    /**
     * @brief Get a detailed metadata string with type and structural info
     * @return String like "Tensor[cuda:0,Float32,shape=(3,224,224)]" or "List[n=5,elem=Tensor[...]]"
     */
    std::string metadata_string() const {
        switch (m_type) {
            case value_type::NONE:
                return "None";
            case value_type::TENSOR:
                return tensor_metadata_string(std::get<hat::Tensor>(m_data));
            case value_type::VECTOR: {
                const auto& vec = std::get<std::vector<generic_value>>(m_data);
                std::string result = "List[n=" + std::to_string(vec.size());
                if (!vec.empty()) {
                    result += ",elem=" + vec[0].metadata_string();
                }
                result += "]";
                return result;
            }
            case value_type::TUPLE: {
                const auto& vec = std::get<std::vector<generic_value>>(m_data);
                std::string result = "Tuple[";
                for (std::size_t i = 0; i < vec.size(); ++i) {
                    if (i > 0) result += ",";
                    result += vec[i].metadata_string();
                }
                result += "]";
                return result;
            }
            case value_type::DICT: {
                const auto& dict = std::get<std::unordered_map<std::string, generic_value>>(m_data);
                std::string result = "Dict[n=" + std::to_string(dict.size());
                if (!dict.empty()) {
                    result += ",value_type=" + dict.begin()->second.metadata_string();
                }
                result += "]";
                return result;
            }
        }
        return "Unknown";
    }

    // ============================================================================
    // SERIALIZATION (1): Full binary serialization including data
    // ============================================================================ 
    /**
     * @brief Serialize the entire value to a threadsafe_stream with chunking
     * @param stream Threadsafe stream to write chunks to
     */
    void serialize(threading::threadsafe_stream& stream) const {
        // Write type tag as a small chunk
        stream.write(serialize_one_int(static_cast<u8>(m_type)));
        
        switch (m_type) {
            case value_type::NONE:
                // Nothing to serialize
                break;
                
            case value_type::TENSOR:
                serialize_tensor_to_stream(stream, std::get<hat::Tensor>(m_data));
                break;
                
            case value_type::VECTOR:
            case value_type::TUPLE: {
                const auto& vec = std::get<std::vector<generic_value>>(m_data);
                // Write size of vector
                stream.write(serialize_one_int(vec.size()));
                for (const auto& elem : vec) {
                    elem.serialize(stream);
                }
                break;
            }
                
            case value_type::DICT: {
                const auto& dict = std::get<std::unordered_map<std::string, generic_value>>(m_data);
                // Write size of dictionary
                stream.write(serialize_one_int(dict.size()));
                for (const auto& [key, value] : dict) {
                    // Write key size
                    stream.write(serialize_one_int(key.size()));
                    // Write key bytes
                    stream.write(serialize_one_string(key));
                    // Write value
                    value.serialize(stream);
                }
                break;
            }
        }
    }
    
    /**
     * @brief Deserialize a value from a threadsafe_stream
     * @param stream Threadsafe stream to read chunks from
     * @return Deserialized generic_value
     */
    static generic_value deserialize(
        threading::threadsafe_stream& stream,
        std::chrono::milliseconds timeout_per_chunk = std::chrono::milliseconds(1000)
    ) {
        // Read type tag
        using ms = std::chrono::milliseconds;

        u8 type_tag = stream.read_exact_nbytes_blocking(1, timeout_per_chunk).first[0];

        generic_value result;
        result.m_type = static_cast<value_type>(type_tag);

        switch (result.m_type) {
            case value_type::NONE:
                result.m_data = std::monostate{};
                break;
                
            case value_type::TENSOR:
                result.m_data = deserialize_tensor_from_stream(stream);
                break;
                
            case value_type::VECTOR:
            case value_type::TUPLE: {
                i64 len = deserialize_one_int(stream.read_exact_nbytes_blocking(sizeof(i64), timeout_per_chunk).first);
                std::vector<generic_value> vec;
                vec.reserve(len);
                for (u64 i = 0; i < len; ++i) {
                    vec.push_back(deserialize(stream, timeout_per_chunk));
                }
                result.m_data = std::move(vec);
                break;
            }
                
            case value_type::DICT: {
                i64 len = deserialize_one_int(stream.read_exact_nbytes_blocking(sizeof(i64), timeout_per_chunk).first);
                std::unordered_map<std::string, generic_value> dict;
                dict.reserve(len);
                for (u64 i = 0; i < len; ++i) {
                    i64 key_len = deserialize_one_int(stream.read_exact_nbytes_blocking(sizeof(i64), timeout_per_chunk).first);
                    std::string key = deserialize_one_string(stream.read_exact_nbytes_blocking(key_len, timeout_per_chunk).first);
                    dict.emplace(std::move(key), deserialize(stream, timeout_per_chunk));
                }
                result.m_data = std::move(dict);
                break;
            }
        }
        
        return result;
    }

private:

    // Helper for tuple conversion
    template<typename TupleT, std::size_t... Is>
    TupleT to_static_tuple(std::index_sequence<Is...>) const {
        const auto& vec = to_tuple();
        if (vec.size() != sizeof...(Is)) {
            throw std::runtime_error("Tuple size mismatch");
        }
        return TupleT{vec[Is].to_static<std::tuple_element_t<Is, TupleT>>()...};
    }

    // Helper: Serialize tensor directly to stream in chunks
    static void serialize_tensor_to_stream(
        threading::threadsafe_stream& stream, 
        const hat::Tensor& tensor,
        std::size_t default_chunk_size = 4 * 1024 * 1024
    ) 
    {
        serialized_tensor_header header;
        
        auto [dev_type, dev_idx] = get_device_info(tensor);
        header.device_type = (dev_type == hat::kCUDA) ? 1 : 0;
        header.device_index = dev_idx;
        header.scalar_type = static_cast<u8>(get_scalar_type(tensor));
        header.ndim = static_cast<u8>(tensor.dim());
        header.total_elements = tensor.numel();

        stream.write(serialize_tensor_header(header));

        // Write shape as chunks
        std::vector<i64> shape(tensor.dim());
        for (int i = 0; i < tensor.dim(); ++i) {
            shape[i] = tensor.size(i);
        }
        stream.write(serialize_shape(shape));
        
        // Write data in chunks (ensure it's contiguous and on CPU)
        hat::Tensor cpu_tensor = tensor.to(hat::kCPU).contiguous();
        const u8* data_ptr = reinterpret_cast<const u8*>(cpu_tensor.data_ptr());
        std::size_t data_size = cpu_tensor.numel() * cpu_tensor.element_size();
        std::vector<u8> data_chunk(default_chunk_size);
        for (std::size_t i = 0; i < data_size; i += default_chunk_size) {
            std::size_t size = std::min(default_chunk_size, data_size - i);
            data_chunk.resize(size);
            std::memcpy(data_chunk.data(), data_ptr + i, size);
            stream.write(data_chunk);
        }
    }

        // Helper: Deserialize tensor incrementally from stream
    static hat::Tensor deserialize_tensor_from_stream(
        threading::threadsafe_stream& stream,
        std::chrono::milliseconds timeout_per_chunk = std::chrono::milliseconds(1000)
    ) {
        // Read header
        auto header_chunk = stream.read_exact_nbytes_blocking(sizeof(serialized_tensor_header), timeout_per_chunk).first;
        serialized_tensor_header header = *reinterpret_cast<const serialized_tensor_header*>(header_chunk.data());
        
        // Read shape
        std::vector<i64> shape = deserialize_shape(
            stream.read_exact_nbytes_blocking(sizeof(i64) * header.ndim, timeout_per_chunk).first,
            header.ndim
        );

        // Validate total elements
        if (util::container_product(shape) != header.total_elements) {
            throw std::runtime_error("Tensor shape does not match total elements in header");
        }
        
        // Create tensor
        hat::ScalarType dtype = static_cast<hat::ScalarType>(header.scalar_type);
        hat::Device device = (header.device_type == 1) 
            ? hat::Device(hat::kCUDA, header.device_index)
            : hat::Device(hat::kCPU);
        
        hat::Tensor tensor = hat::empty(shape, hat::TensorOptions().dtype(dtype).device(hat::kCPU));
        
        // Read data in chunks and copy directly to tensor memory
        std::size_t data_size = tensor.numel() * tensor.element_size();

        u8* tensor_data = reinterpret_cast<u8*>(tensor.data_ptr());
        std::size_t offset = 0;
        while (offset < data_size) {
            std::size_t max_chunk_size = data_size - offset;
            auto data_chunk = stream.read_max_nbytes_blocking(max_chunk_size, timeout_per_chunk).first;
            if (data_chunk.size() > max_chunk_size) {
                throw std::runtime_error("Received more data than expected for tensor");
            }
            std::memcpy(tensor_data + offset, data_chunk.data(), data_chunk.size());
            offset += data_chunk.size();
        }
        
        // Move to target device if needed
        if (header.device_type == 1) {
            tensor = tensor.to(device);
        }
        
        return tensor;
    }
    
private:
    // Helper: Get tensor type string matching the static type_signature format
    static std::string tensor_type_string(const hat::Tensor& tensor) {
        auto [dev_type, dev_idx] = get_device_info(tensor);
        std::string device_str = (dev_type == hat::kCUDA) ? "cuda" : "cpu";
        std::string dtype_str = scalar_type_to_typename(get_scalar_type(tensor));
        std::string rank_str = std::to_string(tensor.dim());
        
        return "Tensor[" + device_str + "," + dtype_str + "," + rank_str + "]";
    }
    
    // Helper: Get tensor metadata string
    static std::string tensor_metadata_string(const hat::Tensor& tensor) {
        auto [dev_type, dev_idx] = get_device_info(tensor);
        std::string device_str = (dev_type == hat::kCUDA) 
            ? "cuda:" + std::to_string(dev_idx)
            : "cpu";
        
        std::string dtype_str = scalar_type_to_string(get_scalar_type(tensor));
        
        std::string shape_str = "shape=(";
        for (int i = 0; i < tensor.dim(); ++i) {
            if (i > 0) shape_str += ",";
            shape_str += std::to_string(tensor.size(i));
        }
        shape_str += ")";
        
        return "Tensor[" + device_str + "," + dtype_str + "," + shape_str + "]";
    }
    
    // Helper: Convert scalar type to type name (matching interface_typing)
    static std::string scalar_type_to_typename(hat::ScalarType dtype) {
        switch (dtype) {
            case hat::kFloat: return "f32";
            case hat::kDouble: return "f64";
            case hat::kComplexFloat: return "c64";
            case hat::kComplexDouble: return "c128";
            case hat::kShort: return "i16";
            case hat::kInt: return "i32";
            case hat::kLong: return "i64";
            case hat::kBool: return "b8";
            case hat::kByte: return "u8";
            default: return "unknown";
        }
    }
    
    // Helper: Convert scalar type to string (human readable)
    static std::string scalar_type_to_string(hat::ScalarType dtype) {
        switch (dtype) {
            case hat::kFloat: return "Float32";
            case hat::kDouble: return "Float64";
            case hat::kComplexFloat: return "Complex64";
            case hat::kComplexDouble: return "Complex128";
            case hat::kInt: return "Int32";
            case hat::kLong: return "Int64";
            case hat::kShort: return "Int16";
            case hat::kByte: return "UInt8";
            case hat::kBool: return "Bool";
            default: return "Unknown";
        }
    }

    thread_local static std::vector<u8> s_int_buffer;
    thread_local static std::vector<u8> s_string_buffer;
    thread_local static std::vector<u8> s_tensor_header_buffer;
    thread_local static std::vector<u8> s_shape_buffer;


    static const std::vector<u8>& serialize_one_int(i64 value) {
        s_int_buffer.resize(sizeof(i64));
        std::memcpy(s_int_buffer.data(), &value, sizeof(i64));
        return s_int_buffer;
    }

    static i64 deserialize_one_int(const std::vector<u8>& buffer) {
        if (buffer.size() < sizeof(i64)) {
            throw std::runtime_error("Buffer too small to deserialize int");
        }
        i64 value;
        std::memcpy(&value, buffer.data(), sizeof(i64));
        return value;
    }

    static const std::vector<u8>& serialize_one_string(const std::string& str) {
        s_string_buffer.resize(str.size());
        std::memcpy(s_string_buffer.data(), str.data(), str.size());
        return s_string_buffer;
    }

    static std::string deserialize_one_string(const std::vector<u8>& buffer) {
        return std::string(buffer.begin(), buffer.end());
    }
    
    static const std::vector<u8>& serialize_tensor_header(const serialized_tensor_header& header) {
        s_tensor_header_buffer.resize(sizeof(serialized_tensor_header));
        std::memcpy(s_tensor_header_buffer.data(), &header, sizeof(serialized_tensor_header));
        return s_tensor_header_buffer;
    }

    static serialized_tensor_header deserialize_tensor_header(const std::vector<u8>& buffer) {
        if (buffer.size() < sizeof(serialized_tensor_header)) {
            throw std::runtime_error("Buffer too small to deserialize tensor header");
        }
        serialized_tensor_header header;
        std::memcpy(&header, buffer.data(), sizeof(serialized_tensor_header));
        return header;
    }

    static const std::vector<u8>& serialize_shape(const std::vector<i64>& shape) {
        s_shape_buffer.resize(sizeof(i64) * shape.size());
        std::memcpy(s_shape_buffer.data(), shape.data(), sizeof(i64) * shape.size());
        return s_shape_buffer;
    }

    static std::vector<i64> deserialize_shape(const std::vector<u8>& buffer, std::size_t ndim) {
        if (buffer.size() < sizeof(i64) * ndim) {
            throw std::runtime_error("Buffer too small to deserialize shape");
        }
        std::vector<i64> shape(ndim);
        std::memcpy(shape.data(), buffer.data(), sizeof(i64) * ndim);
        return shape;
    }

};

// Convenience functions for creating values
export inline generic_value make_generic_vector(std::vector<generic_value> elements) {
    return generic_value(std::move(elements));
}

export inline generic_value make_generic_tuple(std::vector<generic_value> elements) {
    return generic_value::make_tuple(std::move(elements));
}

export inline generic_value make_generic_dict(std::unordered_map<std::string, generic_value> dict) {
    return generic_value(std::move(dict));
}

// Helper to create generic_value from any is_tensor_container type
export template<is_tensor_container T>
generic_value to_generic(T&& container) {
    return generic_value::from_tensor_container(std::forward<T>(container));
}

// ============================================================================
// IValue Conversion - External functions only, no member functions
// ============================================================================

/**
 * @brief Convert torch::jit::IValue to generic_value (move semantics)
 * @param ivalue IValue to convert (will be moved from)
 * @return generic_value containing the converted data
 */
export generic_value ivalue_to_generic(htorch::jit::IValue&& ivalue) {
    if (ivalue.isNone()) {
        return generic_value();
    } else if (ivalue.isTensor()) {
        return generic_value(ivalue.toTensor());
    } else if (ivalue.isList()) {
        auto list = ivalue.toList();
        std::vector<generic_value> elements;
        elements.reserve(list.size());
        for (std::size_t i = 0; i < list.size(); ++i) {
            // list.get(i) returns a proxy type, need to extract as IValue
            auto item = list.get(i);
            elements.push_back(ivalue_to_generic(htorch::jit::IValue(item)));
        }
        return generic_value(std::move(elements));
    } else if (ivalue.isGenericDict()) {
        auto dict = ivalue.toGenericDict();
        std::unordered_map<std::string, generic_value> map;
        // Dict iterator returns DictEntryRef, need to use key() and value() explicitly
        for (auto it = dict.begin(); it != dict.end(); ++it) {
            auto key = it->key();
            auto value = it->value();
            if (key.isString()) {
                map.emplace(key.toStringRef(), ivalue_to_generic(htorch::jit::IValue(value)));
            }
        }
        return generic_value(std::move(map));
    } else if (ivalue.isTuple()) {
        auto tuple = ivalue.toTuple();
        // intrusive_ptr needs arrow operator
        const auto& elements = tuple->elements();
        std::vector<generic_value> tuple_elements;
        tuple_elements.reserve(elements.size());
        // elements is const vector, need to copy each element
        for (const auto& elem : elements) {
            tuple_elements.push_back(ivalue_to_generic(htorch::jit::IValue(elem)));
        }
        return generic_value::make_tuple(std::move(tuple_elements));
    }
    
    throw std::runtime_error("Unsupported IValue type for conversion to generic_value");
}

/**
 * @brief Convert generic_value to torch::jit::IValue (move semantics)
 * @param gvalue generic_value to convert (will be moved from)
 * @return IValue containing the converted data
 */
export htorch::jit::IValue generic_to_ivalue(generic_value&& gvalue) {
    if (gvalue.is_none()) {
        return htorch::jit::IValue();
    } else if (gvalue.is_tensor()) {
        return htorch::jit::IValue(gvalue.to_tensor());
    } else if (gvalue.is_vector()) {
        auto& vec = gvalue.to_vector();
        // First convert all elements to IValue
        std::vector<htorch::jit::IValue> ivalue_vec;
        ivalue_vec.reserve(vec.size());
        for (auto& elem : vec) {
            ivalue_vec.push_back(generic_to_ivalue(std::move(elem)));
        }
        
        // Get the type from the first element
        auto typeptr = ivalue_vec[0].type();
        hc10::impl::GenericList ivalue_list(typeptr);
        ivalue_list.reserve(ivalue_vec.size());
        for (auto& ivalue_elem : ivalue_vec) {
            if (ivalue_elem.type() != typeptr) {
                throw std::runtime_error("Inconsistent types in generic_to_ivalue vector");
            }
            ivalue_list.push_back(std::move(ivalue_elem));
        }
        return htorch::jit::IValue(std::move(ivalue_list));
    } else if (gvalue.is_dict()) {
        auto& dict = gvalue.to_dict();
        // First convert all values to IValue
        //std::unordered_map<std::string, htorch::jit::IValue> ivalue_map;
        std::vector<std::pair<htorch::jit::IValue, htorch::jit::IValue>> ivalue_vec;
        hc10::TypePtr value_typeptr = nullptr;
        ivalue_vec.reserve(dict.size());
        for (auto& [key, value] : dict) {
            auto ivalue = generic_to_ivalue(std::move(value));
            if (value_typeptr == nullptr) {
                value_typeptr = ivalue.type();
            } else if (ivalue.type() != value_typeptr) {
                throw std::runtime_error("Inconsistent types in generic_to_ivalue dict");
            }
            ivalue_vec.emplace_back(
                htorch::jit::IValue(key), 
                std::move(ivalue)
            );
        }
        // Get the value type from the first element
        
        //hc10::impl::GenericDict ivalue_dict(hc10::StringType::get(), value_typeptr);
        //ivalue_dict.reserve(ivalue_map.size());

        //hc10::Dict<htorch::jit::IValue, htorch::jit::IValue> ivalue_dict(std::move(ivalue_vec));

        //hc10::impl::GenericDict ivalue_dict(hc10::StringType::get(), value_typeptr);
        //hc10::Dict<htorch::jit::IValue, htorch::jit::IValue> ivalue_dict(std::move(ivalue_dict));
        //hc10::Dict<htorch::jit::IValue, htorch::jit::IValue> ivalue_dict(std::move(ivalue_vec));
        hc10::Dict<htorch::jit::IValue, htorch::jit::IValue> ivalue_dict(
            hc10::StringType::get(), 
            value_typeptr
        );
        /*
        for (auto& [key_ivalue, value_ivalue] : ivalue_vec) {
            ivalue_dict.insert(std::move(key_ivalue), std::move(value_ivalue));
        }
        */
        for (auto& pair : ivalue_vec) {
            ivalue_dict.insert(std::move(pair.first), std::move(pair.second));
        }
    
        return htorch::jit::IValue(std::move(ivalue_dict));
    } else if (gvalue.is_tuple()) {
        auto& vec = gvalue.to_tuple();  // Tuple uses vector storage
        std::vector<htorch::jit::IValue> ivalue_elements;
        ivalue_elements.reserve(vec.size());
        for (i32 i = 0; i < vec.size(); ++i) {
            ivalue_elements.push_back(generic_to_ivalue(std::move(vec[i])));
        }
        // Tuple is in c10::ivalue namespace
        return hc10::ivalue::Tuple::create(std::move(ivalue_elements));
    }
    
    throw std::runtime_error("Unknown generic_value type for conversion to IValue");
}

}
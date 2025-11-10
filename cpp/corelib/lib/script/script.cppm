module;

#include <valarray>
export module script;

import torch_base;
import util;
import tensor;
export import tensor_proto;

namespace hasty {
namespace script {

export using Module = htorch::jit::Module;

// This is wrong. The runnable script should take is_tensor_container
// the prototype should be for the builders...

export template<is_tensor_container ReturnTt, is_tensor_container... InputTt>
struct runnable_script {
private:
    std::string _script_name;
    uptr<Module> _script_module;
public:

    using ReturnType = ReturnTt;
    using InputTraits = TupleTraits<InputTt...>;

    runnable_script(std::string_view script_name, uptr<Module>&& script_module)
        :   _script_name(script_name),
            _script_module(std::move(script_module))
    {}

    template<typename... Ts>
    requires (sizeof...(Ts) == sizeof...(InputTt)) &&
             (std::same_as<std::remove_cvref_t<Ts>, InputTt> && ...)
    auto run(Ts&&... inputs) const -> ReturnTt
    {
        auto ttscopy = std::tuple(InputTt(std::forward<InputTt>(inputs))...);

        std::vector<htorch::jit::IValue> ivalue_inputs;
        ivalue_inputs.reserve(sizeof...(InputTt));

        // Iterate over tensor_prototype_containers and convert them
        for_sequence<sizeof...(InputTt)>([&](auto i) {
            ivalue_inputs.push_back(
                to_ivalue(std::move(std::get<i>(ttscopy)))
            );
        });

        // Execute
        htorch::jit::IValue ret_ivalue;
        if (_script_module != nullptr) {
            ret_ivalue = _script_module->forward(ivalue_inputs);
        } else {
            throw std::runtime_error("runnable_script has no script unit");
        }

        return from_ivalue<ReturnTt>(std::move(ret_ivalue));
    }

    Module& get_module() const {
        if (_script_module == nullptr) {
            throw std::runtime_error("Module in runnable_script was null");
        }
        return *_script_module;
    }

    uptr<Module> decay_to_module() {
        return std::move(_script_module);
    }

private:

    //====================
    // tensor_container -> IValue conversion utilities
    //====================
    template<typename T>
    static htorch::jit::IValue to_ivalue(T&& container) {
        using U = std::remove_cvref_t<T>;

        if constexpr (is_tensor<U>) {
            return tensor_to_ivalue<U>(std::move(container));
        } else if constexpr (is_tensor_vector<U>) {
            return vector_to_ivalue<U>(std::move(container));
        } else if constexpr (is_tensor_dict<U>) {
            return dict_to_ivalue<U>(std::move(container));
        } else if constexpr (is_tensor_tuple<U>) {
            return tuple_to_ivalue<U>(std::move(container));
        } else {
            static_assert(always_false<T>, "Type is not a tensor container");
        }
    }

    template<is_tensor T>
    static htorch::jit::IValue tensor_to_ivalue(T&& tensor) 
    {
        if (tensor.ninstances() == 1) {
            return htorch::jit::IValue(tensor.decay_to_tensor());
        } else {
            return htorch::jit::IValue(tensor.get_tensor());
        }
    }

    template<is_tensor_container Vec>
    static htorch::jit::IValue vector_to_ivalue(Vec&& vec) {
        //hc10::List<htorch::jit::IValue> ivalue_list;
        std::vector<htorch::jit::IValue> ivalue_vec;
        ivalue_vec.reserve(vec.size());
        using U = std::remove_cvref_t<Vec>::value_type;
        for (auto&& elem : vec) {
            ivalue_vec.push_back(to_ivalue<U>(std::move(elem)));
        }
        hc10::List<htorch::jit::IValue> ivalue_list{hc10::ArrayRef<htorch::jit::IValue>(ivalue_vec)};
        //auto list = hc10::impl::toList(std::move(ivalue_list));
        return htorch::jit::IValue(std::move(ivalue_list));
    }

    template<is_tensor_container Map>
    static htorch::jit::IValue dict_to_ivalue(Map&& map) {
        using U = std::remove_cvref_t<Map>;
        using key_type = typename U::key_type;
        using mapped_type = typename U::mapped_type;

        std::unordered_map<key_type, htorch::jit::IValue> ivalue_map;
        ivalue_map.reserve(map.size());
        for (auto&& [key, val] : map) {
            ivalue_map.insert(key, to_ivalue<mapped_type>(std::move(val)));
        }
        return htorch::jit::IValue(std::move(ivalue_map));
    }

    template<is_tensor_container Tup>
    static htorch::jit::IValue tuple_to_ivalue(Tup&& tup) {
        return htorch::jit::IValue(
            std::apply([](auto&&... elements) {
                return std::make_tuple(
                    to_ivalue<std::remove_cvref_t<decltype(elements)>>(std::move(elements))...
                );
            }, std::forward<Tup>(tup))
        );
    }

    //====================
    // IValue -> tensor_container conversion utilities
    //====================

    template<is_tensor_container T>
    static T from_ivalue(htorch::jit::IValue&& ivalue) {
        using U = std::remove_cvref_t<T>;

        if constexpr (is_tensor<U>) {
            return ivalue_to_tensor<U>(std::move(ivalue));
        } else if constexpr (is_specialization_of<U, std::vector>) {
            return ivalue_to_vector<U>(std::move(ivalue));
        } else if constexpr (is_specialization_of<U, std::unordered_map>) {
            return ivalue_to_dict<U>(std::move(ivalue));
        } else if constexpr (is_specialization_of<U, std::tuple>) {
            return ivalue_to_tuple<U>(std::move(ivalue));
        } else {
            static_assert(always_false<T>, "Type is not a tensor container");
        }
    }

    template<is_tensor T>
    static T ivalue_to_tensor(htorch::jit::IValue&& ivalue)
    {
        if (!ivalue.isTensor()) {
            throw std::runtime_error("IValue is not a tensor");
        }
        hat::Tensor tensor(ivalue.toTensor());
        if (tensor.dim() != T::size()) {
            throw std::runtime_error("IValue tensor rank does not match target tensor rank");
        }
        if (tensor.device().type() != device_type_func<typename T::device_type_t>()) {
            throw std::runtime_error("IValue tensor device does not match target tensor device");
        }
        if (tensor.scalar_type() != scalar_type_func<typename T::tensor_type_t>()) {
            throw std::runtime_error("IValue tensor type does not match target tensor type");
        }
        
        std::array<i64, T::size()> new_shape;
        for_sequence<T::size()>([&](auto i) {
            new_shape[i] = tensor.size(i);
        });

        return T(new_shape, tensor);
    }

    template<is_tensor_container Vec>
    static Vec ivalue_to_vector(htorch::jit::IValue&& ivalue)
    {
        using U = std::remove_cvref_t<Vec>;
        using value_type = typename U::value_type;
        if (!ivalue.isList()) {
            throw std::runtime_error("IValue is not a list");
        }

        auto&& ivalue_list = std::move(ivalue).toList();
        Vec result;
        result.reserve(ivalue_list.size());
        for (int i = 0; i < ivalue_list.size(); i++) {
            result.push_back(from_ivalue<value_type>(std::move(ivalue_list.extract(i))));
        }
        return result;
    }

    template<is_tensor_container Map>
    static Map ivalue_to_dict(htorch::jit::IValue&& ivalue)
    {
        using U = std::remove_cvref_t<Map>;
        using key_type = typename U::key_type;
        using mapped_type = typename U::mapped_type;

        if (!ivalue.isGenericDict()) {
            throw std::runtime_error("IValue is not a dict");
        }

        auto&& ivalue_dict = std::move(ivalue).toGenericDict();
        Map result;
        for (auto&& item : ivalue_dict) {
            key_type key;
            if constexpr (std::is_same_v<key_type, std::string>) {
                if (!item.key().isString()) {
                    throw std::runtime_error("IValue dict key is not a string");
                }
                key = item.key().toStringRef();
            } else if constexpr (std::is_same_v<key_type, i64>) {
                if (!item.key().isInt()) {
                    throw std::runtime_error("IValue dict key is not an int");
                }
                key = item.key().toInt();
            } else {
                throw std::runtime_error("Unsupported key type in IValue dict");
            }

            //result[key] = from_ivalue<mapped_type>(item.value());
            result.emplace(
                std::move(key),
                from_ivalue<mapped_type>(std::move(item.value()))
            );
        }
        return result;
    }

    template<is_tensor_container Tup>
    static Tup ivalue_to_tuple(htorch::jit::IValue&& ivalue)
    {
        if (!ivalue.isTuple()) {
            throw std::runtime_error("IValue is not a tuple");
        }
        hc10::intrusive_ptr<hc10::ivalue::Tuple> ivalue_tuple = std::move(ivalue).toTuple();
        Tup result;
        hc10::ivalue::TupleElements elements = std::move(std::move(ivalue_tuple)->elements());
        
        constexpr std::size_t N = std::tuple_size_v<Tup>;
        using TT = TupleTraits<Tup>;
        for_sequence<TT::Size>([&](auto i) {
            using TYPE = typename TT::template Nth<i>;
            // Cast away const since we own the tuple and are consuming the IValue
            std::get<i>(result) = from_ivalue<TYPE>(std::move(elements[i]));
        });
        return result;
    }

};

export template<typename T>
concept is_runnable_script = requires {
    typename T::ReturnType;
    typename T::InputTraits;
} && is_specialization_of<T, runnable_script>;


template<is_tensor_container T>
std::string get_torchscript_type_string() {
    using U = std::remove_cvref_t<T>;

    if constexpr(is_tensor<U>) {
        return "Tensor";
    } else if constexpr(is_specialization_of<U, std::vector>) {
        using value_type = typename U::value_type;
        return "List[" + get_torchscript_type_string<value_type>() + "]";
    } else if constexpr(is_specialization_of<U, std::unordered_map>) {
        using key_type = typename U::key_type;
        using mapped_type = typename U::mapped_type;
        
        std::string key_str;
        if constexpr(std::is_same_v<key_type, std::string>) {
            key_str = "str";
        } else if constexpr(std::is_same_v<key_type, std::size_t>) {
            key_str = "int";
        } else {
            static_assert(always_false<T>, "Invalid key type for tensor container map");
        }

        return "Dict[" + key_str + ", " + get_torchscript_type_string<mapped_type>() + "]";
    } else if constexpr(is_specialization_of<U, std::tuple>) {
        std::string result = "Tuple[";
        constexpr std::size_t N = std::tuple_size_v<U>;
        for_sequence<N>([&](auto i) {
            result += get_torchscript_type_string<std::tuple_element_t<i, U>>();
            if constexpr (i < N-1) {
                result += ", ";
            }
        });
        result += "]";
        return result;
    } else {
        static_assert(always_false<T>, "Type is not a tensor container");
    }
}

export template<is_tensor_container ReturnTt, is_tensor_container... InputTt>
struct compiled_script_builder {
private:
    std::string _funcname;
    std::string _scriptstr;
    std::string _forwardname;
    std::string _compiled;
    std::tuple<NT<InputTt>...> _script_tensors;
    
    bool _freeze = true;
    bool _optimize_for_inference = true;

    uptr<Module> _mod;

public:

    using ReturnType = ReturnTt;
    using InputTraits = TupleTraits<InputTt...>;

    template<typename... Ts>
    requires    (is_specialization_of<std::remove_cvref_t<Ts>, NT> && ...) &&
                (is_tensor_container<typename std::remove_cvref_t<Ts>::value_type> && ...)
    compiled_script_builder(
        std::string_view funcname,
        std::string_view code,
        Ts&&... tts
    )
        :
        _funcname(funcname),
        _scriptstr(code),
        _script_tensors(
            std::forward<Ts>(tts)...
        ),
        _freeze(true),
        _optimize_for_inference(true)
    {
        reset();
    }

    void reset() {
        _mod = nullptr;

        std::string inputvars = "self";

        // Input parameters
        for_sequence<sizeof...(InputTt)>([this, &inputvars](auto i) {
            auto& itt = std::get<i>(_script_tensors);
            using NI_TYPE = typename InputTraits::template Nth<i>;
            inputvars += ", " + itt.get_name() + ": " + get_torchscript_type_string<NI_TYPE>();
        });

        // Return parameters
        std::string returnvars = " -> " + get_torchscript_type_string<ReturnTt>();

        _forwardname = std::format("def forward({}){}:", inputvars, returnvars);
        _compiled = util::replace_line(_scriptstr, "FORWARD_ENTRYPOINT", _forwardname);
    }

    const std::string& uncompiled_str() const {
        return _scriptstr;
    }

    const std::string& compiled_str() const {
        return _compiled;
    }

    void freeze(bool v) {
        _freeze = v;
    }

    void optimize_for_inference(bool v) {
        _optimize_for_inference = v;
    }

    void compile() {
        if (_mod != nullptr) {
            throw std::runtime_error("Script already compiled");
        }

        _mod = std::make_unique<Module>(_funcname);
        _mod->define(_compiled);

        if (_freeze) {
            *_mod = htorch::jit::freeze(*_mod);
        }
        if (_optimize_for_inference) {
            *_mod = htorch::jit::optimize_for_inference(*_mod);
        }
    }

    const Module& get_module() const {
        if (_mod == nullptr) {
            throw std::runtime_error("Module not compiled");
        }
        return *_mod;
    }

    uptr<Module> decay_to_module() {
        return std::move(_mod);
    }

    runnable_script<ReturnTt, InputTt...> decay_to_runnable_script() {
        if (_mod == nullptr) {
            throw std::runtime_error("Module not compiled");
        }
        return runnable_script<ReturnTt, InputTt...>(
            _funcname,
            decay_to_module()
        );
    }

};

export template<typename T>
concept is_compiled_script_builder = requires {
    typename T::ReturnType;
    typename T::InputTraits;
} && is_specialization_of<T, compiled_script_builder>;


export template<is_tensor_container ReturnTt, typename... Ts>
requires    (is_specialization_of<std::remove_cvref_t<Ts>, NT> && ...) && 
            (is_tensor_container<typename std::remove_cvref_t<Ts>::value_type> && ...)
auto make_compiled_script_builder(
    std::string_view funcname,
    std::string_view code,
    Ts&&... tts
) -> compiled_script_builder<ReturnTt, typename std::remove_cvref_t<Ts>::value_type...> 
{
    return compiled_script_builder<ReturnTt, typename std::remove_cvref_t<Ts>::value_type...>(
        funcname,
        code,
        std::forward<Ts>(tts)...
    );
}

// Type converters

template<typename Builder>
struct builder_to_runnable;

template<is_tensor_container ReturnTt, is_tensor_container... InputTt>
struct builder_to_runnable<compiled_script_builder<ReturnTt, InputTt...>> {
    using type = runnable_script<ReturnTt, InputTt...>;
};

export template<is_compiled_script_builder Builder>
using builder_to_runnable_t = typename builder_to_runnable<Builder>::type;

template<typename Runnable>
struct runnable_to_compiled_builder;

template<is_tensor_container ReturnTt, is_tensor_container... InputTt>
struct runnable_to_compiled_builder<runnable_script<ReturnTt, InputTt...>> {
    using type = compiled_script_builder<ReturnTt, InputTt...>;
};

export template<is_runnable_script Runnable>
using runnable_to_compiled_builder_t = typename runnable_to_compiled_builder<Runnable>::type;



}
}
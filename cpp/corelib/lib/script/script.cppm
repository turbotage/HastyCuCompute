module;

export module script;

import torch_base;
import util;
import tensor;
export import tensor_proto;

namespace hasty {
namespace script {

export using Module = htorch::jit::Module;

export template<is_tensor_prototype_container ReturnTt, is_tensor_prototype_container... InputTt>
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
            ((std::same_as<std::remove_cvref_t<Ts>, 
                        tensor_prototype_container_conversion_t<InputTt>> ||
            std::convertible_to<Ts, tensor_prototype_container_conversion_t<InputTt>>) && ...)
    auto run(Ts&&... inputs) const -> tensor_prototype_container_conversion_t<ReturnTt> 
    {
        auto ttscopy = std::tuple(Ts(std::forward<Ts>(inputs))...);

        std::vector<htorch::jit::IValue> ivalue_inputs;
        ivalue_inputs.reserve(sizeof...(InputTt));

        // Iterate over tensor_prototype_containers and convert them
        for_sequence<sizeof...(Ts)>([&](auto i) {
            ivalue_inputs.push_back(
                to_ivalue(std::get<i>(ttscopy))
            );
        });

        // Execute
        htorch::jit::IValue ret_ivalue;
        //if (_script_unit.)

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

    template<typename T>
    static htorch::jit::IValue to_ivalue(T&& container) {
        using U = std::remove_cvref_t<T>;

        if constexpr (is_tensor<U>) {
            return tensor_to_ivalue(std::move(container));
        } else if constexpr (is_specialization_of<U, std::vector>) {
            return vector_to_ivalue(std::move(container));
        } else if constexpr (is_specialization_of<U, std::unordered_map>) {
            return map_to_ivalue(std::move(container));
        } else if constexpr (is_specialization_of<U, std::tuple>) {
            return tuple_to_ivalue(std::move(container));
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
        std::vector<htorch::jit::IValue> ivalue_list;
        ivalue_list.reserve(vec.size());
        for (auto&& elem : vec) {
            ivalue_list.push_back(to_ivalue(std::move(elem)));
        }
        return htorch::jit::IValue(std::move(ivalue_list));
    }

    template<is_tensor_container Map>
    static htorch::jit::IValue map_to_ivalue(Map&& map) {
        using U = std::remove_cvref_t<Map>;
        using key_type = typename U::key_type;
        using mapped_type = typename U::mapped_type;

        std::unordered_map<key_type, htorch::jit::IValue> ivalue_map;
        ivalue_map.reserve(map.size());
        for (auto&& [key, val] : map) {
            ivalue_map.insert(key, to_ivalue(std::move(val)));
        }
        return htorch::jit::IValue(std::move(ivalue_map));
    }

    template<is_tensor_container Tup>
    static htorch::jit::IValue tuple_to_ivalue(Tup&& tup) {
        return htorch::jit::IValue(
            std::apply([](auto&&... elements) {
                return std::make_tuple(to_ivalue(std::move(elements))...);
            }, std::forward<Tup>(tup))
        );
    }


};

export template<typename T>
concept is_runnable_script = requires {
    typename T::ReturnType;
    typename T::InputTraits;
} && is_specialization_of<T, runnable_script>;


template<is_tensor_prototype_container T>
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
            result += get_torchscript_type_string<std::tuple_element<i, U>>();
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

export template<is_tensor_prototype_container ReturnTt, is_tensor_prototype_container... InputTt>
struct compiled_script_builder {
private:
    std::string _funcname;
    std::string _scriptstr;
    std::string _forwardname;
    std::string _compiled;
    std::tuple<InputTt...> _script_tensors;
    
    bool _freeze = true;
    bool _optimize_for_inference = true;

    uptr<Module> _mod;

public:

    using ReturnType = ReturnTt;
    using InputTraits = TupleTraits<InputTt...>;

    compiled_script_builder(
        std::string_view funcname,
        std::string_view code,
        InputTt&&... tts
    )
        :
        _funcname(funcname),
        _scriptstr(code),
        _script_tensors(
            std::forward<InputTt>(tts)...
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
        for_sequence<InputTraits::Size>([this, &inputvars](auto i) {
            auto& itt = std::get<i>(_script_tensors);
            inputvars += ", " + itt.str() + ": " + get_torchscript_type_string<InputTraits::template Nth<i>>();
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



// Type converters

template<typename Builder>
struct builder_to_runnable;

template<is_tensor_prototype_container ReturnTt, is_tensor_prototype_container... InputTt>
struct builder_to_runnable<compiled_script_builder<ReturnTt, InputTt...>> {
    using type = runnable_script<ReturnTt, InputTt...>;
};

export template<is_compiled_script_builder Builder>
using builder_to_runnable_t = typename builder_to_runnable<Builder>::type;

template<typename Runnable>
struct runnable_to_compiled_builder;

template<is_tensor_prototype_container ReturnTt, is_tensor_prototype_container... InputTt>
struct runnable_to_compiled_builder<runnable_script<ReturnTt, InputTt...>> {
    using type = compiled_script_builder<ReturnTt, InputTt...>;
};

export template<is_runnable_script Runnable>
using runnable_to_compiled_builder_t = typename runnable_to_compiled_builder<Runnable>::type;



}
}
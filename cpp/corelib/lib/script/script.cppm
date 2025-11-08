module;

export module script;

import torch_base;
import util;
import tensor_proto;

namespace hasty {
    namespace script {

        export using CompilationUnit = htorch::jit::CompilationUnit;
    	export using Module = htorch::jit::Module;


        struct runnable_script_unit {
        public:
            
            runnable_script_unit(uptr<CompilationUnit>&& cu)
                : _compilation_unit(std::move(cu)),
                  _module(nullptr)
            {}

            runnable_script_unit(uptr<Module>&& mod)
                : _compilation_unit(nullptr),
                  _module(std::move(mod))
            {}

        private:
            uptr<CompilationUnit> _compilation_unit;
            uptr<Module> _module;
        };

        export template<typename ReturnT, typename... InputTs>
        struct runnable_script;

        export template<is_tensor_prototype_container ReturnTt, is_tensor_prototype_container... InputTt>
        struct runnable_script {
        private:
            std::string _script_name;
            runnable_script_unit _script_unit;
        public:

            runnable_script(std::string_view script_name, runnable_script_unit&& script_unit)
                : _script_name(script_name),
                  _script_unit(std::move(script_unit))
            {}

            template<typename... Ts>
            requires (sizeof...(Ts) == sizeof...(InputTt)) &&
                    ((std::same_as<std::remove_cvref_t<Ts>, 
                                tensor_prototype_container_conversion_t<InputTt>> ||
                    std::convertible_to<Ts, tensor_prototype_container_conversion_t<InputTt>>) && ...)
            auto run(Ts&&... inputs) const -> tensor_prototype_container_conversion_t<ReturnTt> 
            {

            }

        }

    }
}
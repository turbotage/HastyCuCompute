module;

#include "pch.hpp"

export module trace;

export import util;
export import tensor;

namespace hasty {
    namespace trace {

        export template<device_fp FPT, size_t RANK>
        class tensor_prototype {
        public:

            using tensor_device_fp = FPT;
            static constexpr std::integral_constant<size_t, RANK> size = {};

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
            []<device_fp FPT, size_t RANK>(tensor_prototype<FPT, RANK>&){}(t);
        };

        template<typename T>
        struct tensor_prototype_conversion;

        template<is_tensor_prototype T>
        struct tensor_prototype_conversion<T> {
            tensor<typename T::tensor_device_fp, T::size()> operator()(tensor_prototype<typename T::tensor_device_fp, T::size()> t);
        };

        template<is_tensor T>
        struct tensor_prototype_conversion<T> {
            tensor_prototype<typename T::tensor_device_fp, T::size()> operator()(tensor<typename T::tensor_device_fp, T::size()> t);
        };

        export template<is_tensor_prototype T>
        using tensor_prototype_conversion_t = std::invoke_result_t<tensor_prototype_conversion<T>, T>;

        export template<typename... U>
        struct trace_function;

        export template<is_tensor_prototype... ReturnTt, is_tensor_prototype... InputTt>
        struct trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>> {
        private:
            std::string _funcname;
            std::string _tracestr;
            std::shared_ptr<torch::jit::CompilationUnit> _cu;
            std::tuple<InputTt...> _trace_tensors;
        public:

            using ReturnTraits = TupleTraits<ReturnTt...>;
            using InputTraits = TupleTraits<InputTt...>;

            trace_function(const std::string& funcname, InputTt&&... tts)
                : _funcname(funcname), _cu(nullptr), _trace_tensors(std::forward<InputTt>(tts)...)
            {
                reset();
            }

            void reset() {
                _tracestr = "";
                _cu = nullptr;

                std::string variables = for_sequence<sizeof...(InputTt)>([this](auto i, std::string& currentstr){
                    currentstr += std::get<i>(_trace_tensors).str();
                    if constexpr(i < sizeof...(InputTt) - 1) {
                        currentstr += ",";
                    }
                }, std::string(""));

                _tracestr = std::format("def {}({}):", _funcname, variables);
            }

            void add_line(const std::string& line)
            {
                _tracestr += "\n\t" + line;
            }

            void add_lines(const std::string& lines)
            {
                _tracestr += "\n" + lines;
            }

            const std::string& str() const {
                return _tracestr;
            }

            void compile() {
                if (_cu != nullptr) {
                    throw std::runtime_error("CompilationUnit already exists");
                }
                _cu = torch::jit::compile(_tracestr);
            }

            template<is_tensor ...Ts>
            auto run(Ts&&... tts) -> std::tuple<tensor_prototype_conversion_t<ReturnTt>...> {
                
                //auto ttstup = std::make_tuple(tts...);
                using TsTraits = TupleTraits<Ts...>;

                for_sequence<sizeof...(Ts)>([](auto i) {
                    using TS_TN = std::remove_reference_t<typename TsTraits::template Nth<i>>;
                    using INPUT_TN = std::remove_reference_t<typename InputTraits::template Nth<i>>;
                    using ts_fp = typename TS_TN::tensor_device_fp;
                    using input_fp = typename INPUT_TN::tensor_device_fp;
                    static_assert(std::is_same_v<ts_fp, input_fp>, "device_fp types must be the same");
                    static_assert(TS_TN::size() == INPUT_TN::size(), "Sizes must be the same");
                });

                std::tuple<tensor_prototype_conversion_t<ReturnTt>...> rets;

                auto gettensorfunc = []<typename T>(T t) { return t.get_tensor(); };

                //std::tuple<at::Tensor> input_tensors = std::make_tuple(gettensorfunc(tts)...);

                //c10::IValue ret_ivalue = _cu->run_method(_funcname, std::forward<Ts>(tts)...);
                c10::IValue ret_ivalue = _cu->run_method(_funcname, gettensorfunc(tts)...);

                if (ret_ivalue.isTensor()) {
                    if (sizeof...(ReturnTt) > 1) {
                        throw std::runtime_error("ReturnTt indicated multiple return values, but only one was returned");
                    }

                    using RT = std::remove_reference_t<typename ReturnTraits::First>;
                    using RTF = typename RT::tensor_device_fp;

                    auto retten = ret_ivalue.toTensor();

                    tensor_factory<RTF, RT::size()>::assign(std::get<0>(rets), ret_ivalue.toTensor());

                } else if (ret_ivalue.isTuple()) {
                    auto tuple_ptr = ret_ivalue.toTuple();
                    auto& elements = tuple_ptr->elements();
                    
                    if (elements.size() != sizeof...(ReturnTt)) {
                        throw std::runtime_error("Tuple size did not match return size");
                    }

                    for_sequence<sizeof...(ReturnTt)>([this, &rets, &elements](auto i) {
                        auto& element = elements[i];
                        if (element.isTensor()) {
                            using RT = std::remove_reference_t<typename ReturnTraits::template Nth<i>>;
                            using RTF = typename RT::tensor_device_fp;

                            tensor_factory<RTF, RT::size()>::assign(std::get<i>(rets), element.toTensor());
                        } else {
                            throw std::runtime_error("Return type inside tuple was not a Tensor");
                        }
                    });

                } else {
                    throw std::runtime_error("Return type was neither a Tensor nor a Tuple");
                }

                return rets;
            }

        };


        export template<is_tensor_prototype... ReturnTt>
        struct trace_function_factory {

            template<is_tensor_prototype... InputTt>
            static auto make(const std::string& funcname, InputTt&&... tts) {
                return trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>>(
                    funcname, std::forward<InputTt>(tts)...);
            }

            template<is_tensor_prototype... InputTt>
            static auto make_unique(const std::string& funcname, InputTt&&... tts) {
                return std::make_unique<trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>>>(
                    funcname, std::forward<InputTt>(tts)...);
            }

        };


        /*
        export struct trace_scope {
            
            trace_scope(const std::string& scopestr)
                : _tracestr(scopestr) {}

            void add_scope(trace_scope scope) {
                _tracescopes.push_back(std::move(scope));
            }

            void print(std::string& retstr, int indent) {
                retstr += _tracestr + "\n";
                for (auto& scope : _tracescopes) {
                    scope.print(retstr, indent + 1);
                }
            }

            std::string _tracestr;
            std::vector<trace_scope> _tracescopes;
        };
        */


    }

}

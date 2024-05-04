module;

#include "pch.hpp"

export module trace;

import util;
import tensor;

namespace hasty {
    namespace trace {

        export template<is_device D, is_tensor_type TT, size_t RANK>
        class tensor_prototype {
        public:

            using device_type_t = D;
            using tensor_type_t = TT;
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
            []<is_device D, is_tensor_type TT, size_t RANK>(tensor_prototype<D,TT,RANK>&){}(t);
        };

        template<typename T>
        struct tensor_prototype_conversion;

        template<is_tensor_prototype T>
        struct tensor_prototype_conversion<T> {
            tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(
                tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()> t);
        };

        template<is_tensor T>
        struct tensor_prototype_conversion<T> {
            tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(
                tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()> t);
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
                    using ts_device = typename TS_TN::device_type_t;
                    using input_device = typename INPUT_TN::device_type_t;
                    using ts_tensor_type = typename TS_TN::tensor_type_t;
                    using input_tensor_type = typename INPUT_TN::tensor_type_t;
                    static_assert(std::is_same_v<ts_device, input_device>, "device_types must be the same");
                    static_assert(std::is_same_v<ts_tensor_type, input_tensor_type>, "tensor_types must be the same");
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

                    std::get<0>(rets).assign(ret_ivalue.toTensor());
                    
                } else if (ret_ivalue.isTuple()) {
                    auto tuple_ptr = ret_ivalue.toTuple();
                    auto& elements = tuple_ptr->elements();
                    
                    if (elements.size() != sizeof...(ReturnTt)) {
                        throw std::runtime_error("Tuple size did not match return size");
                    }

                    for_sequence<sizeof...(ReturnTt)>([this, &rets, &elements](auto i) {
                        auto& element = elements[i];
                        if (element.isTensor()) {
                            std::get<i>(rets).assign(element.toTensor());
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

    }

}

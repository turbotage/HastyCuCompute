module;

#include "pch.hpp"

export module trace;

export import util;

namespace hasty {
    namespace trace {

        export template<device_fp FPT, size_t RANK>
        class trace_tensor {
        public:

            trace_tensor(std::string name)
                : _tracestr(name)
            {};

        private:
            std::string _tracestr;
        };

        export template<typename T>
        concept is_trace_tensor = requires(T t) {
            []<device_fp FPT, size_t RANK>(trace_tensor<FPT, RANK>&){}(t);
        };



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

        export template<device_fp FP, size_t RETRANK, is_trace_tensor ...Tt>
        struct trace_function {

            trace_function(const std::string& funcname, Tt... tts)
            {
                auto ttstup = std::make_tuple(tts...);

                std::string variables = for_sequence<sizeof...(Tt)>([&ttstup](auto i, std::string& currentstr){
                    currentstr += std::get<i>(ttstup).name();
                    if constexpr(i < sizeof...(Tt) - 1) {
                        currentstr += ",";
                    }
                }, std::string(""));

                _tracestr = std::format("def {}({}):", funcname, variables);
            }

            void add_line(const std::string& line)
            {
                _tracestr += "\n\t" + line;
            }

            const std::string& str() const {
                return _tracestr;
            }

            template<is_tensor ...Ts>
            void run(tensor<FP, RETRANK>& ret, Ts... tts) {
                
                auto tensor_rank = []<device_fp FPT, size_t RANK>() constexpr {
                    
                }

                auto ttstup = std::make_tuple(tts...)
                for_sequence<sizeof...(Ts)>([])

            }

        private:
            auto 
            std::string _tracestr;
        };

    }

}

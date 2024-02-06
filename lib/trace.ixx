module;

#include "pch.hpp"

export module trace;

import tensor;
import util;

namespace hasty {


    export template<device_fp FPT, size_t RANK>
    struct trace_tensor {


        trace_tensor(std::string name)
            : _tracestr(name)
        {
        };

        std::string name() const { return _tracestr; }

        template<index_type ...Idx>
        auto operator[](Idx... indices) {
            std::string newtracestr = _tracestr;

            constexpr auto RETRANK = get_slice_rank(indices...);

            auto idxss = std::make_tuple(indices...);

            newtracestr += "[";
            for_sequence<sizeof...(Idx)>([&](auto i) {
                newtracestr += torchidxstr(std::get<i>(idxss)) + ",";
            });
            newtracestr += "]";

            return trace_tensor<FPT, RETRANK>(newtracestr);
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator=(const trace_tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".copy_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator+=(const trace_tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".add_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator-=(const trace_tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".sub_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator*=(const trace_tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".mul_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator/=(const trace_tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".div_(" + other._tracestr + ");";
            return newtracestr;
        }

        trace_tensor conj() {
            return trace_tensor<FPT, RANK>(_tracestr + ".conj()");
        }

        
    private:
        std::string _tracestr;
    };

    export template<typename T>
    concept is_trace_tensor = requires(T t) {
        []<device_fp FPT, size_t RANK>(trace_tensor<FPT, RANK>&){}(t);
    };

    export template<is_trace_tensor ...Tt>
    struct trace_function {

        trace_function(const std::string& funcname, Tt... tts)
        {
            auto ttstup = std::make_tuple(tts...);

            std::string variables = for_seqyence<sizeof...(Tt)>([&ttstup](auto i, std::string& currentstr){
                currentstr += std::get<i>(ttstup).name();
                if constexpr(i < sizeof...(Tt) - 1) {
                    currentstr += ",";
                }
            }, "");


            _tracestr = std::format("def {}({}):", funcname, variables);
        }

        void add_line(const std::string& line)
        {
            _tracestr += "\n\t" + line;
        }

    private:
        std::string _tracestr;
    };



}

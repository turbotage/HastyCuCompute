module;

#include "pch.hpp"

export module trace;

import util;

namespace hasty {

    export struct trace {

        trace()
        {
            std::vector<at::Tensor> tensorlist;
        }

        void add_line(const std::string& line)
        {
            _tracestr += "\n\t" + line;
        }

    private:
        std::unordered_map<void*,at::Tensor> _tensors;
        std::string _tracestr = 
R"torchscript(
def trace_function(tensorlist):
)torchscript";
    };

    export template<device_fp FPT, size_t RANK>
    export struct trace_tensor {


        trace_tensor(std::string name)
            : _tracestr(name)
        {
        };

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
        std::string operator+=(const tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".add_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator-=(const tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".sub_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator*=(const tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".mul_(" + other._tracestr + ");";
            return newtracestr;
        }

        template<size_t R>
        requires less_than<R, RANK>
        std::string operator/=(const tensor<FPT, R>& other) {
            std::string newtracestr = _tracestr;
            newtracestr += ".div_(" + other._tracestr + ");";
            return newtracestr;
        }



    private:
        std::string _tracestr;
    };


}

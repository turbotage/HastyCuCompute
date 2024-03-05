module;

#include "pch.hpp"

export module trace;

import tensor;
import util;

namespace hasty {

    template<std::integral I, size_t R>
    constexpr std::string to_trace_str(span<I,R> arr, bool as_tuple = true) {
        std::string retstr = as_tuple ? "(" : "[";
        
        for_sequence<R>([&](auto i) {
            retstr += std::to_string(arr.template get<i>());
            if constexpr(i < R - 1) {
                retstr += ",";
            }
        });
        retstr += as_tuple ? ")" : "]";
        return retstr;
    }

    export template<device_fp FPT, size_t RANK>
    struct trace_tensor {


        trace_tensor(std::string name)
            : _tracestr(name)
        {
            std::cout << "constructor was run" << std::endl;
        };

        template<device_fp F, size_t R>
        requires less_than_or_equal<R, RANK>
        auto operator=(const trace_tensor<F, R>& other) {
            std::cout << "op= was run" << std::endl;
            std::string newtracestr = _tracestr;
            newtracestr += ".copy_(" + other._tracestr + ");";
            return newtracestr;
        }

        trace_tensor& operator=(const trace_tensor&) = delete;

        std::string name() const { return _tracestr; }

        trace_tensor<FPT, RANK> shape() {
            return std::format("{}.shape", _tracestr);
        }

        template<size_t R>
        requires less_than<R, RANK>
        trace_tensor<FPT,RANK> shape() {
            return std::format("{}.shape[{}]", _tracestr, R);
        }

        std::string ndim() const {
            return std::to_string(RANK);
        }

        std::string nelem() const {
            return std::format("{}.numel()", _tracestr);
        }

        template<cpu_fp F>
        std::string fill_(F val) {
            return std::format("{}.fill_({})", _tracestr, val);
        }

        trace_tensor<FPT, RANK> clone() const
        {
            return trace_tensor<FPT,RANK>(_tracestr + ".clone()");
        }

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

    export enum struct fft_norm {
        FORWARD,
        BACKWARD,
        ORTHO
    };

/*
    export template<device_fp FPT, size_t RANK, size_t R, std::integral I>
    requires less_than<R, RANK>
    trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
        ospan<int32_t, R> s = std::nullopt, ospan<I, R> dim = std::nullopt, 
        std::optional<fft_norm> norm = std::nullopt) {

        auto normstr = [&norm]() {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return "forward";
                case fft_norm::BACKWARD:
                    return "backward";
                case fft_norm::ORTHO:
                    return "ortho";
                }
            }
        };

        return trace_tensor<FPT, RANK>(std::format("torch.fft.fftn({}{}{}{})", 
            t.name(),
            s.has_value() ? ",s=" + to_trace_str(*s) : "",
            dim.has_value() ? ",dim=" + to_trace_str(*dim) : "",
            norm.has_value() ? ",norm=" + normstr() : ""));
    }
*/

    export template<device_fp FPT, size_t RANK, size_t R1, size_t R2, std::integral I1, std::integral I2>
    requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
    trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
        ospan<I1,R1> s, 
        ospan<I2,R2> dim, 
        std::optional<fft_norm> norm = std::nullopt) {

        auto normstr = [&norm]() {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return std::string("forward");
                case fft_norm::BACKWARD:
                    return std::string("backward");
                case fft_norm::ORTHO:
                    return std::string("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
        };

        return trace_tensor<FPT, RANK>(std::format("torch.fft.fftn({}{}{}{})", 
            t.name(),
            s.has_value() ? ",s=" + to_trace_str(*s) : "",
            dim.has_value() ? ",dim=" + to_trace_str(*dim) : "",
            norm.has_value() ? ",norm=" + normstr() : ""));
    }


    /*
    export template<device_fp FPT, size_t RANK>
    trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t)
    {
        return trace_tensor<FPT, RANK>(std::format("torch.fft.fftn({})", t.name()));
    }
    */



    export template<device_fp FPT, size_t RANK, size_t R1, size_t R2, std::integral I1, std::integral I2>
    requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
    trace_tensor<FPT, RANK> ifftn(const trace_tensor<FPT, RANK>& t, 
        ospan<I1,R1> s, 
        ospan<I2,R2> dim, 
        std::optional<fft_norm> norm) {

        auto normstr = [&norm]() {
            if (norm.has_value()) {
                switch (norm.value()) {
                case fft_norm::FORWARD:
                    return std::string("forward");
                case fft_norm::BACKWARD:
                    return std::string("backward");
                case fft_norm::ORTHO:
                    return std::string("ortho");
                default:
                    throw std::runtime_error("Invalid fft_norm value");
                }
            }
        };

        return trace_tensor<FPT, RANK>(std::format("torch.fft.ifftn({}{}{}{})", 
            t.name(),
            s.has_value() ? ",s=" + to_trace_str(*s) : "",
            dim.has_value() ? ",dim=" + to_trace_str(*dim) : "",
            norm.has_value() ? ",norm=" + normstr() : "")
        );
    }


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

    export template<is_trace_tensor ...Tt>
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

    private:
        std::string _tracestr;
    };



}

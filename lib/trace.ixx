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

            trace_tensor& operator=(const trace_tensor& other) = delete;

            template<device_fp F, size_t R>
            requires less_than_or_equal<R, RANK>
            std::string operator=(const trace_tensor<F, R>& other) const {
                std::string newtracestr = _tracestr;
                newtracestr += "[:] = " + other._tracestr;
                return newtracestr;
            }

            std::string name() const { return _tracestr; }

            trace_tensor<FPT, RANK> shape() {
                return std::format("{}.shape", _tracestr);
            }

            template<std::integral I>
            trace_tensor<FPT, RANK> shape(std::optional<I> start = std::nullopt,
                std::optional<I> end = std::nullopt,
                std::optional<I> step = std::nullopt) 
            {
                if (start.has_value() && end.has_value() && step.has_value()) {
                    return std::format("{}.shape[{}:{}:{}]", _tracestr, start.value(), end.value(), step.value());
                } else if (start.has_value() && end.has_value()) {
                    return std::format("{}.shape[{}:{}]", _tracestr, start.value(), end.value());
                } else if (start.has_value() && step.has_value()) {
                    return std::format("{}.shape[{}::{}]", _tracestr, start.value(), step.value());
                } else if (end.has_value() && step.has_value()) {
                    return std::format("{}.shape[:{}:{}]", _tracestr, end.value(), step.value());
                } else if (start.has_value()) {
                    return std::format("{}.shape[{}]", _tracestr, start.value());
                } else if (end.has_value()) {
                    return std::format("{}.shape[:{}]", _tracestr, end.value());
                } else if (step.has_value()) {
                    return std::format("{}.shape[::{}]", _tracestr, step.value());
                } else {
                    return std::format("{}.shape", _tracestr);
                }
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
                return std::format("{}[:] = {}", _tracestr, val);
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
                newtracestr += "[:] += " + other._tracestr;
                return newtracestr;
            }

            template<size_t R>
            requires less_than<R, RANK>
            std::string operator-=(const trace_tensor<FPT, R>& other) {
                std::string newtracestr = _tracestr;
                newtracestr += "[:] -= " + other._tracestr;
                return newtracestr;
            }

            template<size_t R>
            requires less_than<R, RANK>
            std::string operator*=(const trace_tensor<FPT, R>& other) {
                std::string newtracestr = _tracestr;
                newtracestr += "[:] *= " + other._tracestr;
                return newtracestr;
            }

            template<size_t R>
            requires less_than<R, RANK>
            std::string operator/=(const trace_tensor<FPT, R>& other) {
                std::string newtracestr = _tracestr;
                newtracestr += "[:] /= " + other._tracestr + ");";
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

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
            span<R1> s = nullspan(), 
            span<R2> dim = nullspan(), 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_fftn({},{},{},{})", 
                t.name(),
                s.has_value() ? span_to_str(s, false) : "None",
                dim.has_value() ? span_to_str(dim, false) : "None",
                norm.has_value() ? normstr() : ""));
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
            std::optional<std::string> s = std::nullopt, 
            span<R2> dim = nullspan(), 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_fftn({},{},{},{})", 
                t.name(),
                s.has_value() ? s.value() : "None",
                dim.has_value() ? span_to_str(dim, false) : "None",
                norm.has_value() ? normstr() : ""));
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
            span<R1> s = nullspan(), 
            std::optional<std::string> dim = std::nullopt, 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_fftn({},{},{},{})", 
                t.name(),
                s.has_value() ? span_to_str(s, false) : "None",
                dim.has_value() ? dim.value() : "None",
                norm.has_value() ? normstr() : ""));
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> fftn(const trace_tensor<FPT, RANK>& t, 
            std::optional<std::string> s = std::nullopt, 
            std::optional<std::string> dim = std::nullopt, 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_fftn({},{},{},{})", 
                t.name(),
                s.has_value() ? s.value() : "None",
                dim.has_value() ? dim.value() : "None",
                norm.has_value() ? normstr() : ""));
        }



        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> ifftn(const trace_tensor<FPT, RANK>& t, 
            span<R1> s = nullspan(), 
            span<R2> dim = nullspan(), 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_ifftn({}{}{}{})", 
                t.name(),
                s.has_value() ? span_to_str(s, false) : "None",
                dim.has_value() ? span_to_str(dim, false) : "None",
                norm.has_value() ? normstr() : "")
            );
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> ifftn(const trace_tensor<FPT, RANK>& t, 
            std::optional<std::string> s = std::nullopt, 
            span<R2> dim = nullspan(), 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_ifftn({}{}{}{})", 
                t.name(),
                s.has_value() ? s.value() : "None",
                dim.has_value() ? span_to_str(dim, false) : "None",
                norm.has_value() ? normstr() : "")
            );
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> ifftn(const trace_tensor<FPT, RANK>& t, 
            span<R1> s = nullspan(), 
            std::optional<std::string> dim = std::nullopt, 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_ifftn({}{}{}{})", 
                t.name(),
                s.has_value() ? span_to_str(s, false) : "None",
                dim.has_value() ? dim.value() : "None",
                norm.has_value() ? normstr() : "")
            );
        }

        export template<device_fp FPT, size_t RANK, size_t R1, size_t R2>
        requires less_than_or_equal<R1, RANK> && less_than_or_equal<R2, RANK>
        trace_tensor<FPT, RANK> ifftn(const trace_tensor<FPT, RANK>& t, 
            std::optional<std::string> s = std::nullopt,
            std::optional<std::string> dim = std::nullopt, 
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
                throw std::runtime_error("This should not be possible");
            };

            return trace_tensor<FPT, RANK>(std::format("torch.fft_ifftn({}{}{}{})", 
                t.name(),
                s.has_value() ? s.value() : "None",
                dim.has_value() ? dim.value() : "None",
                norm.has_value() ? normstr() : "")
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

            const std::string& str() const {
                return _tracestr;
            }

        private:
            std::string _tracestr;
        };

    }

}

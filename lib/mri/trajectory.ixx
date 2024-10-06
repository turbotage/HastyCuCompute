module;

#include "pch.hpp"

export module trajectory;

//import pch;

import util;
import tensor;

namespace hasty {

    export template<is_fp_real_tensor_type TT, size_t DIM>
    class trajectory {
    public:

        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, DIM> dim = {};

        trajectory(std::array<cache_tensor<TT,1>,DIM> coords, TT echo)
            : _coords(std::move(coords)), _echo(echo)
        {}
            
        trajectory(std::array<cache_tensor<TT,1>,DIM> coords, cache_tensor<TT,1> time)
            : _coords(std::move(coords)), _time(time)
        {}

        std::array<cache_tensor<TT,1>,DIM>& coords() {
            return _coords;
        }

        const std::array<cache_tensor<TT,1>,DIM>& coords() const {
            return _coords;
        }

        std::optional<cache_tensor<TT,1>>& time() {
            return _time;
        }

        const std::optional<cache_tensor<TT,1>>& time() const {
            return _time;
        }

        std::optional<TT>& echo() {
            return _echo;
        }

        const std::optional<TT>& echo() const {
            return _echo;
        }

    private:
        std::array<cache_tensor<TT,1>,DIM> _coords;
        std::optional<cache_tensor<TT,1>> _time;
        std::optional<TT> _echo;
    };

}
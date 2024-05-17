module;

#include "../pch.hpp"

export module trajectory;

import util;
import tensor;

namespace hasty {

    export template<is_device D, is_fp_real_tensor_type TT, size_t DIM>
    class trajectory {
    public:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, DIM> dim = {};

        trajectory(std::array<tensor<D,TT,1>,DIM> coords, TT echo)
            : _coords(std::move(coords)), _echo(echo)
        {}
            
        trajectory(std::array<tensor<D,TT,1>,DIM> coords, tensor<D,TT,1> time)
            : _coords(std::move(coords)), _time(time)
        {}

    private:
        std::array<tensor<D,TT,1>,DIM> _coords;
        std::optional<tensor<D,TT,1>> _time;
        std::optional<TT> _echo;
    };

}
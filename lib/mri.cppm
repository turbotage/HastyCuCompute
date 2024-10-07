module;

#include "pch.hpp"

export module tensor;

//import pch;

import util;
import tensor;

namespace hasty {

    export template<device_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    class trajectory {
    public:

        //trajectory() = default;
        trajectory(const hasty::tensor<FPT, 2>& coords)
            : _coords(coords)
        {}

        trajectory(const hasty::tensor<FPT, 2>& coords, std::unique_ptr<hasty::tensor<FPT, 1>> timing)
            : _coords(coords), _timing(std::move(timing))
        {}

    private:
        std::unique_ptr<hasty::tensor<FPT, 1>> _timing;
        hasty::tensor<FPT, 2> _coords;
    };


    struct NormalOffResonance {

        template<device_fp FPT, size_t DIM>
        NormalOffResonance(trajectory<FPT, DIM>& traj)
            : _traj(traj)
        {


        }

    };




}
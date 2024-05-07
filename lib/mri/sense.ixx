module;

#include "../pch.hpp"

export module sense;

import tensor;

namespace hasty {

    template<is_device D, is_fp_tensor_type TT, size_t DIM>
    class sense_normal {
    public:

        using device_type_t = D;
        using input_tensor_type_t = TT;
        using output_tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
        static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

        tensor<device_type_t,input_tensor_type_t,input_rank_t> operator()(const tensor<device_type_t,input_tensor_type_t,input_rank_t>& x) {
            return x;
        }


    };

}
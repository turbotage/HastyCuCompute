module;

#include "pch.hpp"

export module tensor;

namespace hasty {

    template<is_device D, is_tensor_type TT, size_t RIN, size_t ROUT>
    concept tensor_operator = requires(const tensor<D,TT,RANK>& t) {
        { t.operator()(tensor<D,TT,RIN>()) } -> std::same_as<tensor<D,TT,ROUT>>;
        { t.adjoint() } -> tensor_operator<>
    }

    /*
    export template<is_device D, is_tensor_type TT, size_t RANK>
    class tensor_operator {
    private:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, RANK> size = {};

    public:




    }
    */

}

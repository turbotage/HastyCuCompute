module;

#include "../pch.hpp"

export module op;


import util;
import tensor;

namespace hasty {

    template<typename T, typename D, typename TIN, typename TOUT, typename RIN, typename ROUT>
    concept is_basic_tensor_operator = requires(T t) {
        typename T::device_type_t;
        typename T::input_tensor_type_t;
        typename T::output_tensor_type_t;
        requires is_device<D>;
        requires is_tensor_type<TIN>;
        requires is_tensor_type<TOUT>;
        { RIN() } -> std::same_as<size_t>;
        { ROUT() } -> std::same_as<size_t>;
        { RIN::value} -> std::same_as<size_t>;
        { ROUT::value} -> std::same_as<size_t>;

        /*
        requires std::same_as<typename T::input_tensor_type_t, TIN>;
        requires std::same_as<typename T::output_tensor_type_t, TOUT>;
        requires std::same_as<typename T::input_rank_t, RIN>;
        requires std::same_as<typename T::output_rank_t, ROUT>;
        */

        // Ensure the operator has a call operator that takes a tensor of type TIN and returns a tensor of type TOUT.
        { t.operator()(tensor<D,TIN,RIN::value>()) } -> std::same_as<tensor<D,TOUT,ROUT::value>>;
    };

    export template<typename T>
    concept is_tensor_operator = requires(T t) {

        requires is_basic_tensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

    };

    export template<typename T>
    concept is_adjointable_tensor_operator = requires(T t) {
        
        requires is_basic_tensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires is_basic_tensor_operator<decltype(t.adjoint()), typename T::device_type_t, 
                    typename T::output_tensor_type_t, typename T::input_tensor_type_t, 
                    typename T::output_rank_t, typename T::input_rank_t>;

    };

    export template<typename T>
    concept is_normalable_tensor_operator = requires(T t) {
        
        requires is_basic_tensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires is_basic_tensor_operator<decltype(t.normal()), typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::input_tensor_type_t, 
                    typename T::input_rank_t, typename T::input_rank_t>;

    };

    export template<typename T>
    concept is_square_tensor_operator = requires(T t) {
        
        requires is_basic_tensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires T::input_rank_t::value == T::output_rank_t::value;

    };

    export template<typename T>
    concept is_hom_tensor_operator = requires(T t) {

        requires is_basic_tensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires std::same_as<typename T::input_tensor_type_t, typename T::output_tensor_type_t>;

    };

    export template<typename T>
    concept is_hom_square_tensor_operator = requires(T t) {

        requires is_square_tensor_operator<T>;

        requires is_hom_tensor_operator<T>;

    };




    export template<is_device D, is_tensor_type TT, size_t RANK>
    class optensor {
    private:
        std::vector<tensor<D,TT,RANK>> _tensors;
    public:

        using device_type_t = D;
        using tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, RANK> size = {};

        auto operator[](size_t idx) -> tensor<D,TT,RANK>&
        {
            return _tensors[idx];
        }

    };

    export template<typename T>
    concept is_optensor = requires(T t) {
        []<is_device D, is_tensor_type TT, size_t RANK>(optensor<D,TT,RANK>&){}(t);
    };

    template<typename T, typename D, typename TIN, typename TOUT, typename RIN, typename ROUT>
    concept is_basic_optensor_operator = requires(T t) {
        typename T::device_type_t;
        typename T::input_tensor_type_t;
        typename T::output_tensor_type_t;
        requires is_device<D>;
        requires is_tensor_type<TIN>;
        requires is_tensor_type<TOUT>;
        { RIN() } -> std::same_as<size_t>;
        { ROUT() } -> std::same_as<size_t>;
        { RIN::value} -> std::same_as<size_t>;
        { ROUT::value} -> std::same_as<size_t>;

        // Ensure the operator has a call operator that takes a tensor of type TIN and returns a tensor of type TOUT.
        { t.operator()(optensor<D,TIN,RIN::value>()) } -> std::same_as<optensor<D,TOUT,ROUT::value>>;
    };

    export template<typename T>
    concept is_optensor_operator = requires(T t) {

        requires is_basic_optensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

    };

    export template<typename T>
    concept is_adjointable_optensor_operator = requires(T t) {
        
        requires is_basic_optensor_operator<T, typename T::device_type_t, typename T::input_tensor_type_t, 
                    typename T::output_tensor_type_t, typename T::input_rank_t, typename T::output_rank_t>;

        requires is_basic_optensor_operator<decltype(t.adjoint()), typename T::device_type_t, 
                    typename T::output_tensor_type_t, typename T::input_tensor_type_t, 
                    typename T::output_rank_t, typename T::input_rank_t>;

    };

    export template<typename T>
    concept is_normalable_optensor_operator = requires(T t) {
        
        requires is_basic_optensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires is_basic_optensor_operator<decltype(t.normal()), typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::input_tensor_type_t, 
                    typename T::input_rank_t, typename T::input_rank_t>;

    };

    export template<typename T>
    concept is_square_optensor_operator = requires(T t) {
        
        requires is_basic_optensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires T::input_rank_t::value == T::output_rank_t::value;

    };

    export template<typename T>
    concept is_hom_optensor_operator = requires(T t) {
            
        requires is_basic_optensor_operator<T, typename T::device_type_t, 
                    typename T::input_tensor_type_t, typename T::output_tensor_type_t, 
                    typename T::input_rank_t, typename T::output_rank_t>;

        requires std::same_as<typename T::input_tensor_type_t, typename T::output_tensor_type_t>;

    };

    export template<typename T>
    concept is_hom_square_optensor_operator = requires(T t) {

        requires is_square_optensor_operator<T>;

        requires is_hom_optensor_operator<T>;

    };




    export template<typename T>
    concept is_anytensor = is_tensor<T> || is_optensor<T>;

    export template<typename T>
    concept is_anytensor_operator = is_tensor_operator<T> || is_optensor_operator<T>;

    export template<typename T>
    concept is_normalable_anytensor_operator = is_normalable_tensor_operator<T> || is_normalable_optensor_operator<T>; 

    export template<typename T>
    concept is_square_anytensor_operator = is_square_tensor_operator<T> || is_square_optensor_operator<T>;

    export template<typename T>
    concept is_hom_anytensor_operator = is_hom_tensor_operator<T> || is_hom_optensor_operator<T>;

    export template<typename T>
    concept is_nom_square_anytensor_operator = is_hom_square_tensor_operator<T> || is_hom_square_optensor_operator<T>;





}

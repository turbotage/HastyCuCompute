module;

#include "../pch.hpp"

export module util:idx;

namespace hasty {

    export struct None {};

    export struct Ellipsis {};

    export struct Slice {

        Slice() = default;

        template<std::integral I>
        Slice(I ival) : start(ival) {}

        template<std::integral I1, std::integral I2>
        Slice(I1 ival1, I2 ival2) : start(ival1), end(ival2) {}

        std::optional<int64_t> start;
        std::optional<int64_t> end;
        std::optional<int64_t> step;
    };

    export template<typename T>
    concept index_type =   std::is_same_v<T, None> 
                        || std::is_same_v<T, Ellipsis> 
                        || std::is_same_v<T, Slice>
                        || std::is_integral_v<T>;

    export using TensorIndex = std::variant<None, Ellipsis, Slice, int64_t>;

    export template<size_t R, index_type... Idx>
    constexpr size_t get_slice_rank()
    {
        int none = 0;
        int ints = 0;
        int ellipsis = 0;

        ((std::is_same_v<Idx, None> ? ++none : 
        std::is_integral_v<Idx> ? ++ints : 
        std::is_same_v<Idx, Ellipsis> ? ++ellipsis : 0), ...);

        return R - ints + none;
    }

    export template<size_t R, index_type... Idx>
    constexpr size_t get_slice_rank(std::tuple<Idx...> idxs)
    {
        int none;
        int ints;
        int ellipsis;

        for_sequence<std::tuple_size_v<decltype(idxs)>>([&](auto i) constexpr {
            //if constexpr(std::is_same_v<decltype(idxs.template get<i>()), None>) {
            if constexpr(std::is_same_v<decltype(std::get<i>(idxs)), None>) {
                ++none;
            } 
            //else if constexpr(std::is_integral_v<decltype(idxs.template get<i>())>) {
            else if constexpr(std::is_integral_v<decltype(std::get<i>(idxs))>) {
                ++ints;
            }
            /*
            else if constexpr(std::is_same_v<decltype(idxss.template get<i>()), Ellipsis>) {
                ++ellipsis;
            } 
            */
        });

        return R - ints + none;
    }

    export template<size_t R, index_type... Itx>
    constexpr size_t get_slice_rank(Itx... idxs)
    {
        return get_slice_rank<R>(std::make_tuple(idxs...));
    }

}

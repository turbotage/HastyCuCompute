module;

#include "../pch.hpp"

export module util:meta;

namespace hasty {

    export {

        template<size_t T1, size_t T2>
        concept less_than = T1 < T2;

        template<size_t T1, size_t T2>
        concept less_than_or_equal = T1 <= T2;

        template<size_t T1, size_t T2>
        concept equal_or_one_zero = !((T1 != 0) && (T2 != 0) && (T1 != T2));

        template<size_t T1, size_t T2>
        concept equal_or_right_zero = (T1 == T2) || (T2 == 0);

        template<size_t T1, size_t T2>
        concept equal_or_left_zero = (T1 == T2) || (T1 == 0);

        template <typename T, T... S, typename F>
        constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
            (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
        }

        template<auto n, typename F>
        constexpr void for_sequence(F f) {
            for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
        }

        template<auto n, typename F, typename V>
        constexpr V for_sequence(F f, const V& t) {
            V tcopy = t;
            for_sequence<n>([&tcopy, &f](auto i) {
                f(i, tcopy);
            });
            return tcopy;
        }

    }

}
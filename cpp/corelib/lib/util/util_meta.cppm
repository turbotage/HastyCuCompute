module;

#include "pch.hpp"

export module util:meta;

//import pch;
import std;

namespace hasty {

export template<size_t T1, size_t T2>
concept less_than = T1 < T2;

export template<size_t T1, size_t T2>
concept less_than_or_equal = T1 <= T2;

export template<size_t T1, size_t T2>
concept equal_or_one_zero = !((T1 != 0) && (T2 != 0) && (T1 != T2));

export template<size_t T1, size_t T2>
concept equal_or_right_zero = (T1 == T2) || (T2 == 0);

export template<size_t T1, size_t T2>
concept equal_or_left_zero = (T1 == T2) || (T1 == 0);

export template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
	(static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

export template<auto n, typename F>
constexpr void for_sequence(F f) {
	for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

export template<auto n, typename F, typename V>
constexpr V for_sequence(F f, const V& t) {
	V tcopy = t;
	for_sequence<n>([&tcopy, &f](auto i) {
		f(i, tcopy);
	});
	return tcopy;
}

template<typename Tuple, typename F, std::size_t... I>
constexpr void for_each_type_impl(F&& f, std::index_sequence<I...>) {
	(f(std::tuple_element_t<I,Tuple>{}), ...);
}

export template<typename Tuple, typename F>
constexpr void for_each_type(F&& f) {
	for_each_type_impl<Tuple>(std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

export template<typename... Args>
struct TupleTraits {
	using Tuple = std::tuple<Args...>;
	static constexpr size_t Size = sizeof...(Args);

	template <std::size_t N>
	using Nth = typename std::tuple_element<N, Tuple>::type;

	using First = Nth<0>;
	using Last = Nth<Size - 1>;
};

export template<typename... Args>
struct TupleTraits<std::tuple<Args...>> {
	using Tuple = std::tuple<Args...>;
	static constexpr size_t Size = sizeof...(Args);

	template <std::size_t N>
	using Nth = typename std::tuple_element<N, Tuple>::type;

	using First = Nth<0>;
	using Last = Nth<Size - 1>;
};

export template<>
struct TupleTraits<> {
	using Tuple = std::tuple<>;
	static constexpr size_t Size = 0;
};

export template<>
struct TupleTraits<std::tuple<>> {
	using Tuple = std::tuple<>;
	static constexpr size_t Size = 0;
};

}
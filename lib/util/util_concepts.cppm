module;

#include "pch.hpp"

export module util:concepts;

import std;

namespace hasty {

    export template<typename T>
    concept is_string_appendable = requires(std::string s, T t) {
        { s += t} -> std::same_as<std::string&>;
    };

    export template<typename T>
    concept is_stringable = requires(const T& t) {
        { t.to_string() } -> std::convertible_to<std::string>;
    };

    export template<typename T>
    concept is_equal_comparable = requires(const T& a, const T& b) {
        { a == b } -> std::convertible_to<bool>;
    };

    export template<typename T>
    concept is_equal_comparable_and_stringable = is_equal_comparable<T> && is_stringable<T>;

}
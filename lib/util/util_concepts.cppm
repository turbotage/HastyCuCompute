module;

#include "pch.hpp"

export module util:concepts;

namespace hasty {

    export template<typename T>
    concept is_string_appendable = requires(std::string s, T t) {
        { s += t} -> std::same_as<std::string&>;
    };

}
module;

#include "pch.hpp"

export module util;

export import :funcs;
export import :idx;
export import :meta;
export import :span;
export import :torch;
export import :typing;


namespace hasty {

    export template<typename T>
    class move {
    public:
    
        explicit move(T&& obj) : _obj(std::move(obj)) {}

        // Deleted copy constructor and copy assignment operator
        move(const move&) = delete;
        move& operator=(const move&) = delete;

        // Deleted move constructor and move assignment operator
        move(move&&) = delete;
        move& operator=(move&&) = delete;

        // Access the underlying object
        T& get() { return _obj; }
        const T& get() const { return _obj; }

    private:
        T&& _obj;
    };
    
}
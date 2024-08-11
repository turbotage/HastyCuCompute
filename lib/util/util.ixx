module;

#include "pch.hpp"

export module util;

export import :funcs;
export import :idx;
export import :meta;
export import :span;
export import :torch;
export import :typing;

namespace debug {

    export void print_memory_usage(const std::string& prepend = "") {
        std::ifstream file("/proc/self/status");
        std::string line;
        while (std::getline(file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::cout << prepend << " Resident Set Size: " << line.substr(6) << std::endl;
            } else if (line.substr(0, 6) == "VmSize:") {
                std::cout << prepend << " Virtual Memory Size: " << line.substr(6) << std::endl;
            }
        }
    }

}

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
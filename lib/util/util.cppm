module;

#include "pch.hpp"

export module util;

//import pch;

export import :funcs;
export import :idx;
export import :meta;
export import :span;
export import :torch;
export import :typing;
export import :io;

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

    export enum struct device_idx {
        CPU = -1,
        CUDA0 = 0,
        CUDA1 = 1,
        CUDA2 = 2,
        CUDA3 = 3,
        CUDA4 = 4,
        CUDA5 = 5,
        CUDA6 = 6,
        CUDA7 = 7,
        CUDA8 = 8,
        CUDA9 = 9,
        CUDA10 = 10,
        CUDA11 = 11,
        CUDA12 = 12,
        CUDA13 = 13,
        CUDA14 = 14,
        CUDA15 = 15,
        MAX_CUDA_DEVICES = 16
    };

    export void synchronize() {
        torch::cuda::synchronize();
    }

    export void synchronize(device_idx idx) {
        torch::cuda::synchronize(i32(idx));
    }

}
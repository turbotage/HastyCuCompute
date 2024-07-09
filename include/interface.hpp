#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define LIB_EXPORT __declspec(dllexport)
#define LIB_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define LIB_EXPORT __attribute__((visibility("default")))
#define LIB_IMPORT
#else
#define LIB_EXPORT
#define LIB_IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

#include "pch.hpp"
#include <pybind11/pybind11.h>

namespace at {
    class Tensor;
}

namespace ffi {

    class LIB_EXPORT Tensor {
    public:
        Tensor(std::unique_ptr<at::Tensor> ten) 
            : _ten(std::move(ten))
        {}

        at::Tensor& get() { return *_ten; }

    private:
        std::unique_ptr<at::Tensor> _ten;
    };

    LIB_EXPORT void run_main();

}

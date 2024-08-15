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

/*
namespace at {
    class Tensor;
}
*/


namespace ffi {

    LIB_EXPORT std::vector<at::Tensor> test_simple_invert();

}


#include <test_interface.hpp>

import std;

int main() {

    //auto ret = ffi::test_simple_invert();

    std::cout << "Hello, World!" << std::endl;

    //auto ret = ffi::test_whitten_offresonance_operator();

    {
        ffi::jit_checking();
    }

    {
        ffi::test_normal_operators();
    }
    
    {
        ffi::test_prototype_stuff();
    }

    std::cout <<  "Goodbye, World!" << std::endl;

    return 0;

}


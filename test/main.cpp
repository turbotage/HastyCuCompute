#include <interface.hpp>


int main() {

    //auto ret = ffi::test_simple_invert();

    std::cout << "Hello, World!" << std::endl;

    //auto ret = ffi::test_whitten_offresonance_operator();

    auto ret = ffi::test_normal_operators();
    
    ffi::test_prototype_stuff();

    return 0;

}


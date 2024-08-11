//#include <interface.hpp>
#include <interface.hpp>

void print_memory_usage(const std::string& prepend = "");
at::Tensor test();

int main() {

    //ffi::run_main();
    c10::InferenceMode im_guard{};

    //auto output = ffi::leak_test();

    auto ret = ffi::leak_test2();

    //auto ret = test();

    //test();

    //auto output2 = ffi::test_simple_invert();

    return 0;

}

at::Tensor test() {
    c10::InferenceMode im_guard{};

    at::Tensor cput = at::ones({5,20,256,256,256}, at::kComplexFloat);
    print_memory_usage();

    for (int i = 0; i < 5; ++i) {

        print_memory_usage(std::to_string(i) + " ");
        at::Tensor output = cput.index({i, at::indexing::Ellipsis});

        for (int j = 0; j < 20; ++j) {

            print_memory_usage(std::to_string(i) + " " + std::to_string(j) + " ");

            auto output_slice = output.index({j, at::indexing::Ellipsis});
            auto cuda_output = output_slice.to(at::Device("cuda:0"));

            cuda_output *= at::rand({});

            output_slice.copy_(cuda_output);
        }

    }

    return cput;
}

void print_memory_usage(const std::string& prepend) {
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

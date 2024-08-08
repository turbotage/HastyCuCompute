//#include <interface.hpp>
#include <interface.hpp>

void test() {
    at::Tensor cput = at::ones({44, 100000000}, at::kComplexFloat);

    for (int i = 0; i < 44; ++i) {
        auto cpu_slice = cput.index({i, at::indexing::Ellipsis});
        auto cuda_slice = cpu_slice.to(at::Device("cuda:0"));
        cuda_slice *= 2;
        auto cpu_slice_copy = std::move(cuda_slice.to(at::Device("cpu")));
        at::Tensor& cpu_slice_ref = cpu_slice_copy;

        cpu_slice.copy_(cpu_slice_ref);
    }
}

int main() {

    //ffi::run_main();
    c10::InferenceMode im_guard{};

    //auto output = ffi::leak_test();

    auto output2 = ffi::test_simple_invert();

    return 0;

}
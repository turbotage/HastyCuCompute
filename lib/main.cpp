
#include "pch.hpp"

import util;
import nufft;
import trace;

int main() {

    using namespace hasty;

    int64_t ntransf = 32;
    int64_t nx = 128;
    int64_t ny = 128;
    int64_t nz = 128;
    int64_t nupts = 100000;

    std::cout << "type2\n";
    //type_2_tests(ntransf, nx, ny, nz, nupts);
    std::cout << "type1\n";
    //type_1_tests(ntransf, nx, ny, nz, nupts);

    trace::trace_tensor<cuda_f32, 3> a("a");
    trace::trace_tensor<cuda_f32, 3> b("b");

    auto filter = trace::trace_function("fft_filter", a, b);
    std::cout << b.name() << "\n";
    std::cout << filter.str() << "\n\n\n";

    filter.add_line(b.operator=<cuda_f32,3>(
        trace::fftn(a, span({128,128,128}), nullspan())));

    filter.add_line(b.operator=<cuda_f32,3>(
        trace::ifftn(b, span({128,128,128}), nullspan())));


    
    auto kernel = make_tensor<cuda_c64, 3>(
        span({2*nx,2*ny,2*nz}), 
        "cuda:0", tensor_make_opts::ZEROS);

    
    auto coords = std::array<tensor<cuda_f32, 1>, 3>{
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0"),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0"),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0")
    };
    
    auto nudata = make_tensor<cuda_c64, 2>(span<2>({1, nupts}), "cuda:0");

    toeplitz_kernel(coords, kernel, nudata);
    
    auto multiply_data = make_tensor<cuda_c64, 4>(span<4>({ntransf, nx, ny, nz}), "cuda:0",
        tensor_make_opts::RAND_UNIFORM);
    
    toeplitz_multiply(multiply_data, kernel);

    //std::cout << tt.name() << "\n";

    return 0;
}
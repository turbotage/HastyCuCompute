
#include <iostream>
#include "pch.hpp"

import util;
import nufft;
import trace;

import hdf5;

int main() {

    using namespace hasty;

    int64_t ntransf = 4;
    int64_t nx = 32;
    int64_t ny = 32;
    int64_t nz = 32;
    int64_t nupts = 10000;

    auto randvec = at::rand({4,15,72}, at::kCPU);

    export_tensor(randvec, "/home/turbotage/Documents/SmallTests/randvec.h5", "randvec");

    //std::cout << "type2\n";
    //type_2_tests(ntransf, nx, ny, nz, nupts);
    //std::cout << "type1\n";
    //type_1_tests(ntransf, nx, ny, nz, nupts);
    /*
    trace::trace_tensor<cuda_f32, 3> a("a");
    trace::trace_tensor<cuda_f32, 3> b("b");

    auto filter = trace::trace_function("fft_filter", a, b);
    //std::cout << b.name() << "\n";
    //std::cout << filter.str() << "\n\n\n";

    filter.add_line(b.operator=<cuda_f32,3>(
        trace::fftn(a, span({128,128,128}), nullspan())));

    filter.add_line(b.operator=<cuda_f32,3>(
        trace::ifftn(b, span({128,128,128}), nullspan())));
    */

    /*
    auto kernel = make_tensor<cuda_c64, 3>(
        span({2*nx,2*ny,2*nz}), 
        "cuda:0", tensor_make_opts::ZEROS);

    
    auto coords = std::array<tensor<cuda_f32, 1>, 3>{
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0"),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0"),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0")
    };
    
    {
        auto nudata = make_tensor<cuda_c64, 2>(span<2>({1, nupts}), "cuda:0");
        nudata.fill_(1.0f);

        toeplitz_kernel(coords, kernel, nudata);
    }
    
    auto multiply_data = make_tensor<cuda_c64, 4>(span<4>({ntransf, nx, ny, nz}), "cuda:0",
        tensor_make_opts::ONES);
    
    auto multiply_data2 = multiply_data.clone();


    toeplitz_multiply(multiply_data, kernel);

    auto nudata = make_tensor<cuda_c64, 2>(span<2>({ntransf, nupts}), "cuda:0");

    float subnelem = (1e-3 / std::sqrt((double)nx*ny*nz));
    {
        auto plan = nufft_plan<cuda_f32, 3, nufft_type::TYPE_2>::make(
            nufft_opts<cuda_f32, 3>{
                .nmodes = {nx, ny, nz},
                .ntransf = static_cast<i32>(ntransf),
                .tol = 1e-5,
                .sign = nufft_sign::DEFAULT_TYPE_2,
                .upsamp = nufft_upsamp_cuda::DEFAULT,
                .method = nufft_method_cuda::DEFAULT
            }
        );
        plan->setpts(coords);
        plan->execute(multiply_data2, nudata);
        nudata *= subnelem;
    }
    {
        auto plan = nufft_plan<cuda_f32, 3, nufft_type::TYPE_1>::make(
            nufft_opts<cuda_f32, 3>{
                .nmodes = {nx, ny, nz},
                .ntransf = static_cast<i32>(ntransf),
                .tol = 1e-5,
                .sign = nufft_sign::DEFAULT_TYPE_1,
                .upsamp = nufft_upsamp_cuda::DEFAULT,
                .method = nufft_method_cuda::DEFAULT
            }
        );
        plan->setpts(coords);
        plan->execute(nudata, multiply_data2);
        multiply_data2 *= subnelem;
    }
    
    auto diff = multiply_data - multiply_data2;

    std::cout << diff.norm() / multiply_data.norm() << "\n";
    std::cout << diff.abs().max() << "\n";
    */

    return 0;
}
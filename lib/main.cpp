
#include <iostream>
#include "pch.hpp"


import util;
import nufft;
import trace;

import hdf5;

void trace_test() {
    using namespace hasty;
    using namespace hasty::trace;

    tensor_prototype<cuda_c64, 3> input("input");
    tensor_prototype<cuda_c64, 4> coilmap("coilmap");
    tensor_prototype<cuda_c64, 3> kernel("kernel");

    tensor_prototype<cuda_c64, 3> output("output");

    auto toeplitz = trace_function_factory<decltype(output)>::make("toeplitz", input, coilmap, kernel);

    toeplitz.add_lines(
std::format(R"ts(
    #shp = input.shape
    spatial_shp = input.shape #shp[1:]
    expanded_shp = [2*s for s in spatial_shp]
    transform_dims = [i+1 for i in range(len(spatial_shp))]

    ncoil = coilmap.shape[0]
    nrun = ncoil // {0}
    
    out = torch.zeros_like(input)
    for run in range(nrun):
        bst = run*{0}
        cmap = coilmap[bst:(bst+{0})]
        c = cmap * input
        c = torch.fft_fftn(c, expanded_shp, transform_dims)
        c *= kernel
        c = torch.fft_ifftn(c, None, transform_dims)

        for dim in range(len(spatial_shp)):
            c = torch.slice(c, dim+1, spatial_shp[dim]-1, -1)

        c *= cmap.conj()
        out += torch.sum(c, 0)

    out *= (1 / torch.prod(torch.tensor(spatial_shp)))
    
    return out

)ts", 2));

    std::cout << toeplitz.str() << "\n";

    toeplitz.compile();

    tensor<cuda_c64, 3> input_data = make_tensor<cuda_c64, 3>(span({256, 256, 256}), 
                                        "cuda:0", tensor_make_opts::RAND_UNIFORM);

    tensor<cuda_c64, 4> coilmap_data = make_tensor<cuda_c64, 4>(span({8, 256, 256, 256}), "cuda:0", tensor_make_opts::RAND_UNIFORM);

    tensor<cuda_c64, 3> kernel_data = make_tensor<cuda_c64, 3>(span({512, 512, 512}), "cuda:0", tensor_make_opts::RAND_UNIFORM);


    auto start = std::chrono::high_resolution_clock::now();
    std::tuple<tensor<cuda_c64, 3>> output_data = toeplitz.run(input_data, coilmap_data, kernel_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time Toep Kernel First Run: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    int runs = 500;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        output_data = toeplitz.run(input_data, coilmap_data, kernel_data);
    }
    torch::cuda::synchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::format("Time Toep Kernel {} runs: ", runs) << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";


}

void toeplitz_test() {
    using namespace hasty;

    int64_t ntransf = 4;
    int64_t nx = 256;
    int64_t ny = 256;
    int64_t nz = 256;
    int64_t nupts = 400000;

    auto kernel = make_tensor<cuda_c64, 3>(
        span({2*nx,2*ny,2*nz}), 
        "cuda:0", tensor_make_opts::ZEROS);

    auto coords = std::array<tensor<cuda_f32, 1>, 3>{
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0", tensor_make_opts::RAND_UNIFORM),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0", tensor_make_opts::RAND_UNIFORM),
        make_tensor<cuda_f32, 1>({nupts}, "cuda:0", tensor_make_opts::RAND_UNIFORM)
    };
    
    coords[0].mul_(2*3.141592f).add_(-3.141592f);
    coords[1].mul_(2*3.141592f).add_(-3.141592f);
    coords[2].mul_(2*3.141592f).add_(-3.141592f);

    {
        auto nudata = make_tensor<cuda_c64, 2>(span<2>({1, nupts}), "cuda:0");
        nudata.fill_(1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        toeplitz_kernel(coords, kernel, nudata);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time Toep Multiply: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    }
    
    auto multiply_data = make_tensor<cuda_c64, 4>(span<4>({ntransf, nx, ny, nz}), "cuda:0",
        tensor_make_opts::ONES);
    
    int runs = 50;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i) {
        toeplitz_multiply(multiply_data, kernel);
        torch::cuda::synchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time Toep Multiply: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    auto nudata = make_tensor<cuda_c64, 2>(span<2>({ntransf, nupts}), "cuda:0");
    float subnelem = (1.0 / std::sqrt((double)nx*ny*nz));

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i) {
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
            plan->execute(multiply_data, nudata);
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
            plan->execute(nudata, multiply_data);
            multiply_data *= subnelem;
        }
    }

    end = std::chrono::high_resolution_clock::now();

    std::cout << "Time NUFFT: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    //export_tensor(multiply_data.get_tensor().cpu(), "/home/turbotage/Documents/SmallTests/multdata.h5", "data");
    //export_tensor(multiply_data2.get_tensor().cpu(), "/home/turbotage/Documents/SmallTests/multdata2.h5", "data");

    /*
    auto diff = multiply_data - multiply_data2;
    std::cout << diff.norm() / multiply_data.norm() << "\n";
    std::cout << diff.abs().max() << "\n";
    */
}

void compile_test() {

    /*
    auto module = torch::jit::compile(R"JIT(
        def run(a: Tensor, b: Tensor) -> Tensor:
            shp = torch.size(a)
            _0 = [torch.mul(shp[1], 2), torch.mul(shp[2], 2), torch.mul(shp[3], 2)]
            c = torch.fft_fftn(a, _0, [1, 2, 3])
            c0 = torch.mul_(c, b)
            _1 = torch.slice(torch.fft_ifftn(c0, None, [1, 2, 3]))
            _2 = torch.slice(_1, 1, torch.sub(shp[1], 1), -1)
            _3 = torch.slice(_2, 2, torch.sub(shp[2], 1), -1)
            _4 = torch.slice(_3, 3, torch.sub(shp[3], 1), -1)
            _5 = torch.mul(torch.mul(shp[1], shp[2]), shp[3])
            return torch.div(_4, _5)
    )JIT");
    */


    auto module = torch::jit::compile(R"JIT(
        def run(a: Tensor, b: Tensor) -> Tensor:
            shp = a.shape
            c = torch.fft_fftn(a, [shp[1]*2, shp[2]*2, shp[3]*2], [1,2,3])
            c *= b
            c = torch.fft_ifftn(c, None, [1,2,3])
            return c[:, shp[1]-1:-1, shp[2]-1:-1, shp[3]-1:-1] / (shp[1]*shp[2]*shp[3])
    )JIT");

    auto a = torch::rand({2, 256, 256, 256}, at::TensorOptions("cuda:0").dtype(at::kComplexFloat));
    auto kernel = torch::rand({512,512,512}, at::TensorOptions("cuda:0").dtype(at::kComplexFloat));

    auto start = std::chrono::high_resolution_clock::now();
    at::Tensor c1 = module->run_method("run", a, kernel).toTensor();
    c1 = c1.to(c10::Device("cpu"));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        c1 = module->run_method("run", a, kernel).toTensor();
    }
    c1 = c1.to(c10::Device("cpu"));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    auto shp = a.sizes();

    start = std::chrono::high_resolution_clock::now();
    at::Tensor c2 = torch::fft::fftn(a, {shp[1]*2, shp[2]*2, shp[3]*2}, {1, 2, 3});
    c2 *= kernel;
    c2 = torch::fft::ifftn(c2, {shp[1]*2, shp[2]*2, shp[3]*2}, {1, 2, 3});
    c2 = c2.slice(1, shp[1]-1, -1).slice(2, shp[2]-1, -1).slice(3, shp[3]-1, -1);
    c2 = c2.to(c10::Device("cpu"));
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        c2 = torch::fft::fftn(a, {shp[1]*2, shp[2]*2, shp[3]*2}, {1, 2, 3});
        c2 *= kernel;
        c2 = torch::fft::ifftn(c2, {shp[1]*2, shp[2]*2, shp[3]*2}, {1, 2, 3});
        c2 = c2.slice(1, shp[1]-1, -1).slice(2, shp[2]-1, -1).slice(3, shp[3]-1, -1);
        c2 = c2.to(c10::Device("cpu"));
    }
    end = std::chrono::high_resolution_clock::now();

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

int main() {

    trace_test();

    //compile_test();

    //toeplitz_test();

    return 0;
}
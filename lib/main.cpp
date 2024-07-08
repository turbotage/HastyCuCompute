
#include <iostream>
#include "pch.hpp"
#include <fstream>

import util;
import nufft;
import trace;
import tensor;

import min;

import hdf5;

void trace_test() {
    using namespace hasty;
    using namespace hasty::trace;

    tensor_prototype<cuda_t,c64_t, 3> input("input");
    tensor_prototype<cuda_t,c64_t, 4> coilmap("coilmap");
    tensor_prototype<cuda_t,c64_t, 3> kernel("kernel");

    tensor_prototype<cuda_t,c64_t, 3> output("output");

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

    tensor<cuda_t,c64_t,3> input_data = make_tensor<cuda_t,c64_t,3>(span({256, 256, 256}), 
                                        device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM);

    tensor<cuda_t,c64_t,4> coilmap_data = make_tensor<cuda_t,c64_t,4>(span({8, 256, 256, 256}), device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM);

    tensor<cuda_t,c64_t,3> kernel_data = make_tensor<cuda_t,c64_t,3>(span({512, 512, 512}), device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM);


    auto start = std::chrono::high_resolution_clock::now();
    std::tuple<tensor<cuda_t,c64_t,3>> output_data = toeplitz.run(input_data, coilmap_data, kernel_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time Toep Kernel First Run: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    int runs = 50;
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

    auto kernel = make_tensor<cuda_t,c64_t,3>(
        span({2*nx,2*ny,2*nz}), 
        device_idx::CUDA0, tensor_make_opts::ZEROS);

    auto coords = std::array<tensor<cuda_t,f32_t,1>, 3>{
        make_tensor<cuda_t,f32_t,1>({nupts}, device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM),
        make_tensor<cuda_t,f32_t,1>({nupts}, device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM),
        make_tensor<cuda_t,f32_t,1>({nupts}, device_idx::CUDA0, tensor_make_opts::RAND_UNIFORM)
    };
    
    coords[0].mul_(2*3.141592f).add_(-3.141592f);
    coords[1].mul_(2*3.141592f).add_(-3.141592f);
    coords[2].mul_(2*3.141592f).add_(-3.141592f);

    {
        auto nudata = make_tensor<cuda_t,c64_t,2>(span<2>({1, nupts}), device_idx::CUDA0);
        nudata.fill_(1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        toeplitz_kernel(coords, kernel, nudata);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time Toep Multiply: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    }
    
    auto multiply_data = make_tensor<cuda_t,c64_t,4>(span<4>({ntransf, nx, ny, nz}), device_idx::CUDA0,
        tensor_make_opts::ONES);
    
    int runs = 50;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i) {
        toeplitz_multiply(multiply_data, kernel);
        torch::cuda::synchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time Toep Multiply: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    auto nudata = make_tensor<cuda_t,c64_t, 2>(span<2>({ntransf, nupts}), device_idx::CUDA0);
    float subnelem = (1.0 / std::sqrt((double)nx*ny*nz));

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i) {
        {
            auto plan = nufft_plan<cuda_t,f32_t,3,nufft_type::TYPE_2>::make(
                nufft_opts<cuda_t,f32_t,3>{
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
            auto plan = nufft_plan<cuda_t,f32_t,3,nufft_type::TYPE_1>::make(
                nufft_opts<cuda_t,f32_t,3>{
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

void iotest() {

    std::vector<std::complex<float>> data(640*640*640);

    std::ofstream outfile("/home/turbotage/Documents/io.mypt", std::ios::out | std::ios::binary);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    outfile.write(reinterpret_cast<char*>(data.data()), data.size()*sizeof(std::complex<float>));
    outfile.close();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: " << millis << "ms\n";
    std::cout << "Write Speed: " << (640*640*640*sizeof(std::complex<float>) / (1000*1000)) / ((double)millis / 1000.0) << "MB/s\n";

}

#include <matplot/matplot.h>

void simple_invert() {

    using namespace hasty;

    cache_dir = "/home/turbotage/Documents/hasty_cache/";

    std::vector<std::regex> matchers = {
        std::regex("^/Kdata/KData_.*"),
        std::regex("^/Kdata/KX_.*"),
        std::regex("^/Kdata/KY_.*"),
        std::regex("^/Kdata/KZ_.*"),
        std::regex("^/Kdata/KW_.*")
    };
    auto tset = import_tensors(
        "/home/turbotage/Documents/4DRecon/other_data/MRI_Raw.h5", matchers);

    std::array<std::array<cache_tensor<f32_t,1>,3>,5> coords;
    std::array<cache_tensor<f32_t,1>,5> weights;

    std::array<cache_tensor<c64_t,2>,5> kdata;

    auto shape_getter = []<size_t R>(const at::Tensor& ten) -> std::array<i64,R> 
    {
        if (ten.ndimension() != R) {
            throw std::runtime_error("Invalid number of dimensions");
        }
        std::array<i64,R> shape;
        for_sequence<R>([&](auto i) {
            shape[i] = ten.size(i);
        });
        return shape;
    };

    int ncoil = 0;
    for (int e = 0; e < 5; ++e) {

        at::Tensor temp = std::get<at::Tensor>(tset["/Kdata/KX_E" + std::to_string(e)]).flatten();
        coords[e][0] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KX_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KX_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KY_E" + std::to_string(e)]).flatten();
        coords[e][1] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KY_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KY_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KZ_E" + std::to_string(e)]).flatten();
        coords[e][2] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KZ_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KZ_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KW_E" + std::to_string(e)]).flatten();
        weights[e] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KW_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KW_E" + std::to_string(e));

        std::vector<at::Tensor> kdata_tensors;
        kdata_tensors.reserve(48);
        for (int c = 0; true; ++c) {
            auto key = "/Kdata/KData_E" + std::to_string(e) + "_C" + std::to_string(c);
            if (tset.find(key) == tset.end()) {
                break;
            }
            kdata_tensors.push_back(std::get<at::Tensor>(tset[key]).flatten());
        
            tset.erase(key);
        }

        auto kdata_tensor = at::stack(kdata_tensors, 0);
        ncoil = kdata_tensor.size(0);
        kdata[0] = cache_tensor<c64_t,2>(
            tensor_factory<cpu_t,c64_t,2>::make(shape_getter.template operator()<2>(kdata_tensor), kdata_tensor),
            std::hash<std::string>{}("KData_E" + std::to_string(e))
        );

    }


    for (int e = 0; e < 5; ++e) {

        std::array<tensor<cuda_t,f32_t,1>,3> coords_gpu;
        for (int i = 0; i < 3; ++i) {
            coords_gpu[i] = coords[e][i].template get<cuda_t>(device_idx::CUDA0);
        }

        //auto weights_gpu = weights[e].template get<cuda_t>(device_idx::CUDA0);
        //auto kdata_gpu = kdata[e].template get<cuda_t>(device_idx::CUDA0);

        auto input = weights[e].template get<cpu_t>().unsqueeze(0) * kdata[e].template get<cpu_t>();

        tensor<cpu_t,c64_t,4> images = make_tensor<cpu_t,c64_t,4>(
            span<4>({ncoil, 320, 320, 320}), device_idx::CUDA0, tensor_make_opts::EMPTY);

        auto output = nufft_backward_cpu_over_cuda(span<3>({320, 320, 320}), input, coords_gpu);

        

    }




    std::cout << "Hello" << std::endl; 
}


int main() {

    //concept_test();

    //iotest();

    simple_invert();

    //trace_test();

    //compile_test();

    //toeplitz_test();

    return 0;
}
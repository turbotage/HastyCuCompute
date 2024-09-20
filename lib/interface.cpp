#include "interface.hpp"

import util;
import nufft;
import trace;
import tensor;

import min;

import hdf5;

std::vector<at::Tensor> ffi::test_simple_invert() {
    c10::InferenceMode im_guard{};
    torch::NoGradGuard no_grad_guard;

    using namespace hasty;

    cache_dir = "/home/turbotage/Documents/hasty_cache/";

    /*
    std::vector<std::regex> matchers = {
        std::regex("^/Kdata/KData_E[01].*"),
        std::regex("^/Kdata/KX_E[01].*"),
        std::regex("^/Kdata/KY_E[01].*"),
        std::regex("^/Kdata/KZ_E[01].*"),
        std::regex("^/Kdata/KW_E[01].*")
    };
    */
    
    std::vector<std::regex> matchers = {
        std::regex("^/Kdata/KData_E.*"),
        std::regex("^/Kdata/KX_E.*"),
        std::regex("^/Kdata/KY_E.*"),
        std::regex("^/Kdata/KZ_E.*"),
        std::regex("^/Kdata/KW_E.*")
    };

    std::cout << "Importing tensors" << std::endl;
    auto tset = import_tensors(
        "/home/turbotage/Documents/4DRecon/other_data/MRI_Raw.h5", matchers);

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

    std::vector<at::Tensor> output_tensors;
    output_tensors.reserve(5);
    for (int e = 0; e < 5; ++e) {

        std::array<cache_tensor<f32_t,1>,3> coords;
        cache_tensor<f32_t,1> weights;

        cache_tensor<c64_t,2> kdata;

        std::cout << "Starting encode " << e << std::endl;

        at::Tensor temp = std::get<at::Tensor>(tset["/Kdata/KX_E" + std::to_string(e)]).flatten();
        temp *= (3.141592 / 160.0);
        coords[0] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KX_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KX_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KY_E" + std::to_string(e)]).flatten();
        temp *= (3.141592 / 160.0);
        coords[1] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KY_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KY_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KZ_E" + std::to_string(e)]).flatten();
        temp *= (3.141592 / 160.0);
        coords[2] = cache_tensor<f32_t,1>(
            tensor_factory<cpu_t,f32_t,1>::make(shape_getter.template operator()<1>(temp), temp),
            std::hash<std::string>{}("KZ_E" + std::to_string(e))
        );
        tset.erase("/Kdata/KZ_E" + std::to_string(e));

        temp = std::get<at::Tensor>(tset["/Kdata/KW_E" + std::to_string(e)]).flatten();
        weights = cache_tensor<f32_t,1>(
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
            temp = std::get<at::Tensor>(tset[key]).flatten();

            kdata_tensors.push_back(temp);
        
            tset.erase(key);
        }
        auto kdata_tensor = at::stack(kdata_tensors, 0);
        kdata_tensors.clear();

        kdata = cache_tensor<c64_t,2>(
            tensor_factory<cpu_t,c64_t,2>::make(shape_getter.template operator()<2>(kdata_tensor), std::move(kdata_tensor)),
            std::hash<std::string>{}("KData_E" + std::to_string(e))
        );
        //kdata_tensor = at::empty({0}, at::kFloat);

        std::cout << "Starting nuffts" << std::endl;
        
        
        /*
        std::array<tensor<cuda_t,f64_t,1>,3> coords_gpu;
        for (int i = 0; i < 3; ++i) {
            coords_gpu[i] = move(coords[i].template get<cuda_t>(device_idx::CUDA0).template to<f64_t>());
        }
        */
        
        
        std::array<tensor<cuda_t,f32_t,1>,3> coords_gpu;
        for (int i = 0; i < 3; ++i) {
            coords_gpu[i] = move(coords[i].template get<cuda_t>(device_idx::CUDA0));
        }

        auto input = weights.template get<cpu_t>().unsqueeze(0) * kdata.template get<cpu_t>();

        coords[0] = cache_tensor<f32_t,1>();
        coords[1] = cache_tensor<f32_t,1>();
        coords[2] = cache_tensor<f32_t,1>();
        weights = cache_tensor<f32_t,1>();
        kdata = cache_tensor<c64_t,2>();

        auto start = std::chrono::high_resolution_clock::now();

        auto output = nufft_backward_cpu_over_cuda(span<3>({320, 320, 320}), input, coords_gpu);        

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Encode: " << e << " Time: " << 
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        output_tensors.push_back(output.get_tensor());
    }

    return output_tensors;
}


std::vector<at::Tensor> ffi::test_offresonance_operator() {

    using namespace hasty;

    cache_dir = "/home/turbotage/Documents/hasty_cache/";

    int xres = 64;
    int yres = 64;
    int zres = 64;
    int ncoils = 8;
    int offresonance_n = 5;

    cache_tensor<c64_t, 3> diagonal = tensor_factory<cpu_t,c64_t,3>::make({xres, yres, zres}, tensor_make_opts::RAND_UNIFORM);
    cache_tensor<c64_t, 4> smaps = tensor_factory<cpu_t,c64_t,4>::make({ncoils, xres, yres, zres}, tensor_make_opts::RAND_UNIFORM);
    std::vector<cache_tensor<c64_t, 3>> kernels;
    std::vector<cache_tensor<c64_t, 3>> ratemap_diagonals;
    for (int i = 0; i < 5; ++i) {
        kernels.push_back(tensor_factory<cpu_t,c64_t,3>::make({2*xres, 2*yres, 2*zres}, tensor_make_opts::RAND_UNIFORM));
        ratemap_diagonals.push_back(tensor_factory<cpu_t,c64_t,3>::make({xres, yres, zres}, tensor_make_opts::RAND_UNIFORM));
    }

    sense_normal_image_offresonance_diagonal<cuda_t, c64_t, 3> sense(smaps, diagonal, kernels, ratemap_diagonals);

}


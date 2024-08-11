#include "interface.hpp"
#include "py_interface.hpp"

import util;
import nufft;
import trace;
import tensor;

import min;

import hdf5;

namespace py = pybind11;

#ifdef _WIN32
#define MODULE_NAME HastyCuCompute
#else
#define MODULE_NAME libHastyCuCompute
#endif

namespace {

    at::Tensor from_buffer(const py::buffer& buf) {
        c10::InferenceMode im_guard{};

        py::buffer_info info = buf.request();

        std::cout << "Format: " << info.format << std::endl;
        // Determine the data type
        at::ScalarType dtype;
        if (info.format == py::format_descriptor<float>::format()) {
            dtype = at::kFloat;
        } else if (info.format == py::format_descriptor<double>::format()) {
            dtype = at::kDouble;
        } else if (info.format == py::format_descriptor<int>::format()) {
            dtype = at::kInt;
        } else if (info.format == py::format_descriptor<int64_t>::format()) {
            dtype = at::kLong;
        } else if (info.format == "Zf") { // Complex float
            dtype = at::kComplexFloat;
        } else if (info.format == "Zd") { // Complex double
            dtype = at::kComplexDouble;
        } else {
            throw std::runtime_error("Unsupported data type");
        }


        /*
        std::vector<int64_t> strides_vec(info.strides.size());
        for (int i = 0; i < info.strides.size(); ++i) {
            strides_vec[i] = info.strides[i] / info.itemsize;
        }
        */
        at::IntArrayRef shape(info.shape);
        for (int i = 0; i < info.strides.size(); ++i) {
            info.strides[i] /= info.itemsize;
        }
        at::IntArrayRef strides(info.strides);


        // Create the tensor from the buffer
        return at::from_blob(info.ptr, shape, strides, dtype);
    }

    py::array from_tensor(const at::Tensor& ten) {
        // Determine the format string
        std::string format;
        if (ten.scalar_type() == at::kFloat) {
            format = py::format_descriptor<float>::format();
        } else if (ten.scalar_type() == at::kDouble) {
            format = py::format_descriptor<double>::format();
        } else if (ten.scalar_type() == at::kInt) {
            format = py::format_descriptor<int>::format();
        } else if (ten.scalar_type() == at::kLong) {
            format = py::format_descriptor<int64_t>::format();
        } else if (ten.scalar_type() == at::kComplexFloat) {
            format = "Zf"; // Complex float
        } else if (ten.scalar_type() == at::kComplexDouble) {
            format = "Zd"; // Complex double
        } else if (ten.scalar_type() == at::kBool) {
            format = py::format_descriptor<bool>::format();
        } else {
            throw std::runtime_error("Unsupported data type");
        }

        // Get the shape and strides
        std::vector<ssize_t> shape(ten.sizes().begin(), ten.sizes().end());
        std::vector<ssize_t> strides(ten.strides().begin(), ten.strides().end());

        // Convert strides from element counts to byte counts
        ssize_t itemsize = ten.element_size();
        for (auto& stride : strides) {
            stride *= itemsize;
        }

        // Create the buffer_info
        return py::array(py::buffer_info(
            ten.data_ptr(),                // Pointer to buffer
            itemsize,                      // Size of one scalar
            format,                        // Python struct-style format descriptor
            ten.dim(),                     // Number of dimensions
            shape,                         // Buffer dimensions
            strides                        // Strides (in bytes) for each index
        ));
    }

}

void print_memory_usage(const std::string& prepend = "") {
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

at::Tensor ffi::leak_test1() {
    c10::InferenceMode im_guard{};

    at::Tensor cput = at::ones({3,10,320,320,320}, at::kComplexFloat);
    print_memory_usage();

    for (int i = 0; i < cput.size(0); ++i) {

        print_memory_usage(std::to_string(i) + " ");
        at::Tensor output = cput.index({i, at::indexing::Ellipsis});

        for (int j = 0; j < cput.size(1); ++j) {

            print_memory_usage(std::to_string(i) + " " + std::to_string(j) + " ");

            auto output_slice = output.index({j, at::indexing::Ellipsis});
            auto cuda_output = output_slice.to(at::Device("cuda:0"));

            cuda_output *= at::rand({});

            auto cpu_output = cuda_output.to(at::Device("cpu"));

            //output_slice.copy_(cpu_output);
            at::copy_out(output_slice, output_slice, cpu_output);
        }

    }

    return cput;
}

at::Tensor ffi::leak_test2() {

    c10::InferenceMode im_guard{};
    torch::NoGradGuard no_grad_guard;

    using namespace hasty;

    //print_memory_usage();

    cache_dir = "/home/turbotage/Documents/hasty_cache/";

    auto cput = make_empty_tensor<cpu_t, c64_t, 5>(span<5>({1,10,320,320,320}));

    for (int i = 0; i < cput.template shape<0>(); ++i) {
        debug::print_memory_usage("Outer Loop Start: ");

        tensor<cpu_t,c64_t,4> output = cput[i, Ellipsis{}];

        for (int i = 0; i < output.template shape<0>(); ++i) {
            debug::print_memory_usage("Loop Start: ");
            auto output_slice = output[i, Ellipsis{}].unsqueeze(0);
            debug::print_memory_usage();
            auto cuda_output = output_slice.template to<cuda_t>(device_idx::CUDA0);
            debug::print_memory_usage();
            //cuda_output *= float(rand() % 100) / 300.0;
            //print_memory_usage();
            auto back_on_cpu = cuda_output.template to<cpu_t>(device_idx::CPU);
            debug::print_memory_usage();
            output_slice = back_on_cpu;
            debug::print_memory_usage("Loop End: ");
            /*
            */
        }

       debug::print_memory_usage("Outer Loop End: ");

    }

    return cput.get_tensor(); 
}

at::Tensor ffi::test_simple_invert() {
    c10::InferenceMode im_guard{};
    torch::NoGradGuard no_grad_guard;

    using namespace hasty;

    print_memory_usage();

    cache_dir = "/home/turbotage/Documents/hasty_cache/";

    std::vector<std::regex> matchers = {
        std::regex("^/Kdata/KData_.*"),
        std::regex("^/Kdata/KData_.*"),
        std::regex("^/Kdata/KX_.*"),
        std::regex("^/Kdata/KY_.*"),
        std::regex("^/Kdata/KZ_.*"),
        std::regex("^/Kdata/KW_.*")
    };
    
    std::cout << "Importing tensors" << std::endl;
    auto tset = import_tensors(
        "/home/turbotage/Documents/4DRecon/other_data/MRI_Raw.h5", matchers);

    print_memory_usage();

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
        
        print_memory_usage();

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
             tensor_factory<cpu_t,c64_t,2>::make(shape_getter.template operator()<2>(kdata_tensor), kdata_tensor),
            std::hash<std::string>{}("KData_E" + std::to_string(e))
        );
        kdata_tensor = at::empty({0}, at::kFloat);

        std::cout << "Starting nuffts" << std::endl;

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

        auto output = nufft_backward_cpu_over_cuda(span<3>({320, 320, 320}), input, coords_gpu);        

        output_tensors.push_back(output.get_tensor());

        print_memory_usage();
    }

    auto output = at::stack(output_tensors, 0);

    return output;
}
 
py::array pyffi::test_simple_invert() {
    return from_tensor(ffi::test_simple_invert());
}



PYBIND11_MODULE(MODULE_NAME, m) {
    
    m.doc() = "HastyCuCompute module";

    m.def("test_simple_invert", &pyffi::test_simple_invert, "Simple invert");

}
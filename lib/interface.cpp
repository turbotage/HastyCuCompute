#include "interface.hpp"
#include "pch.hpp"

import util;
import nufft;
import trace;
import tensor;

import min;

import hdf5;

void ffi::run_main() {
    std::cout << "Running main" << std::endl;
}

namespace py = pybind11;

#ifdef _WIN32
#define MODULE_NAME HastyCuCompute
#else
#define MODULE_NAME libHastyCuCompute
#endif

namespace {

    at::Tensor from_buffer(const py::buffer& buf) {
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


py::array ffi::add(py::buffer a, py::buffer b) {
    auto aten = from_buffer(a);
    auto bten = from_buffer(b);
    return from_tensor(aten + bten);
}


py::array ffi::simple_invert() {

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

    std::vector<at::Tensor> output_tensors;
    output_tensors.reserve(5);
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

        output_tensors.push_back(output.get_tensor());
    }

    auto output = at::stack(output_tensors, 0);

    return from_tensor(output);

}



PYBIND11_MODULE(MODULE_NAME, m) {
    
    m.doc() = "HastyCuCompute module";

    m.def("add", &ffi::add, "Add two tensors");

    m.def("simple_invert", &ffi::simple_invert, "Simple invert");

}

/*
    py::class_<ffi::Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](ffi::Tensor& m) -> py::buffer_info {

            auto ten = m.get();

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
                format = "Zf"; //py::format_descriptor<std::complex<float>>::format();
            } else if (ten.scalar_type() == at::kComplexDouble) {
                format = "Zd"; //py::format_descriptor<std::complex<double>>::format();
            } else if (ten.scalar_type() == at::kBool) {
                format = py::format_descriptor<bool>::format();
            } else {
                throw std::runtime_error("Unsupported format");
            }
            
            // Get the number of dimensions
            py::ssize_t ndim = ten.ndimension();
            py::ssize_t itemsize = ten.element_size();

            // Get the shape and strides
            std::vector<py::ssize_t> shape(ndim);
            std::vector<py::ssize_t> strides(ndim);
            for (int i = 0; i < ndim; ++i) {
                shape[i] = ten.size(i);
                strides[i] = ten.stride(i) * itemsize;
            }

            return py::buffer_info(
                ten.data_ptr(),
                itemsize,
                format.c_str(),
                ndim,
                shape,
                strides
            );
        });
    */

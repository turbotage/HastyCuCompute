#include "pch.hpp"
#include "py_interface.hpp"

#include "interface.hpp"

import std;
import torch_base;

namespace py = pybind11;

#ifdef _WIN32
#define MODULE_NAME HastyCuCompute
#else
#define MODULE_NAME libHastyCuCompute
#endif

namespace {

    hat::Tensor from_buffer(const py::buffer& buf) {
        hat::InferenceMode im_guard{};

        py::buffer_info info = buf.request();

        std::cout << "Format: " << info.format << std::endl;
        // Determine the data type
        hat::ScalarType dtype;
        if (info.format == py::format_descriptor<float>::format()) {
            dtype = hat::kFloat;
        } else if (info.format == py::format_descriptor<double>::format()) {
            dtype = hat::kDouble;
        } else if (info.format == py::format_descriptor<int>::format()) {
            dtype = hat::kInt;
        } else if (info.format == py::format_descriptor<int64_t>::format()) {
            dtype = hat::kLong;
        } else if (info.format == "Zf") { // Complex float
            dtype = hat::kComplexFloat;
        } else if (info.format == "Zd") { // Complex double
            dtype = hat::kComplexDouble;
        } else {
            throw std::runtime_error("Unsupported data type");
        }


        /*
        std::vector<int64_t> strides_vec(info.strides.size());
        for (int i = 0; i < info.strides.size(); ++i) {
            strides_vec[i] = info.strides[i] / info.itemsize;
        }
        */
        hat::IntArrayRef shape(info.shape);
        for (int i = 0; i < info.strides.size(); ++i) {
            info.strides[i] /= info.itemsize;
        }
        hat::IntArrayRef strides(info.strides);


        // Create the tensor from the buffer
        return hat::from_blob(info.ptr, shape, strides, dtype);
    }

    py::array from_tensor(hat::Tensor ten) {
        // Determine the format string
        std::string format;
        if (ten.scalar_type() == hat::kFloat) {
            format = py::format_descriptor<float>::format();
        } else if (ten.scalar_type() == hat::kDouble) {
            format = py::format_descriptor<double>::format();
        } else if (ten.scalar_type() == hat::kInt) {
            format = py::format_descriptor<int>::format();
        } else if (ten.scalar_type() == hat::kLong) {
            format = py::format_descriptor<int64_t>::format();
        } else if (ten.scalar_type() == hat::kComplexFloat) {
            format = "Zf"; // Complex float
        } else if (ten.scalar_type() == hat::kComplexDouble) {
            format = "Zd"; // Complex double
        } else if (ten.scalar_type() == hat::kBool) {
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

        auto dataptr = ten.data_ptr();
        auto dims = ten.dim();

        auto capsule = py::capsule(new hat::Tensor(std::move(ten)), [](void* p) {
            delete static_cast<hat::Tensor*>(p);
        });

        // Create the buffer_info
        return py::array(py::buffer_info(
            dataptr,                        // Pointer to buffer
            itemsize,                       // Size of one scalar
            format,                         // Python struct-style format descriptor
            dims,                           // Number of dimensions
            shape,                          // Buffer dimensions
            strides                         // Strides (in bytes) for each index
        ), capsule);
    }

}

namespace pyffi {

    std::vector<py::array> test_simple_invert() {
        std::vector<py::array> output_arrays;
        output_arrays.reserve(5);
        ffi::test_simple_invert();
        return {};
    }

}



PYBIND11_MODULE(MODULE_NAME, m) {
    
    m.doc() = "HastyCuCompute module";

    m.def("test_simple_invert", &pyffi::test_simple_invert, "Simple invert");

}
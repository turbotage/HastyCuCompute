#include "interface.hpp"
#include "pch.hpp"



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

PYBIND11_MODULE(MODULE_NAME, m) {
    
    m.doc() = "HastyCuCompute module";

    m.def("add", &ffi::add, "Add two tensors");

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

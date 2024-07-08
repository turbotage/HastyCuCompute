#include "pch.hpp"
#include "interface.hpp"


void ffi::run_main() {
    std::cout << "Running main" << std::endl;
}


namespace py = pybind11;

PYBIND11_MODULE(HastyCuCompute, m) {
    
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



}

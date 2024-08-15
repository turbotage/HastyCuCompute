#include "interface.hpp"
#include "py_interface.hpp"

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

    py::array from_tensor(at::Tensor ten) {
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

        auto dataptr = ten.data_ptr();
        auto dims = ten.dim();

        auto capsule = py::capsule(new at::Tensor(std::move(ten)), [](void* p) {
            delete static_cast<at::Tensor*>(p);
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

std::vector<py::array> pyffi::test_simple_invert() {
    std::vector<py::array> output_arrays;
    output_arrays.reserve(5);
    auto tens = ffi::test_simple_invert();
    for (int i = 0; i < tens.size(); ++i) {
        output_arrays.push_back(std::move(from_tensor(std::move(tens[i]))));
    }
    return output_arrays;
}


PYBIND11_MODULE(MODULE_NAME, m) {
    
    m.doc() = "HastyCuCompute module";

    m.def("test_simple_invert", &pyffi::test_simple_invert, "Simple invert");

}
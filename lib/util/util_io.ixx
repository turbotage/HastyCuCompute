module;

#include "pch.hpp"

export module util:io;


namespace hasty {


    export void export_binary_tensor(at::Tensor tensor, const std::filesystem::path& path)
    {
        try {
            std::ofstream file(path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing");
            }

            file.write(reinterpret_cast<const char*>(tensor.data_ptr()), tensor.numel() * tensor.element_size());
            file.close();
        } catch (std::ifstream::failure& e) {
            throw std::runtime_error("Exception opening/reading file");
        } catch (...) {
            throw;
        }
    }

    export auto import_binary_tensor(const std::filesystem::path& path, at::IntArrayRef sizes, at::ScalarType dtype) 
        -> at::Tensor
    {
        at::Tensor tt = at::empty(sizes, at::TensorOptions().dtype(dtype));
        try {
            std::ifstream file(path, std::ios::binary | std::ios::in);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for reading");
            }

            file.read(reinterpret_cast<char*>(tt.data_ptr()), tt.numel() * tt.element_size());
            file.close();
        } catch (std::ifstream::failure& e) {
            throw std::runtime_error("Exception opening/reading file");
        } catch (...) {
            throw;
        }

        return tt;
    }
}
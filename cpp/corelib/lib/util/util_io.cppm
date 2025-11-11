module;

#include "pch.hpp"

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
    #include <dlfcn.h>
#endif
#include "configure_file_settings.hpp"

export module util:io;

import std;
import torch_base;

namespace hasty {

export std::filesystem::path get_library_path() {
#if defined(_WIN32)
    HMODULE hModule = nullptr;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&get_library_path,
            &hModule)) {
        throw std::runtime_error("Failed to get module handle");
    }
    char path[MAX_PATH];
    if (GetModuleFileNameA(hModule, path, MAX_PATH) == 0) {
        throw std::runtime_error("Failed to get module file name");
    }
    return std::filesystem::path(path).parent_path();
#elif defined(__linux__) || defined(__APPLE__)
    Dl_info dl_info;
    if (dladdr((void*)&get_library_path, &dl_info) == 0) {
        throw std::runtime_error("Failed to get library path");
    }   
    return std::filesystem::path(dl_info.dli_fname).parent_path();
#endif
    throw std::runtime_error("Unsupported platform for get_library_path");
}

export std::filesystem::path get_data_path() {
    return get_library_path() / DATA_RELATIVE_PATH;
}

export void export_binary_tensor(hat::Tensor tensor, const std::filesystem::path& path)
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

export auto import_binary_tensor(const std::filesystem::path& path, hat::IntArrayRef sizes, hat::ScalarType dtype) 
    -> hat::Tensor
{
    hat::Tensor tt = hat::empty(sizes, hat::TensorOptions().dtype(dtype));
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
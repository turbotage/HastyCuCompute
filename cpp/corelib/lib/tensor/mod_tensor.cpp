module;

#include "pch.hpp"
#include "configure_file_settings.hpp"

module tensor;

//import pch;

namespace hasty {

std::filesystem::path tensor_cache_dir = get_library_path() / TENSOR_CACHE_RELATIVE_PATH;

}
module;

#include "configure_file_settings.hpp"

module script_cache;

import std;
import util;

namespace hasty {
namespace script {

std::filesystem::path module_cache_dir = get_library_directory() / MODULE_CACHE_RELATIVE_PATH;;

}
}
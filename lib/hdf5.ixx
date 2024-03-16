module;

#include "pch.hpp"

export module hdf5;

import util;

namespace hasty {

	export at::Tensor import_tensor(const std::string& filepath, const std::string& dataset);
	
    export void export_tensor(const at::Tensor& tensor, const std::string& filepath, const std::string& dataset);

}

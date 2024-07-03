module;

#include "pch.hpp"

export module hdf5;

import util;

namespace hasty {

	export at::Tensor import_tensor(const std::string& filepath, const std::string& dataset);

    export void export_tensor(const at::Tensor& tensor, const std::string& filepath, const std::string& dataset);

    export auto import_tensors(const std::string& filepath, const std::optional<std::vector<std::regex>>& matchers) ->
        std::unordered_map<std::string, std::variant<at::Tensor, std::vector<at::Tensor>>>;

}

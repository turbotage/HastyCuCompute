module;

#include "pch.hpp"

export module hdf5;

//import pch;

import torch_base;
import util;

namespace hasty {

	export auto import_tensor(const std::string& filepath, const std::string& dataset) ->
        std::variant<hat::Tensor, std::vector<hat::Tensor>>;

    export auto import_tensors(const std::string& filepath, const std::optional<std::vector<std::regex>>& matchers) ->
        std::unordered_map<std::string, std::variant<hat::Tensor, std::vector<hat::Tensor>>>;

    export void export_tensor(const hat::Tensor& tensor, const std::string& filepath, const std::string& dataset);


}

module;

#include "pch.hpp"

export module trace;

import util;

namespace hasty {

    
    export struct trace {

        trace()
        {
            std::vector<at::Tensor> tensorlist;
        }

        void add_line(const std::string& line)
        {
            _tracestr += "\n\t" + line;
        }

    private:
        std::unordered_map<void*,at::Tensor> _tensors;
        std::string _tracestr = 
R"torchscript(
def trace_function(tensorlist):
)torchscript";
    };




}

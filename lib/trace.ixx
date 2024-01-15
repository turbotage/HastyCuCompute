module;

#include "util.hpp"
#include <unordered_map>
#include <torch/jit.h>

export module trace;

namespace hasty {

    
    struct trace {

        trace()
        {
            std::vector<at::Tensor> tensorlist;
        }

        void add_line(const std::string_view& line)
        {
            _tracerstr += "\n\t" + line;
        }

    private:
        std::unordered_map<void*,at::Tensor> _tensors;
        std::string _tracestr = 
R"torchscript(
def trace_function(tensorlist):
)torchscript";
    }


    

}

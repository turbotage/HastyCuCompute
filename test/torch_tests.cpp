#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/extension.h>
#include <torch/library.h>

at::Tensor test_func(at::Tensor t) {
    return t + 1;
}

TORCH_LIBRARY(my_test, m) {
    m.def("test_func(Tensor t) -> Tensor", test_func);
}

int main() {

    auto mod = torch::jit::Module("Module");
    mod.define(R"ts(
    def forward(self, x):
        return torch.ops.my_test.test_func(x)
    )ts");

    mod = torch::jit::freeze(mod);
    mod = torch::jit::optimize_for_inference(mod);

    auto res = mod({torch::randn({2,3})});

    std::cout << res << std::endl;

}
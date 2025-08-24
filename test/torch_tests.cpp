#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/extension.h>

int main() {

    auto mod = torch::jit::Module("Module");
    mod.define(R"ts(
    def forward(self, x):
        with torch.no_grad():
            return x**2 + torch.sin(x)
    )ts");

    mod = torch::jit::freeze(mod);
    mod = torch::jit::optimize_for_inference(mod);

    auto res = mod({torch::randn({2,3})});



    std::cout << res << std::endl;

}
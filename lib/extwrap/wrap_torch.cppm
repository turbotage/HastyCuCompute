module;

#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "c10/cuda/CUDAStream.h"
#include <c10/cuda/CUDAGuard.h>


export module torch_base;


//export using at::Tensor;
//export using at::IntArrayRef;
//export using at::TensorOptions;

export template <typename Cond, typename... Args>
inline void torch_check(Cond&& cond, Args&&... args) {
    TORCH_CHECK(std::forward<Cond>(cond), std::forward<Args>(args)...);
}

export inline std::ostream& operator<<(std::ostream& os, const c10::FunctionSchema& schema) {
    return ::operator<<(os, schema); // calls the original operator<<
}

export namespace hat {
    // Types
    using at::Tensor;
    using at::Scalar;
    using at::ScalarType;
    using at::ArrayRef;
    using at::IntArrayRef;
    using at::TensorOptions;
    using at::Device;
    using at::DeviceType;
    using at::DeviceIndex;
    using at::Stream;
    using at::OptionalArrayRef;
    using at::InferenceMode;

    inline constexpr auto kFloat = at::kFloat;
    inline constexpr auto kDouble = at::kDouble;
    inline constexpr auto kComplexFloat = at::kComplexFloat;
    inline constexpr auto kComplexDouble = at::kComplexDouble;
    inline constexpr auto kInt = at::kInt;
    inline constexpr auto kLong = at::kLong;
    inline constexpr auto kShort = at::kShort;
    inline constexpr auto kByte = at::kByte;
    inline constexpr auto kBool = at::kBool;


    using at::empty;
    using at::zeros;
    using at::ones;
    using at::rand;
    using at::empty_like;
    using at::zeros_like;
    using at::ones_like;
    using at::rand_like;
    using at::real;
    using at::imag;
    using at::makeArrayRef;
    using at::from_blob;
    using at::scalar_tensor;
    using at::complex;
    using at::stack;
    using at::cat;
    using at::vdot;
    using at::exp;



    namespace indexing {
        using at::indexing::TensorIndex;
        inline constexpr auto None = at::indexing::None;
        using at::indexing::Ellipsis;
        using at::indexing::Slice;
    }
    namespace cuda {
        using at::cuda::CUDAGuard;
        namespace CUDACachingAllocator {
            using at::cuda::CUDACachingAllocator::getDeviceStats;
            using at::cuda::CUDACachingAllocator::emptyCache;
        }

        using at::cuda::getDefaultCUDAStream;
        using at::cuda::device_count;
    }


}

export namespace htorch {

    using torch::NoGradGuard;

    namespace cuda {
        using torch::cuda::synchronize;
    }
    namespace fft {
        using torch::fft::fftn;
        using torch::fft::ifftn;
        using torch::fft::rfftn;
        using torch::fft::irfftn;
    }
    namespace jit {
        using torch::jit::CompilationUnit;
        using torch::jit::Module;
        using torch::jit::IValue;

        using torch::jit::freeze;
        using torch::jit::optimize_for_inference;
        using torch::jit::load;
        using torch::jit::getAllOperators;
    }

}

export namespace hc10 {
    using c10::SymInt;
    using c10::Error;
    using c10::string_view;
    using c10::complex;
}

/*
export namespace hc10 {
    using c10::optional
}
*/
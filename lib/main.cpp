#include <iostream>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-io/xhighfive.hpp>

#include <matx.h>

import nufft;
import tensor;

int main() {
    std::array<size_t, 3> shape = {1,3,3};
    xt::xtensor<double, 3> test_tensor(shape);

    xt::dump_hdf5("test.h5", "testdata", test_tensor);

    auto test_tensor_back = xt::load_hdf5<xt::xtensor<double, 3>>("test.h5", "testdata");

    std::cout << "test_tensor: " << test_tensor << std::endl;

    std::cout << "test_tensor_back: " << test_tensor_back << std::endl;

    hasty::cuda_nufft_opts<float, 3, hasty::nufft_type::TYPE_2> opts{
        .nmodes={128,128,128}, .sign=hasty::nufft_sign::DEFAULT_TYPE_2, .ntransf=32, .tol=1e-5};

    auto plan = hasty::nufft_make_plan(opts);

    auto image = matx::make_tensor<cuda::std::complex<float>, 4>({32, 128, 128, 128});

    auto randOp = matx::random<cuda::std::complex<float>>(image.Shape(), matx::UNIFORM);
    (image = -3.141592f + 2*3.141592f*randOp).run();
    
    
    
    return 0;
}
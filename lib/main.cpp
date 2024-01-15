
#include "util.hpp"
#include <cuda_runtime_api.h>

import nufft;
import tensor;



float run_nufft_type2_timing(int ntransf, int nx, int ny, int nz, int nupts, hasty::nufft_cuda_method method, hasty::nufft_upsamp upsamp) {

    hasty::cuda_nufft_opts<hasty::cuda_f32, 3, hasty::nufft_type::TYPE_2> opts{
        .nmodes={nx,ny,nz}, 
        .sign=hasty::nufft_sign::DEFAULT_TYPE_2, 
        .ntransf=ntransf, 
        .tol=1e-4,
        
        .upsamp=upsamp,
        .method=method
        };

    std::unique_ptr<hasty::cuda_nufft_plan<hasty::cuda_f32, 3, hasty::nufft_type::TYPE_2>> plan = 
        hasty::nufft_make_plan(opts);


    hasty::tensor<hasty::cuda_c64, 2> kspace = hasty::make_tensor<hasty::cuda_c64, 2>({ntransf, nupts}, "cuda:0");
    hasty::tensor<hasty::cuda_c64, 4> image = hasty::make_tensor<hasty::cuda_c64, 4>({ntransf,nx,ny,nz}, "cuda:0");

    std::array<hasty::tensor<hasty::cuda_f32, 1>, 3> coord = { 
        hasty::make_tensor<hasty::cuda_f32,1>({nupts}, "cuda:0"),
        hasty::make_tensor<hasty::cuda_f32,1>({nupts}, "cuda:0"),
        hasty::make_tensor<hasty::cuda_f32,1>({nupts}, "cuda:0") 
    };


    hasty::nufft_setpts(*plan, coord);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);

    for (int i = 0; i < 100; ++i) {
        hasty::nufft_execute(*plan, image, kspace);
    }

    float elapsed;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    return elapsed;
}

void type_2_tests(int ntransf, int nx, int ny, int nz, int nupts) {
    
    std::cout << "UPSAMP: 2.0\n";
    {   
        // GM_NO_SORT
        std::cout << "GM_NO_SORT: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_NO_SORT, hasty::nufft_upsamp::DEFAULT);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_NO_SORT, hasty::nufft_upsamp::DEFAULT);
        std::cout << "\n";

        // GM_SORT
        std::cout << "GM_SORT: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_SORT, hasty::nufft_upsamp::DEFAULT);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_SORT, hasty::nufft_upsamp::DEFAULT);
        std::cout << "\n";

        // SM
        std::cout << "SM: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::SM, hasty::nufft_upsamp::DEFAULT);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::SM, hasty::nufft_upsamp::DEFAULT);
        std::cout << "\n";
    }


    std::cout << "UPSAMP: 1.25\n";
    {   
        // GM_NO_SORT
        std::cout << "GM_NO_SORT: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_NO_SORT, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_NO_SORT, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "\n";

        // GM_SORT
        std::cout << "GM_SORT: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_SORT, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::GM_SORT, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "\n";

        // SM
        std::cout << "SM: \n";
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::SM, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "ms: " << run_nufft_type2_timing(ntransf, nx, ny, nz, nupts, 
            hasty::nufft_cuda_method::SM, hasty::nufft_upsamp::UPSAMP_1_25);
        std::cout << "\n";
    }


}


int main() {

    int ntransf = 32;
    int nx = 170;
    int ny = 170;
    int nz = 170;
    int nupts = 175000;

    type_2_tests(ntransf, nx, ny, nz, nupts);
    
    


    return 0;
}



import nufft;
import tensor;



float run_nufft_type2_timing(int ntransf, int nx, int ny, int nz, int nupts, hasty::nufft_cuda_method method, hasty::nufft_upsamp upsamp) {

    hasty::cuda_nufft_opts<float, 3, hasty::nufft_type::TYPE_2> opts{
        .nmodes={nx,ny,nz}, 
        .sign=hasty::nufft_sign::DEFAULT_TYPE_2, 
        .ntransf=ntransf, 
        .tol=1e-5,
        
        .upsamp=upsamp,
        .method=method
        };

    auto plan = hasty::nufft_make_plan(opts);


    

    hasty::nufft_setpts(*plan, coords);

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

int main() {

    std::cout << "ms: " << run_nufft_type2_timing(32, 128, 128, 128, 300000, 
        hasty::nufft_cuda_method::DEFAULT, hasty::nufft_upsamp::DEFAULT);

    return 0;
}
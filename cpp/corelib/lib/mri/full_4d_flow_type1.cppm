module;

#include "pch.hpp"

export module mri:full_4d_flow_type1;

import threading;

namespace hasty {

    tensor<cpu_t,complex_t<f32_t>> gridding_estimate(
        storage_thread_pool& resourced_thread_pool,
        const tensor<cpu_t,complex_t<f32_t>,4>& data,
        const trajectory<complex_t<f32_t>,3>& traj,
        const cache_tensor<complex_t<f32_t,2>>& bil,
        const cache_tensor<complex_t<f32_t>,4>& cjl
    ) 
    {
        
    }

    tensor<cpu_t,complex_t<f32_t>,4> create_rhs(
        storage_thread_pool& resourced_thread_pool,
        const tensor<cpu_t,complex_t<f32_t>,4>& smaps,
        const tensor<cpu_t,complex_t<f32_t>,4>& data,
        const tensor<cpu_t,complex_t<f32_t>,4>& fixed_img,
        const tensor<cpu_t,complex_t<f32_t>,1>& diagonal,
        const tensor<cpu_t,complex_t<f32_t>,1>& lambdas,
        const tensor<cpu_t,b8_t,4>& interior_mask,
        const tensor<cpu_t,b8_t,1>& fixed_mask
    )
    {
        auto output = make_empty_tensor_like(data);
        auto d_output = output.template to<cuda_t>(device_idx::GPU);

    }

}
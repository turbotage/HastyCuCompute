module;

#include "pch.hpp"

export module forward;

import util;
import tensor;

namespace hasty {

    template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
    tensor<cpu_t,TT,DIM+1> sense_adjoint_offresonance(
        trajectory<real_t<TT>,D> traj, 
        cache_tensor<TT,2>& kdata,
        cache_tensor<TT,DIM+1>& smaps, 
        cache_tensor<TT,DIM>& offrensonance, 
        std::vector<std::pair<
            base_t<real_t<TT>>,
            base_t<TT>
        >>&& interpt_interpcoeff,
        storage_thread_pool& thread_pool) 
    {
        auto& coords = traj.coords();

        auto run_lambda = [&coords, &kdata, &smaps, &offrensonance, &interpt_interpcoeff](storage& store) {
            device_idx didx = store.device_idx();

        };


        if (data.)

    }

}

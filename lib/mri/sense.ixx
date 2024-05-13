module;

#include "../pch.hpp"

export module sense;

import trajectory;
import tensor;
import nufft;

namespace hasty {

    template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
    class sense_normal {
    public:

        using device_type_t = D;
        using input_tensor_type_t = TT;
        using output_tensor_type_t = TT;
        static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
        static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

        sense_normal(cache_tensor<D,TT,DIM-1>&& kernel, cache_tensor<D,TT,DIM>&& smaps)
            : _kernel(kernel), _smaps(smaps)
        {}

        sense_normal(const trajectory<D,TT,DIM-1>& traj, span<DIM-1> shape, device_idx didx, bool precise,
            tensor<D,TT,1>& smaps)
            : _smaps(smaps)
        {
            auto M = traj.coords[0].shape<0>();

            auto twoshape = shape * 2;

            using UTT = up_precision_t<TT>;
            if (precise && !std::is_same_v<UTT, TT>) {

                auto upkernel = make_tensor<D,UTT,DIM-1>(span<DIM-1>(twoshape));

                std::array<tensor<D,UTT,1>,DIM-1> coords;
                for_sequence<DIM-1>([&](auto i) {
                    coords[i] = traj.coords[i].to<UTT>();
                });

                auto ones = make_tensor<D,UTT,2>({1, M}, didx, tensor_make_opts::ONES);

                nufft::toeplitz_kernel<D,UTT,DIM-1> upkernel(coords, upkernel, ones);

                _kernel = upkernel.to<TT>();
            } else {

                auto ones = make_tensor<D,TT,2>({1, M}, didx, tensor_make_opts::ONES);

                _kernel = make_tensor<D,TT,DIM-1>(span<DIM-1>(twoshape));

                nufft::toeplitz_kernel<D,TT,DIM-1> kernel(traj.coords, _kernel, ones);
            }
        }
        {
            auto M = traj.coords[0].shape<0>();

            auto twoshape = shape * 2;

            using UTT = up_precision_t<TT>;
            if (precise && !std::is_same_v<UTT, TT>) {

                auto upkernel = make_tensor<D,UTT,DIM-1>(span<DIM-1>(twoshape));

                std::array<tensor<D,UTT,1>,DIM-1> coords;
                for_sequence<DIM-1>([&](auto i) {
                    coords[i] = traj.coords[i].to<UTT>();
                });

                auto ones = make_tensor<D,UTT,2>({1, M}, didx, tensor_make_opts::ONES);

                nufft::toeplitz_kernel<D,UTT,DIM-1> upkernel(coords, upkernel, ones);

                _kernel = upkernel.to<TT>();
            } else {

                auto ones = make_tensor<D,TT,2>({1, M}, didx, tensor_make_opts::ONES);

                _kernel = make_tensor<D,TT,DIM-1>(span<DIM-1>(twoshape));

                nufft::toeplitz_kernel<D,TT,DIM-1> kernel(traj.coords, _kernel, ones);
            }
        }

        tensor<D,TT,DIM> operator()(const tensor<D,TT,DIM>& x) {
            return x;
        }

    private:
        cache_tensor<D,TT,DIM-1> _kernel;
        cache_tensor<D,TT,DIM> _smaps;
    };

}
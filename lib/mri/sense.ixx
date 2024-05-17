module;

#include "../pch.hpp"

export module sense;

import util;
import trajectory;
import tensor;
import trace;
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

        sense_normal(cache_tensor<D,TT,DIM>&& kernel, cache_tensor<D,TT,DIM+1>&& smaps)
            : _kernel(kernel), _smaps(smaps)
        {}

        sense_normal(const trajectory<D,TT,DIM>& traj, cache_tensor<D,TT,1>&& smaps, span<DIM> shape, device_idx didx, bool precise)
            : _smaps(smaps)
        {
            auto M = traj.coords[0].template shape<0>();

            auto twoshape = shape * 2;

            using UTT = up_precision_t<TT>;
            if (precise && !std::is_same_v<UTT, TT>) {

                auto upkernel = make_tensor<D,UTT,DIM>(span<DIM>(twoshape));

                std::array<tensor<D,UTT,1>,DIM> coords;
                for_sequence<DIM-1>([&](auto i) {
                    coords[i] = traj.coords[i].template to<UTT>();
                });

                auto ones = make_tensor<D,UTT,2>({1, M}, didx, tensor_make_opts::ONES);

                toeplitz_kernel(coords, upkernel, ones);

                _kernel = upkernel.template to<TT>();
            } else {

                auto ones = make_tensor<D,TT,2>({1, M}, didx, tensor_make_opts::ONES);

                _kernel = make_tensor<D,TT,DIM>(span<DIM>(twoshape));

                toeplitz_kernel(traj.coords, _kernel, ones);
            }


            trace::tensor_prototype<D,TT,DIM>     input("input");
            trace::tensor_prototype<D,TT,DIM+1>   coilmap("coilmap");
            trace::tensor_prototype<D,TT,DIM>     kernel("kernel");
            trace::tensor_prototype<D,TT,DIM>     output("output");

            _toeplitz = trace::trace_function_factory<decltype(output)>::make("toeplitz", input, coilmap, kernel);

            _toeplitz.add_lines(std::format(R"ts(
    #shp = input.shape
    spatial_shp = input.shape #shp[1:]
    expanded_shp = [2*s for s in spatial_shp]
    transform_dims = [i+1 for i in range(len(spatial_shp))]

    ncoil = coilmap.shape[0]
    nrun = ncoil // {0}
    
    out = torch.zeros_like(input)
    for run in range(nrun):
        bst = run*{0}
        cmap = coilmap[bst:(bst+{0})]
        c = cmap * input
        c = torch.fft_fftn(c, expanded_shp, transform_dims)
        c *= kernel
        c = torch.fft_ifftn(c, None, transform_dims)

        for dim in range(len(spatial_shp)):
            c = torch.slice(c, dim+1, spatial_shp[dim]-1, -1)

        c *= cmap.conj()
        out += torch.sum(c, 0)

    out *= (1 / torch.prod(torch.tensor(spatial_shp)))
    
    return out
)ts", 2));

            _toeplitz.compile();

        }

        tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& x) {
            auto didx = x.get_device_idx();
            std::tuple<tensor<D,TT,DIM>> output_data = _toeplitz.run(x, 
                _smaps.template get<D>(didx), _kernel.template get<D>(didx));
            return std::get<0>(output_data);
        }

    private:
        cache_tensor<D,TT,DIM> _kernel;
        cache_tensor<D,TT,DIM+1> _smaps;

        using RETT = trace::tensor_prototype<D,TT,DIM>;
        using IN1 = trace::tensor_prototype<D,TT,DIM+1>;
        using IN2 = trace::tensor_prototype<D,TT,DIM>;

        trace::trace_function<std::tuple<RETT>, std::tuple<IN1,IN2,IN1>> _toeplitz;
    };

}
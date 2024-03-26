module;

#include "pch.hpp"

#include <cufinufft_opts.h>
#include <cufinufft.h>

export module nufft;

export import util;
export import tensor;

namespace hasty {

    // UTIL

    template<size_t N>
    concept is_dim3 = N > 0 && N < 4;

    template<device_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void verify_coords(const std::array<tensor<FPT,1>,DIM>& coords)
    {
        if constexpr(DIM == 2) {
            if (coords[0].template shape<0>() != coords[1].template shape<0>()) {
                throw std::runtime_error("x and y coords have different lengths");
            }
        }
        if constexpr(DIM == 3) {
            if (
                (coords[0].template shape<0>() != coords[1].template shape<0>()) || 
                (coords[1].template shape<0>() != coords[2].template shape<0>()) ||
                (coords[0].template shape<0>() != coords[2].template shape<0>())
            ){  throw std::runtime_error("x,y  x,z  or  y,z  coords have different lengths"); }
        }
    }


    export enum struct nufft_type {
        TYPE_1 = 1,
        TYPE_2 = 2,
        TYPE_3 = 3,
        FORWARD = TYPE_2,
        BACKWARD = TYPE_1
    };

    export enum struct nufft_sign {
        POS = 1,
        NEG = -1,
        DEFAULT_TYPE_1 = POS,
        DEFAULT_TYPE_2 = NEG
    };
    
    // CUDA SETTINGS

    export enum struct nufft_upsamp_cuda {
        UPSAMP_2_0,
        UPSAMP_1_25,
        DEFAULT,
    };

    export enum struct nufft_method_cuda {
        GM_NO_SORT,
        GM_SORT,
        SM,
        DEFAULT
    }; 

    // NUFFT OPTS

    export template<typename FPT, size_t DIM, typename... Variadic>
    requires device_real_fp<FPT> && is_dim3<DIM>
    struct nufft_opts {};

    export template<typename FPT, size_t DIM> 
    requires cuda_real_fp<FPT> && is_dim3<DIM>
    struct nufft_opts<FPT,DIM> {

        std::array<int64_t, DIM> nmodes;
        int32_t ntransf;
        underlying_type<FPT> tol;

        nufft_sign sign;
        nufft_upsamp_cuda upsamp;
        nufft_method_cuda method;

        std::shared_ptr<cudaStream_t> pstream;
        int32_t device_idx;

        inline int32_t get_sign() const { return static_cast<int>(sign); }

    };

    // NUFFT PLAN

    export template<typename FPT, size_t DIM, nufft_type NT, typename... Variadic>
    requires device_real_fp<FPT> && is_dim3<DIM>
    struct nufft_plan {};

    export template<typename FPT, size_t DIM, nufft_type NT>
    requires cuda_real_fp<FPT> && is_dim3<DIM>
    struct nufft_plan<FPT,DIM,NT> {

        static std::unique_ptr<nufft_plan<FPT,DIM,NT>> make(const nufft_opts<FPT,DIM>& opts)
        {
            struct starter : public nufft_plan<FPT, DIM, NT> {
                starter() : nufft_plan<FPT, DIM, NT>() {}
            };

            std::unique_ptr<nufft_plan<FPT, DIM, NT>> ret = std::make_unique<starter>();

            ret->_opts = opts;

            cufinufft_default_opts(&ret->_finufft_opts);

            switch (opts.method) {
                case nufft_method_cuda::GM_NO_SORT:
                {
                    ret->_finufft_opts.gpu_method = 1;
                    ret->_finufft_opts.gpu_sort = false;
                } break;
                case nufft_method_cuda::GM_SORT:
                {
                    ret->_finufft_opts.gpu_method = 1;
                    ret->_finufft_opts.gpu_sort = true;
                }break;
                case nufft_method_cuda::SM:
                {
                    ret->_finufft_opts.gpu_method = 2;
                    ret->_finufft_opts.gpu_sort = true;
                }break;
                case nufft_method_cuda::DEFAULT:
                break;
                default:
                    throw std::runtime_error("Invalid nufft_cuda_method");
            }

            switch (opts.upsamp) {
                case nufft_upsamp_cuda::UPSAMP_2_0:
                {
                    ret->_finufft_opts.upsampfac = 2.0;
                    //ret->_finufft_opts.gpu_kerevalmeth = 0;
                }break;
                case nufft_upsamp_cuda::UPSAMP_1_25:
                {
                    ret->_finufft_opts.upsampfac = 1.25;
                    ret->_finufft_opts.gpu_kerevalmeth = 0;
                }break;
                case nufft_upsamp_cuda::DEFAULT:
                {} break;
                default:
                throw std::runtime_error("nufft_make_plan encountered invalid nufft_upsamp argument");
            }

            int result;
            if constexpr(std::is_same_v<FPT, cuda_f32>) {
                result = cufinufftf_makeplan(static_cast<int>(NT), DIM, opts.nmodes.data(), opts.get_sign(),
                        opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
            } else {
                result = cufinufft_makeplan(static_cast<int>(NT), DIM, opts.nmodes.data(), opts.get_sign(),
                        opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
            }

            if (result) 
                throw std::runtime_error("cufinufft: makeplan failed, code: " + std::to_string(result));
            
            return ret;

        }


        void setpts(const std::array<tensor<FPT, 1>,DIM>& coords)
        {
            int32_t M = coords[0].template shape<0>();
            verify_coords(coords);

            using namespace std::placeholders;
            constexpr auto setptsfunc = std::bind([]()
            { 
                if constexpr(std::is_same_v<FPT,cuda_f32>) {
                    return cufinufftf_setpts;
                } else {
                    return cufinufft_setpts;
                }
            }(), _1, _2, _3, _4, _5, 0, nullptr, nullptr, nullptr);

            int result;
            if constexpr(DIM == 1) {
                result = setptsfunc(_finufft_plan, M, coords[0].unconsted_data_ptr(), nullptr, nullptr);
            }
            else if constexpr(DIM == 2) {
                result = setptsfunc(_finufft_plan, M, coords[1].unconsted_data_ptr(), coords[0].unconsted_data_ptr(), nullptr);
            }
            else if constexpr(DIM == 3) {
                result = setptsfunc(_finufft_plan, M, coords[2].unconsted_data_ptr(), coords[1].unconsted_data_ptr(), coords[0].unconsted_data_ptr());
            }

            if (result)
                throw std::runtime_error("cufinufft: setpts function failed with code: " + std::to_string(result));
        }
        
        template<nufft_type U = NT>
        requires (U == nufft_type::TYPE_1)
        void execute(const tensor<complex_t<FPT>,2>& input, tensor<complex_t<FPT>, DIM+1>& output) const
        {
            int result;

            if (input.template shape<0>() != _opts.ntransf)
                throw std::runtime_error("input first dimension must match ntransf");

            if constexpr(std::is_same_v<FPT, cuda_f32>) {
                result = cufinufftf_execute(_finufft_plan, input.unconsted_data_ptr(), output.mutable_data_ptr());
            } else {
                result = cufinufft_execute(_finufft_plan, input.unconsted_data_ptr(), output.mutable_data_ptr());
            }

            if (result)
                throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
        }

        template<nufft_type U = NT>
        requires (U == nufft_type::TYPE_2)
        void execute(const tensor<complex_t<FPT>, DIM+1>& input, tensor<complex_t<FPT>, 2>& output) const
        {
            int result;

            if constexpr(std::is_same_v<FPT, cuda_f32>) {
                result = cufinufftf_execute(_finufft_plan, output.mutable_data_ptr(), input.unconsted_data_ptr());
            } else {
                result = cufinufft_execute(_finufft_plan, output.mutable_data_ptr(), input.unconsted_data_ptr());
            }

            if (result)
                throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
        }
        

        
        auto& pa_finufft_plan() { return _finufft_plan; }
        const auto& pa_finufft_plan() const { return _finufft_plan; }

        auto& pa_finufft_opts() { return _finufft_opts; }
        const auto& pa_finufft_opts() const { return _finufft_opts; }

        auto& pa_coords() { return _coords; }
        const auto& pa_coords() const { return _coords; }

        template<size_t D>
        requires less_than<D,DIM>
        auto& pa_coords() { return _coords[D]; }

        template<size_t D>
        requires less_than<D,DIM>
        const auto& pa_coords() const { return _coords[D]; }

        nufft_opts<FPT,DIM>& pa_opts() { return _opts; }
        const nufft_opts<FPT,DIM>& pa_opts() const { return _opts; }

        const std::array<int64_t, DIM>& nmodes() const { return _opts.nmodes; }
        int32_t ntransf() const { return _opts.ntransf; }
        underlying_type<FPT> tol() const { return _opts.tol;}


        void free() {
            if (_finufft_plan != nullptr) {
                int result;
                if constexpr(std::is_same_v<FPT, cuda_f32>) {
                    result = cufinufftf_destroy(_finufft_plan);
                } else {
                    result = cufinufft_destroy(_finufft_plan);
                }
                _finufft_plan = nullptr;

                if (result) {
                    std::string errstr = "cufinufft: destroy failed, code: " + std::to_string(result);
                    std::cerr << errstr;
                    // if we ended up here after free call in destructor, this may crash the application 
                    // if the user require nothrow behaviour, always manually free first
                    throw std::runtime_error(errstr);
                }
            }
        }

        ~nufft_plan() {
            free();
        }

    private:
        nufft_opts<FPT,DIM> _opts;
        std::array<tensor<FPT,1>, DIM> _coords;

        cufinufft_opts _finufft_opts;
        std::conditional_t<std::is_same_v<FPT,cuda_f32>, 
            cufinufftf_plan, cufinufft_plan> _finufft_plan;
    };

    // TOEPLITZ KERNEL
    
    export template<cuda_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void toeplitz_kernel(const std::array<tensor<FPT, 1>,DIM>& coords, tensor<complex_t<FPT>, DIM>& kernel,
        tensor<complex_t<FPT>, 2>& nudata)
    {
        int M = coords[0].template shape<0>();
        verify_coords(coords);

        if (nudata.template shape<1>() != M)
            throw std::runtime_error("shape<1>() of nudata didn't match length of coord vectors");

        //nudata.fill_(1.0f);

        std::array<int64_t, DIM+1> one_nmodes;
        one_nmodes[0] = 1;
        std::array<int64_t, DIM> nmodes;
        int64_t nvox = 1;
        for_sequence<DIM>([&nmodes, &one_nmodes, &kernel, &nvox](auto i) {
            int64_t dimsize = kernel.template shape<i>();
            nmodes[i] = dimsize - 1;
            one_nmodes[i+1] = nmodes[i];
            nvox *= (dimsize / 2);
        });

        using FLTYPE = std::conditional_t<is_32bit_precission<FPT>(), float, double>;
        FLTYPE normfactor = double(1.0) / double(nvox);

        auto reduced_kernel = make_tensor<complex_t<FPT>, DIM+1>(span(one_nmodes), kernel.devicestr());

        {
            auto plan = nufft_plan<FPT,DIM,nufft_type::TYPE_1>::make(nufft_opts<FPT,DIM>{
                .nmodes = nmodes,
                .ntransf = 1,
                .tol = std::is_same_v<FPT,cuda_f32> ? 1e-5 : 1e-12,
                .sign = nufft_sign::DEFAULT_TYPE_1,
                .upsamp = nufft_upsamp_cuda::DEFAULT,
                .method = nufft_method_cuda::DEFAULT
            });

            plan->setpts(coords);
            plan->execute(nudata, reduced_kernel);
        }

        std::array<Slice, DIM> slices;
        slices[0] = Slice();
        for_sequence<DIM>([&slices, &nmodes](auto i) {
            slices[i] = Slice{0, nmodes[i]};
        });
 
        //reduced_kernel.fill_(1.0);

        kernel[slices] = reduced_kernel[0];

        kernel = fftn(kernel, nullspan(), nullspan(), std::nullopt);
        kernel *= normfactor;
    }

    export template<cuda_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void toeplitz_multiply(tensor<FPT,DIM+1>& inout, const tensor<FPT,DIM>& kernel)
    {
        int64_t batchsize = inout.template shape<0>();

        if constexpr(DIM == 1) {
            int64_t xsize = inout.template shape<1>();
            auto mid = fftn(inout, span<1>({2 * xsize}), 
                span<1>({1}), std::nullopt);
            mid *= kernel;
            mid = ifftn(mid, nullspan(), span<1>({1}), std::nullopt);
            inout = mid[Slice(), Slice(xsize-1, -1)];
        } else if constexpr(DIM == 2) {
            int64_t xsize = inout.template shape<1>();
            int64_t ysize = inout.template shape<2>();
            auto mid = fftn(inout, span<2>({2 * xsize, 2 * ysize}), 
                span<2>({1,2}), std::nullopt);
            mid *= kernel;
            mid = ifftn(mid, nullspan(), span<2>({1,2}), std::nullopt);
            inout = mid[Slice(), Slice(xsize-1, -1), Slice(ysize-1, -1)];
        } else if constexpr(DIM == 3) {
            int64_t xsize = inout.template shape<1>();
            int64_t ysize = inout.template shape<2>();
            int64_t zsize = inout.template shape<3>();
            auto mid = fftn(inout, span<3>({2 * xsize, 2 * ysize, 2 * zsize}), 
                span<3>({1,2,3}), std::nullopt);
            mid *= kernel;
            mid = ifftn(mid, nullspan(), span<3>({1,2,3}), std::nullopt);
            inout = mid[Slice(), Slice(xsize-1, -1), Slice(ysize-1, -1), Slice(zsize-1, -1)];
        }
    }


    
    export template<cuda_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void cuda_toeplitz_multiply(tensor<FPT,DIM+1>& inout, const tensor<FPT,DIM>& kernel);


    

}
module;

#include <cufinufft_opts.h>
#include <cufinufft.h>
#include "util.hpp"

export module nufft;

import tensor;

namespace hasty {

    template<size_t N>
    concept is_dim3 = N > 0 && N < 4;

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

    export enum struct nufft_upsamp {
        UPSAMP_2_0,
        UPSAMP_1_25,
        DEFAULT,
    };

    export enum struct nufft_cuda_method {
        GM_NO_SORT,
        GM_SORT,
        SM,
        DEFAULT
    }; 

    export template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM>
    struct cuda_nufft_opts {

        std::array<int64_t,DIM> nmodes;

        nufft_sign sign;

        int32_t ntransf;
        underlying_type<FPT> tol;

        nufft_upsamp upsamp = nufft_upsamp::UPSAMP_2_0;
        nufft_cuda_method method = nufft_cuda_method::DEFAULT; 

        inline int get_sign() const { return static_cast<int>(sign); }

    };  

    export template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM>
    struct cuda_nufft_plan {
    public:


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

        ~cuda_nufft_plan() {
            free();
        } 

    protected:

        cuda_nufft_plan() {}

    private:
        cufinufft_opts _finufft_opts;
        std::conditional_t<std::is_same_v<FPT,cuda_f32>, cufinufftf_plan, cufinufft_plan> _finufft_plan = nullptr;

        cuda_nufft_opts<FPT, DIM, NFT> _opts;
        std::array<tensor<FPT,1>,DIM> _coords;

        template<cuda_real_fp F, size_t D, nufft_type N> 
        requires is_dim3<D>
        friend std::unique_ptr<cuda_nufft_plan<F, D, N>> nufft_make_plan(const cuda_nufft_opts<F, D, N>& opts);

        template<cuda_real_fp F, size_t D, nufft_type N>
        friend cufinufft_opts& cudaplan_finufft_opts(cuda_nufft_plan<F,D,N>& plan);

        template<cuda_real_fp F, size_t D, nufft_type N>
        friend std::conditional_t<std::is_same_v<F,cuda_f32>, cufinufftf_plan, cufinufft_plan>& cudaplan_finufft_plan(cuda_nufft_plan<F,D,N>& plan);

        template<cuda_real_fp F, size_t D, nufft_type N>
        friend cuda_nufft_opts<F,D,N>& cudaplan_opts(cuda_nufft_plan<F,D,N>& plan);

        template<cuda_real_fp F, size_t D, nufft_type N>
        friend std::array<tensor<F, 1>,D>& cudaplan_coords(cuda_nufft_plan<F,D,N>& plan);

    };

    export template<cuda_real_fp F, size_t D, nufft_type N> 
    requires is_dim3<D>
    std::unique_ptr<cuda_nufft_plan<F, D, N>> nufft_make_plan(const cuda_nufft_opts<F, D, N>& opts)
    {

        struct starter : public cuda_nufft_plan<F, D, N> {
            starter() : cuda_nufft_plan<F, D, N>() {}
        };

        std::unique_ptr<cuda_nufft_plan<F, D, N>> ret = std::make_unique<starter>();

        ret->_opts = opts;

        cufinufft_default_opts(&ret->_finufft_opts);
        
        switch (opts.method) {
            case nufft_cuda_method::GM_NO_SORT:
            {
                ret->_finufft_opts.gpu_method = 1;
                ret->_finufft_opts.gpu_sort = false;
            } break;
            case nufft_cuda_method::GM_SORT:
            {
                ret->_finufft_opts.gpu_method = 1;
                ret->_finufft_opts.gpu_sort = true;
            }break;
            case nufft_cuda_method::SM:
            {
                ret->_finufft_opts.gpu_method = 2;
                ret->_finufft_opts.gpu_sort = true;
            }break;
            case nufft_cuda_method::DEFAULT:
            break;
            default:
                throw std::runtime_error("Invalid nufft_cuda_method");
        }

        switch (opts.upsamp) {
            case nufft_upsamp::UPSAMP_2_0:
            {
                ret->_finufft_opts.upsampfac = 2.0;
                //ret->_finufft_opts.gpu_kerevalmeth = 0;
            }break;
            case nufft_upsamp::UPSAMP_1_25:
            {
                ret->_finufft_opts.upsampfac = 1.25;
                ret->_finufft_opts.gpu_kerevalmeth = 0;
            }break;
            case nufft_upsamp::DEFAULT:
            {} break;
            default:
            throw std::runtime_error("nufft_make_plan encountered invalid nufft_upsamp argument");
        }

        int result;
        if constexpr(std::is_same_v<F, cuda_f32>) {
            result = cufinufftf_makeplan(static_cast<int>(N), D, opts.nmodes.data(), opts.get_sign(),
                    opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
        } else {
            result = cufinufft_makeplan(static_cast<int>(N), D, opts.nmodes.data(), opts.get_sign(),
                    opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
        }

        if (result) 
            throw std::runtime_error("cufinufft: makeplan failed, code: " + std::to_string(result));
        
        return ret;
    }

    template<cuda_real_fp F, size_t D, nufft_type N>
    cufinufft_opts& cudaplan_finufft_opts(cuda_nufft_plan<F,D,N>& plan) 
    { 
        return plan._finufft_opts; 
    }

    template<cuda_real_fp F, size_t D, nufft_type N>
    std::conditional_t<std::is_same_v<F,cuda_f32>, cufinufftf_plan, cufinufft_plan>& cudaplan_finufft_plan(cuda_nufft_plan<F,D,N>& plan) 
    { 
        return plan._finufft_plan; 
    } 

    template<cuda_real_fp F, size_t D, nufft_type N>
    cuda_nufft_opts<F,D,N>& cudaplan_opts(cuda_nufft_plan<F,D,N>& plan) 
    { 
        return plan._opts;
    }

    template<cuda_real_fp F, size_t D, nufft_type N>
    std::array<tensor<F, 1>,D>& cudaplan_coords(cuda_nufft_plan<F,D,N>& plan) 
    { 
        return plan._coords; 
    }



    export template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM> && (static_cast<int>(NFT) == 1 || static_cast<int>(NFT) == 2)
    void nufft_setpts(cuda_nufft_plan<FPT, DIM, NFT>& plan, const std::array<tensor<FPT, 1>, DIM>& coords)
    {

        int M = coords[0].template shape<0>();

        auto& plancoords = cudaplan_coords(plan);
        plancoords = coords;

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
            ){  throw std::runtime_error("x and y coords have different lengths"); }
        }
        
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
            result = setptsfunc(cudaplan_finufft_plan(plan), M, coords[0].const_cast_data(), nullptr, nullptr);
        }
        else if constexpr(DIM == 2) {
            result = setptsfunc(cudaplan_finufft_plan(plan), M, coords[0].const_cast_data(), coords[1].const_cast_data(), nullptr);
        }
        else if constexpr(DIM == 3) {
            result = setptsfunc(cudaplan_finufft_plan(plan), M, coords[0].const_cast_data(), coords[1].const_cast_data(), coords[2].const_cast_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: setpts function failed with code: " + std::to_string(result));
    }

    export template<cuda_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void nufft_execute( cuda_nufft_plan<FPT, DIM, nufft_type::TYPE_1>& plan, 
                        const tensor<complexify_type<FPT>, 2>& input,
                        tensor<complexify_type<FPT>, DIM+1>& output)
    {
        int result;

        if constexpr(std::is_same_v<FPT, cuda_f32>) {
            result = cufinufftf_execute(cudaplan_finufft_plan(plan), input.const_cast_data(), output.mutable_data());
        } else {
            result = cufinufft_execute(cudaplan_finufft_plan(plan), input.const_cast_data(), output.mutable_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }

    export template<cuda_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void nufft_execute( cuda_nufft_plan<FPT, DIM, nufft_type::TYPE_2>& plan, 
                        const tensor<complexify_type<FPT>, DIM+1>& input,
                        tensor<complexify_type<FPT>, 2>& output)
    {
        int result;

        if constexpr(std::is_same_v<FPT, cuda_f32>) {
            result = cufinufftf_execute(cudaplan_finufft_plan(plan), output.mutable_data(), input.const_cast_data());
        } else {
            result = cufinufft_execute(cudaplan_finufft_plan(plan), output.mutable_data(), input.const_cast_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }











}
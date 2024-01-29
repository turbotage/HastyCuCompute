module;

#include "pch.hpp"

#include <cufinufft_opts.h>
#include <cufinufft.h>

export module nufft;

import util;
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

    template<device_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    struct nufft_opts;

    export template<cuda_real_fp FPT, size_t DIM> 
    requires is_dim3<DIM>
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


    template<device_real_fp FPT, size_t DIM, nufft_type NT>
    struct nufft_plan;

    template<cuda_real_fp FPT, size_t DIM, nufft_type NT>
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
            if constexpr(std::is_same_v<FPT, f32>) {
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
                if constexpr(std::is_same_v<FPT,f32>) {
                    return cufinufftf_setpts;
                } else {
                    return cufinufft_setpts;
                }
            }(), _1, _2, _3, _4, _5, 0, nullptr, nullptr, nullptr);

            int result;
            if constexpr(DIM == 1) {
                result = setptsfunc(_finufft_plan, M, coords[0].const_cast_data(), nullptr, nullptr);
            }
            else if constexpr(DIM == 2) {
                result = setptsfunc(_finufft_plan, M, coords[0].const_cast_data(), coords[1].const_cast_data(), nullptr);
            }
            else if constexpr(DIM == 3) {
                result = setptsfunc(_finufft_plan, M, coords[0].const_cast_data(), coords[1].const_cast_data(), coords[2].const_cast_data());
            }

            if (result)
                throw std::runtime_error("cufinufft: setpts function failed with code: " + std::to_string(result));
        }


        void execute(const tensor<complex_cuda_t<FPT>,>);



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

        nufft_opts<FPT,DIM,device_type::CUDA>& pa_opts() { return _opts; }
        const nufft_opts<FPT,DIM,device_type::CUDA>& pa_opts() const { return _opts; }


        void free() {
            if (_finufft_plan != nullptr) {
                int result;
                if constexpr(std::is_same_v<FPT, f32>) {
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
        nufft_opts<FPT,DIM,device_type::CUDA> _opts;
        std::array<tensor<cuda_type_t<FPT>,1>, DIM> _coords;

        cufinufft_opts _finufft_opts;
        std::conditional_t<std::is_same_v<FPT,f32>, 
            cufinufftf_plan, cufinufft_plan> _finufft_plan;
    };





}
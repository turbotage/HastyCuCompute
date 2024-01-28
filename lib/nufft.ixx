module;

#include <array>

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

    template<real_fp FPT, size_t DIM, device_type DT>
    requires is_dim3<DIM>
    struct nufft_opts;

    export template<floating_point FPT, size_t DIM> 
    requires is_dim3<DIM>
    struct nufft_opts<FPT,DIM,device_type::CUDA> {

        std::array<int64_t, DIM> nmodes;
        int32_t ntransf;
        FPT tol;

        nufft_sign sign;
        nufft_upsamp_cuda upsamp;
        nufft_method_cuda method;

        std::shared_ptr<cudaStream_t> pstream;
        int32_t device_idx;

        inline int32_t get_sign() const { return static_cast<int>(sign); }

    };


    template<floating_point FPT, size_t DIM, device_type DT>
    struct nufft_plan;

    template<floating_point FPT, size_t DIM>
    struct nufft_plan<FPT,DIM,device_type::CUDA> {


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


    private:
        nufft_opts<FPT,DIM,device_type::CUDA> _opts;
        std::array<tensor<FPT,1,device_type::CUDA>, DIM> _coords;

        cufinufft_opts _finufft_opts;
        std::conditional_t<std::is_same_v<FPT,float>, cufinufftf_plan, cufinufft_plan> _finufft_plan;
    };


    /*
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
    */

}
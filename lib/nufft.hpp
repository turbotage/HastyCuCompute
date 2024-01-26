#pragma once

#include <cufinufft_opts.h>
#include <cufinufft.h>
#include "util.hpp"



namespace hasty {

    template<size_t N>
    concept is_dim3 = N > 0 && N < 4;

    enum struct nufft_type {
        TYPE_1 = 1,
        TYPE_2 = 2,
        TYPE_3 = 3,
        FORWARD = TYPE_2,
        BACKWARD = TYPE_1
    };

    enum struct nufft_sign {
        POS = 1,
        NEG = -1,
        DEFAULT_TYPE_1 = POS,
        DEFAULT_TYPE_2 = NEG
    };

    enum struct nufft_upsamp {
        UPSAMP_2_0,
        UPSAMP_1_25,
        DEFAULT,
    };

    enum struct nufft_cuda_method {
        GM_NO_SORT,
        GM_SORT,
        SM,
        DEFAULT
    }; 

    template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM>
    struct cuda_nufft_opts {

        std::array<int64_t,DIM> nmodes;

        nufft_sign sign;

        int32_t ntransf;
        underlying_type<FPT> tol;

        nufft_upsamp upsamp = nufft_upsamp::UPSAMP_2_0;
        nufft_cuda_method method = nufft_cuda_method::DEFAULT; 

        std::shared_ptr<cudaStream_t> stream;
        int32_t deviceidx;

        inline int get_sign() const { return static_cast<int>(sign); }

    };  

    template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM>
    struct cuda_nufft_plan {
    public:

        using cufinufft_plan_type = std::conditional_t<std::is_same_v<FPT,cuda_f32>, cufinufftf_plan, cufinufft_plan>;


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


        cufinufft_opts& pa_cufinufft_opts() { return _finufft_opts; }
        const cufinufft_opts& pa_cufinufft_opts() const { return _finufft_opts; }

        cufinufft_plan_type& pa_finufft_plan() { return _finufft_plan; }
        const cufinufft_plan_type& pa_finufft_plan() const { return _finufft_plan; }

        cuda_nufft_opts<FPT,DIM,NFT>& pa_opts() { return _opts; }
        const cuda_nufft_opts<FPT,DIM,NFT>& pa_opts() const { return _opts; }

        std::array<tensor<FPT, 1>, D>& pa_coords() { return _coords; }
        const std::array<tensor<FPT, 1>, D>& pa_coords() const { return _coords; }

    protected:

        cuda_nufft_plan() {}

    private:
        cufinufft_opts _finufft_opts;
        cufinufft_plan_type _finufft_plan = nullptr;

        cuda_nufft_opts<FPT, DIM, NFT> _opts;
        std::array<tensor<FPT,1>,DIM> _coords;

        template<cuda_real_fp F, size_t D, nufft_type N> 
        requires is_dim3<D>
        friend std::unique_ptr<cuda_nufft_plan<F, D, N>> nufft_make_plan(const cuda_nufft_opts<F, D, N>& opts);

    };


    template<cuda_real_fp FPT, size_t DIM, nufft_type NFT> 
    requires is_dim3<DIM> && (static_cast<int>(NFT) == 1 || static_cast<int>(NFT) == 2)
    void nufft_setpts(cuda_nufft_plan<FPT, DIM, NFT>& plan, const std::array<tensor<FPT, 1>, DIM>& coords)
    {

        int M = coords[0].template shape<0>();

        auto& plancoords = plan.pa_coords();
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
            result = setptsfunc(plan.pa_finufft_plan(), M, coords[0].const_cast_data(), nullptr, nullptr);
        }
        else if constexpr(DIM == 2) {
            result = setptsfunc(plan.pa_finufft_plan(), M, coords[0].const_cast_data(), coords[1].const_cast_data(), nullptr);
        }
        else if constexpr(DIM == 3) {
            result = setptsfunc(plan.pa_finufft_plan(), M, coords[0].const_cast_data(), coords[1].const_cast_data(), coords[2].const_cast_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: setpts function failed with code: " + std::to_string(result));
    }

    template<cuda_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void nufft_execute( cuda_nufft_plan<FPT, DIM, nufft_type::TYPE_1>& plan, 
                        const tensor<complexify_type<FPT>, 2>& input,
                        tensor<complexify_type<FPT>, DIM+1>& output)
    {
        int result;

        if constexpr(std::is_same_v<FPT, cuda_f32>) {
            result = cufinufftf_execute(plan.pa_finufft_plan(), input.const_cast_data(), output.mutable_data());
        } else {
            result = cufinufft_execute(plan.pa_finufft_plan(), input.const_cast_data(), output.mutable_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }

    template<cuda_real_fp FPT, size_t DIM>
    requires is_dim3<DIM>
    void nufft_execute( cuda_nufft_plan<FPT, DIM, nufft_type::TYPE_2>& plan, 
                        const tensor<complexify_type<FPT>, DIM+1>& input,
                        tensor<complexify_type<FPT>, 2>& output)
    {
        int result;

        if constexpr(std::is_same_v<FPT, cuda_f32>) {
            result = cufinufftf_execute(plan.pa_finufft_plan(), output.mutable_data(), input.const_cast_data());
        } else {
            result = cufinufft_execute(plan.pa_finufft_plan(), output.mutable_data(), input.const_cast_data());
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }











}
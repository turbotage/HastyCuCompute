module;

#include <type_traits>
#include <matx.h>
#include <cufinufft_opts.h>
#include <cufinufft.h>

export module nufft;

static_assert(alignof(cuda::std::complex<double>) == alignof(cuDoubleComplex));
static_assert(sizeof(cuda::std::complex<double>) == sizeof(cuDoubleComplex));
static_assert(alignof(cuda::std::complex<float>) == alignof(cuFloatComplex));
static_assert(sizeof(cuda::std::complex<float>) == sizeof(cuFloatComplex));

namespace hasty {

    template<typename T>
    concept floating_point = std::is_same_v<float, T> || std::is_same_v<double, T>;
    
    template<typename T>
    concept complex_floating_point = std::is_same_v<cuda::std::complex<float>,T> || std::is_same_v<cuda::std::complex<float>,T>;

    template<size_t N>
    concept nufft_dim = N > 0 && N < 4;

    export enum nufft_type {
        TYPE_1 = 1,
        TYPE_2 = 2,
        TYPE_3 = 3,
        Forward = TYPE_2,
        Backward = TYPE_1
    };

    export template<floating_point FP_TYPE, size_t DIM> 
    requires nufft_dim<DIM>
    struct cuda_nufft_opts {

        nufft_type type;

        std::array<int64_t,DIM> nmodes;

        enum sign {
            POS = 1,
            NEG = -1
        } sign;

        int32_t ntransf;
        FP_TYPE tol;
    };

    export template<floating_point FP_TYPE, size_t DIM> 
    requires nufft_dim<DIM>
    struct cuda_nufft_plan {
    public:

        ~cuda_nufft_plan() {}

    protected:

        cuda_nufft_plan() {}

    private:
        cufinufft_opts _plan_opts;
        cuda_nufft_opts<FP_TYPE, DIM> _nufft_opts;
        std::conditional_t<std::is_same_v<FP_TYPE,float>, cufinufftf_plan, cufinufft_plan> _plan = nullptr;

        std::array<matx::tensor_t<FP_TYPE,1>,DIM> sources;

        template<floating_point FPT, size_t D> 
        requires nufft_dim<D>
        friend std::unique_ptr<cuda_nufft_plan<FPT,D>> make_nufft_plan(const cuda_nufft_opts<FPT, D>& opts);

        template<floating_point FPT, size_t D> 
        requires nufft_dim<D>
        void nufft_setpts(const cuda_nufft_plan<FPT, D>& plan, std::array<matx::tensor_t<FPT, 1>, D> sources);
    };

    export template<floating_point FP_TYPE, size_t DIM> 
    requires nufft_dim<DIM>
    std::unique_ptr<cuda_nufft_plan<FP_TYPE, DIM>> nufft_make_plan(const cuda_nufft_opts<FP_TYPE, DIM>& opts)
    {

        struct starter : public cuda_nufft_plan<FP_TYPE, DIM> {
            starter() : cuda_nufft_plan<FP_TYPE, DIM>() {}
        };

        auto ret = std::make_unique<starter>();

        ret->_nufft_opts = opts;

        cufinufft_default_opts(&ret->_plan_opts);
        
        if constexpr(std::is_same_v<FP_TYPE, float>) {
            int result = cufinufftf_makeplan(opts.type, DIM, opts.nmodes.data(), opts.sign,
                    opts.ntransf, opts.precission, &ret->_planf, &ret->_opts);
            if (result) 
                throw std::runtime_error("cufinufftf_makeplan failed, code: " + std::to_string(result));
        } else {
            int result = cufinufft_makeplan(opts.type, DIM, opts.nmodes.data(), opts.sign,
                    opts.ntransf, opts.precission, &ret->_plan, &ret->_opts);
            if (result) 
                throw std::runtime_error("cufinufft_makeplan failed, code: " + std::to_string(result));
        }
        
        return ret;
    }

    export template<floating_point FP_TYPE, size_t DIM> 
    requires nufft_dim<DIM>
    void nufft_setpts(const cuda_nufft_plan<FP_TYPE, DIM>& plan, std::array<matx::tensor_t<FP_TYPE, 1>, DIM> sources)
    {

        int M = sources[0].Size(0);
        
        if (plan.type == nufft_type::TYPE_3) {
            throw std::runtime_error("nufft_setpts reuires two sources for TYPE 3");
        }

        if constexpr(DIM == 2) {
            if (sources[0].Size(0) != sources[1].Size(1)) {
                throw std::runtime_error("x and y sources have different lengths");
            }
        }
        if constexpr(DIM == 3) {
            if (
                (sources[0].Size(0) != sources[1].Size(0)) || 
                (sources[1].Size(0) != sources[2].Size(0)) ||
                (sources[0].Size(0) != sources[2].Size(0))
            ){  throw std::runtime_error("x and y sources have different lengths"); }
        }

        matx::tensor_t<float,3> a;

        constexpr auto setptsfunc = [&](FP_TYPE* x, FP_TYPE* y, FP_TYPE* z) 
        { 
            if constexpr(std::is_same_v<FP_TYPE,float>) {
                return cufinufftf_setpts(plan->_plan, M, x, y, z, 0, nullptr, nullptr, nullptr);
            } else {
                return cufinufft_setpts(plan->_plan, M, x, y, z, 0, nullptr, nullptr, nullptr);
            }
        };

        int result;
        if constexpr(DIM == 1) {
            result = setptsfunc(sources[0].Data(), nullptr, nullptr);
        }
        else if constexpr(DIM == 2) {
            result = setptsfunc(sources[0].Data(), sources[1].Data(), nullptr);
        }
        else if constexpr(DIM == 3) {
            result = setptsfunc(sources[0].Data(), sources[1].Data(), sources[2].Data());
        }

        if (result)
            throw std::runtime_error("cufinufft setpts function failed with code: " + std::to_string(result));
    }

    export template<floating_point FP_TYPE, size_t DIM, nufft_type TYPE>
    requires nufft_dim<DIM> && (TYPE == nufft_type::TYPE_1)
    void nufft_execute( const cuda_nufft_plan<FP_TYPE, DIM>& plan, 
                        const matx::tensor_t<cuda::std::complex<FP_TYPE>, DIM+1>& input,
                        matx::tensor_t<cuda::std::complex<FP_TYPE>, 2>& output)
    {
        if constexpr(std::is_same_v<FP_TYPE, float>) {
            
        } else {

        }
    }

}
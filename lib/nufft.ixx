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
        FORWARD = TYPE_2,
        BACKWARD = TYPE_1
    };

    export enum nufft_sign {
        POS = 1,
        NEG = -1,
        DEFAULT_TYPE_1 = POS,
        DEFAULT_TYPE_2 = NEG
    };

    export template<floating_point FPT, size_t DIM, nufft_type NFT> 
    requires nufft_dim<DIM>
    struct cuda_nufft_opts {

        std::array<int64_t,DIM> nmodes;

        nufft_sign sign;

        int32_t ntransf;
        FPT tol;
    };

    export template<floating_point FPT, size_t DIM, nufft_type NFT> 
    requires nufft_dim<DIM>
    struct cuda_nufft_plan {
    public:


        void free() {
            if (_finufft_plan != nullptr) {
                int result;
                if constexpr(std::is_same_v<FPT, float>) {
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
        std::conditional_t<std::is_same_v<FPT,float>, cufinufftf_plan, cufinufft_plan> _finufft_plan = nullptr;

        cuda_nufft_opts<FPT, DIM, NFT> _opts;
        std::array<matx::tensor_t<FPT,1>,DIM> _coords;

        template<floating_point F, size_t D, nufft_type N> 
        requires nufft_dim<D>
        friend std::unique_ptr<cuda_nufft_plan<F, D, N>> nufft_make_plan(const cuda_nufft_opts<F, D, N>& opts);

        template<floating_point F, size_t D, nufft_type N>
        friend cufinufft_opts& cudaplan_finufft_opts(cuda_nufft_plan<F,D,N>& plan);

        template<floating_point F, size_t D, nufft_type N>
        friend cuda_nufft_opts<F,D,N>& cudaplan_opts(cuda_nufft_plan<F,D,N>& plan);

        template<floating_point F, size_t D, nufft_type N>
        friend std::conditional_t<std::is_same_v<F,float>, cufinufftf_plan, cufinufft_plan>& cudaplan_finufft_plan(cuda_nufft_plan<F,D,N>& plan);

        template<floating_point F, size_t D, nufft_type N>
        friend std::array<matx::tensor_t<F, 1>,D>& cudaplan_coords(cuda_nufft_plan<F,D,N>& plan);

    };

    export template<floating_point F, size_t D, nufft_type N> 
    requires nufft_dim<D>
    std::unique_ptr<cuda_nufft_plan<F, D, N>> nufft_make_plan(const cuda_nufft_opts<F, D, N>& opts)
    {

        struct starter : public cuda_nufft_plan<F, D, N> {
            starter() : cuda_nufft_plan<F, D, N>() {}
        };

        std::unique_ptr<cuda_nufft_plan<F, D, N>> ret = std::make_unique<starter>();

        ret->_opts = opts;

        cufinufft_default_opts(&ret->_finufft_opts);
        
        int result;
        if constexpr(std::is_same_v<F, float>) {
            int result = cufinufftf_makeplan(N, D, opts.nmodes.data(), opts.sign,
                    opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
        } else {
            int result = cufinufft_makeplan(N, D, opts.nmodes.data(), opts.sign,
                    opts.ntransf, opts.tol, &ret->_finufft_plan, &ret->_finufft_opts);
        }

        if (result) 
            throw std::runtime_error("cufinufft: makeplan failed, code: " + std::to_string(result));
        
        return ret;
    }

    template<floating_point F, size_t D, nufft_type N>
    inline cufinufft_opts& cudaplan_finufft_opts(cuda_nufft_plan<F,D,N>& plan) { return plan._plan_opts; }

    template<floating_point F, size_t D, nufft_type N>
    inline cuda_nufft_opts<F,D,N>& cudaplan_opts(cuda_nufft_plan<F,D,N>& plan) { return plan._nufft_opts; }

    template<floating_point F, size_t D, nufft_type N>
    inline std::conditional_t<std::is_same_v<F,float>, cufinufftf_plan, cufinufft_plan>& cudaplan_finufft_plan(cuda_nufft_plan<F,D,N>& plan) { return plan._plan; } 

    template<floating_point F, size_t D, nufft_type N>
    inline std::array<matx::tensor_t<F, 1>,D>& cudaplan_coords(cuda_nufft_plan<F,D,N>& plan) { return plan._coords; }



    template<floating_point FP_TYPE, size_t DIM, nufft_type NUFFT_TYPE> 
    requires nufft_dim<DIM> && (NUFFT_TYPE == 1 || NUFFT_TYPE == 2)
    void nufft_setpts(const cuda_nufft_plan<FP_TYPE, DIM, NUFFT_TYPE>& plan, std::array<matx::tensor_t<FP_TYPE, 1>, DIM> coords)
    {

        int M = coords[0].Size(0);

        auto& plancoords = cudaplan_coords(plan);
        plancoords = coords;

        if constexpr(DIM == 2) {
            if (coords[0].Size(0) != coords[1].Size(1)) {
                throw std::runtime_error("x and y coords have different lengths");
            }
        }
        if constexpr(DIM == 3) {
            if (
                (coords[0].Size(0) != coords[1].Size(0)) || 
                (coords[1].Size(0) != coords[2].Size(0)) ||
                (coords[0].Size(0) != coords[2].Size(0))
            ){  throw std::runtime_error("x and y coords have different lengths"); }
        }

        constexpr auto setptsfunc = [&](FP_TYPE* x, FP_TYPE* y, FP_TYPE* z) 
        { 
            if constexpr(std::is_same_v<FP_TYPE,float>) {
                return cufinufftf_setpts(cudaplan_finufft_plan(plan), M, x, y, z, 0, nullptr, nullptr, nullptr);
            } else {
                return cufinufft_setpts(cudaplan_finufft_plan(plan), M, x, y, z, 0, nullptr, nullptr, nullptr);
            }
        };

        int result;
        if constexpr(DIM == 1) {
            result = setptsfunc(coords[0].Data(), nullptr, nullptr);
        }
        else if constexpr(DIM == 2) {
            result = setptsfunc(coords[0].Data(), coords[1].Data(), nullptr);
        }
        else if constexpr(DIM == 3) {
            result = setptsfunc(coords[0].Data(), coords[1].Data(), coords[2].Data());
        }

        if (result)
            throw std::runtime_error("cufinufft: setpts function failed with code: " + std::to_string(result));
    }

    export template<floating_point FP_TYPE, size_t DIM>
    requires nufft_dim<DIM>
    void nufft_execute( const cuda_nufft_plan<FP_TYPE, DIM, nufft_type::TYPE_1>& plan, 
                        const matx::tensor_t<cuda::std::complex<FP_TYPE>, 2>& input,
                        matx::tensor_t<cuda::std::complex<FP_TYPE>, DIM+1>& output)
    {
        int result;

        if constexpr(std::is_same_v<FP_TYPE, float>) {
            result = cufinufftf_execute(plan->_plan, 
                static_cast<cuFloatComplex*>(input.Data()), static_cast<cuFloatComplex*>(output.Data()));
        } else {
            result = cufinufft_execute(plan->_plan, 
                static_cast<cuDoubleComplex*>(input.Data(), static_cast<cuDoubleComplex*>(output.Data())));
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }

    export template<floating_point FP_TYPE, size_t DIM>
    requires nufft_dim<DIM>
    void nufft_execute( const cuda_nufft_plan<FP_TYPE, DIM, nufft_type::TYPE_2>& plan, 
                        const matx::tensor_t<cuda::std::complex<FP_TYPE>, DIM+1>& input,
                        matx::tensor_t<cuda::std::complex<FP_TYPE>, 2>& output)
    {
        int result;

        if constexpr(std::is_same_v<FP_TYPE, float>) {
            result = cufinufftf_execute(plan->_plan, 
                static_cast<cuFloatComplex*>(output.Data()), static_cast<cuFloatComplex*>(input.Data()));
        } else {
            result = cufinufft_execute(plan->_plan, 
                static_cast<cuDoubleComplex*>(output.Data()), static_cast<cuDoubleComplex*>(input.Data()));
        }

        if (result)
            throw std::runtime_error("cufinufft: execute failed with error code: " + std::to_string(result));
    }

}
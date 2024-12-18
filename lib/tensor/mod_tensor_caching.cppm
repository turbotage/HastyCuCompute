module;

#include "pch.hpp"

export module tensor:caching;

import util;
import :intrinsic;

namespace hasty {
    
    export extern std::filesystem::path cache_dir;

    export template<is_tensor_type TT, size_t RANK>
    class cache_tensor {
    private:

        struct block {
            std::mutex mutex;

            sptr<tensor<cpu_t,TT,RANK>> tensor_cpu;
            std::array<i64, RANK> shape;
            size_t hashidx;
            
            std::array<sptr<tensor<cuda_t,TT,RANK>>, size_t(device_idx::MAX_CUDA_DEVICES)> cuda_tensors;
        };
        sptr<block> _block;

        
        void clean_cuda() 
        {
            for (auto& ct : _block->cuda_tensors) {
                // Only the cache tensor is holding the cuda tensor so it's unecessary to keep it

                if (ct.use_count() == 1) {
                    ct = nullptr;
                }
            }
        }

        auto get_cpu() -> tensor<cpu_t,TT,RANK>& {
            if (_block->tensor_cpu)
                return *_block->tensor_cpu;

            uncache_disk();

            return *_block->tensor_cpu;
        }

        auto get_cpu_ptr() -> sptr<tensor<cpu_t,TT,RANK>>&  {
            if (_block->tensor_cpu)
                return _block->tensor_cpu;

            uncache_disk();

            return _block->tensor_cpu;
        }

        void cache_disk() {

            if (_block->tensor_cpu == nullptr)
                throw std::runtime_error("cache_disk: no tensor to cache");

            auto tt = _block->tensor_cpu->get_tensor().contiguous();

            export_binary_tensor(
                std::move(tt), 
                cache_dir / std::filesystem::path(std::to_string(_block->hashidx) + ".htc")
            );
        }

        void uncache_disk() {
            namespace fs = std::filesystem;

            auto tt = import_binary_tensor(
                        cache_dir / fs::path(std::to_string(_block->hashidx) + ".htc"), 
                        span(_block->shape).to_arr_ref(), 
                        scalar_type_func<TT>());

            _block->tensor_cpu = std::make_shared<tensor<cpu_t,TT,RANK>>(_block->shape, std::move(tt));
        }

        auto get_cuda(device_idx didx) -> tensor<cuda_t,TT,RANK>& {
            if (_block->cuda_tensors[i32(didx)])
                return *_block->cuda_tensors[i32(didx)];

            _block->cuda_tensors[i32(didx)] = std::make_shared<tensor<cuda_t,TT,RANK>>(get_cpu_ptr()->template to<cuda_t>(didx));

            return *_block->cuda_tensors[i32(didx)];
        }

        auto get_cuda_ptr(device_idx didx) -> sptr<tensor<cuda_t,TT,RANK>>& {
            if (_block->cuda_tensors[i32(didx)])
                return _block->cuda_tensors[i32(didx)];

            _block->cuda_tensors[i32(didx)] = std::make_shared<tensor<cuda_t,TT,RANK>>(get_cpu_ptr()->template to<cuda_t>(didx));

            return _block->cuda_tensors[i32(didx)];
        }

    public:

        cache_tensor() = default;

        cache_tensor(tensor<cpu_t,TT,RANK>&& cputen, size_t hashidx)
        {
            _block = std::make_shared<block>();
            _block->shape = cputen.shape();
            _block->tensor_cpu = std::make_shared<tensor<cpu_t,TT,RANK>>(std::move(cputen));
            _block->hashidx = hashidx;
        }

        template<is_device D>
        auto get_ptr(device_idx idx = device_idx::CPU) -> sptr<tensor<D,TT,RANK>> {
            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu_ptr();
            } else {
                return get_cuda_ptr(idx);
            }
        }

        template<is_device D>
        auto get(device_idx idx = device_idx::CPU) -> tensor<D,TT,RANK> {
            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu();
            } else {
                return get_cuda(idx);
            }
        }

        void cache() {
            std::unique_lock<std::mutex> lock(_block->mutex);
            cache_disk();
        }

        void free_cpu() {
            std::unique_lock<std::mutex> lock(_block->mutex);
            _block->tensor_cpu = nullptr;
        }

        void free(device_idx idx) {
            std::unique_lock<std::mutex> lock(_block->mutex);
            _block->cuda_tensors[i32(idx)] = nullptr;
        }

        void uncache() {
            std::unique_lock<std::mutex> lock(_block->mutex);
            uncache_disk();
        }

    };


    // Explicit instantiations
    template class cache_tensor<f32_t,1>;
    template class cache_tensor<f32_t,2>;
    template class cache_tensor<f32_t,3>;
    template class cache_tensor<f32_t,4>;

    template class cache_tensor<f64_t,1>;
    template class cache_tensor<f64_t,2>;
    template class cache_tensor<f64_t,3>;
    template class cache_tensor<f64_t,4>;

    template class cache_tensor<c64_t,1>;
    template class cache_tensor<c64_t,2>;
    template class cache_tensor<c64_t,3>;
    template class cache_tensor<c64_t,4>;

    template class cache_tensor<c128_t,1>;
    template class cache_tensor<c128_t,2>;
    template class cache_tensor<c128_t,3>;
    template class cache_tensor<c128_t,4>;

    template class cache_tensor<i16_t,1>;
    template class cache_tensor<i16_t,2>;
    template class cache_tensor<i16_t,3>;
    template class cache_tensor<i16_t,4>;

    template class cache_tensor<i32_t,1>;
    template class cache_tensor<i32_t,2>;
    template class cache_tensor<i32_t,3>;
    template class cache_tensor<i32_t,4>;

    template class cache_tensor<i64_t,1>;
    template class cache_tensor<i64_t,2>;
    template class cache_tensor<i64_t,3>;
    template class cache_tensor<i64_t,4>;

    template class cache_tensor<b8_t,1>;
    template class cache_tensor<b8_t,2>;
    template class cache_tensor<b8_t,3>;
    template class cache_tensor<b8_t,4>;


}
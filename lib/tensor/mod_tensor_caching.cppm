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

        auto get_cpu() const -> const tensor<cpu_t,TT,RANK>& {
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

        auto get_cpu_ptr() const -> const sptr<tensor<cpu_t,TT,RANK>>&  {
            if (_block->tensor_cpu)
                return _block->tensor_cpu;

            uncache_disk();

            return _block->tensor_cpu;
        }

        void cache_disk() const {

            if (_block->tensor_cpu == nullptr)
                throw std::runtime_error("cache_disk: no tensor to cache");

            auto tt = _block->tensor_cpu->get_tensor().contiguous();

            export_binary_tensor(
                std::move(tt), 
                cache_dir / std::filesystem::path(std::to_string(_block->hashidx) + ".htc")
            );
        }

        void uncache_disk() const {
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

        auto get_cuda(device_idx didx) const -> const tensor<cuda_t,TT,RANK>& {
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

        auto get_cuda_ptr(device_idx didx) const -> const sptr<tensor<cuda_t,TT,RANK>>& {
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

        template<size_t R>
		requires less_than<R, RANK>
		int64_t shape() const
        {
            if (!_block)
                throw std::runtime_error("shape: cache_tensor is not initialized");
            return _block->shape[R];
        }

        cache_tensor<TT,RANK> copy() const {
            return *this;
        }

        /*
        @brief Be careful with this operator, if asked device is cpu it will return a view of the tensor,
        if cuda is asked for and the tensor is already cuda backed, it will return a view of the tensor,
        if it is not cuda backed, it will create a new cuda tensor that sliced the backing cpu tensor.
        I.e no view. This behaviour is intentional. It allows a user to make sure that large cpu tensors
        that should not be copied to cuda are not copied. But slices of the tensors can be copied into
        cuda. I.e with slicing, you will not populate the cuda cache of this cache_tensor.
        */
        template<class D, size_t N>
        requires less_than_or_equal<N, RANK> && less_than<0, RANK>
        auto operator[](device_idx didx, const std::array<Slice, N>& slices) const -> tensor<D, TT, RANK>
        {
            if (!_block)
                throw std::runtime_error("operator[]: cache_tensor is not initialized");

            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu()[slices];
            } else {
                if (_block->cuda_tensors[i32(didx)]) {
                    return get_cuda(didx)[slices];
                } else {
                    return get_cpu()[slices].template to<cuda_t>(didx);
                }
            }
        }

        /*
        @brief Be careful with this operator, if asked device is cpu it will return a view of the tensor,
        if cuda is asked for and the tensor is already cuda backed, it will return a view of the tensor,
        if it is not cuda backed, it will create a new cuda tensor that sliced the backing cpu tensor.
        I.e no view. This behaviour is intentional. It allows a user to make sure that large cpu tensors
        that should not be copied to cuda are not copied. But slices of the tensors can be copied into
        cuda. I.e with slicing, you will not populate the cuda cache of this cache_tensor.
        */
        template<class D, index_type ...Idx>
        requires less_than<0, RANK>
        auto operator[](device_idx didx, std::tuple<Idx...> indices) const -> tensor<D, TT, RANK>
        {
            if (!_block)
                throw std::runtime_error("operator[]: cache_tensor is not initialized");

            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu()[indices];
            } else {
                if (_block->cuda_tensors[i32(didx)]) {
                    return get_cuda(didx)[indices];
                } else {
                    return get_cpu()[indices].template to<cuda_t>(didx);
                }
            }
        }

        /*
        @brief Be careful with this operator, if asked device is cpu it will return a view of the tensor,
        if cuda is asked for and the tensor is already cuda backed, it will return a view of the tensor,
        if it is not cuda backed, it will create a new cuda tensor that sliced the backing cpu tensor.
        I.e no view. This behaviour is intentional. It allows a user to make sure that large cpu tensors
        that should not be copied to cuda are not copied. But slices of the tensors can be copied into
        cuda. I.e with slicing, you will not populate the cuda cache of this cache_tensor.
        */
        template<class D, index_type ...Idx>
        requires less_than<0, RANK>
        auto operator[](device_idx didx, Idx... indices) const
        {
            if (!_block)
                throw std::runtime_error("operator[]: cache_tensor is not initialized");

            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu()[indices...];
            } else {
                if (_block->cuda_tensors[i32(didx)]) {
                    return get_cuda(didx)[indices...];
                } else {
                    return get_cpu()[indices...].template to<cuda_t>(didx);
                }
            }
        }


        template<class D>
        auto operator[](device_idx didx, const tensor<D, b8_t, RANK>& mask) const -> tensor<D, TT, 1>
        {
            if (!_block)
                throw std::runtime_error("operator[]: cache_tensor is not initialized");
            
            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu()[mask];
            } else {
                if (_block->cuda_tensors[i32(didx)]) {
                    return get_cuda(didx)[mask];
                } else {
                    return get_cpu()[mask].template to<cuda_t>(didx);
                }
            }
        }


        template<is_device D>
        auto get_ptr(device_idx idx = device_idx::CPU) -> sptr<tensor<D,TT,RANK>> {
            if (!_block)
                throw std::runtime_error("get_ptr: cache_tensor is not initialized");
            
            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu_ptr();
            } else {
                return get_cuda_ptr(idx);
            }
        }

        template<is_device D>
        auto get(device_idx idx = device_idx::CPU) -> tensor<D,TT,RANK> {
            if (!_block)
                throw std::runtime_error("get: cache_tensor is not initialized");

            std::unique_lock<std::mutex> lock(_block->mutex);

            if constexpr(std::is_same_v<D,cpu_t>) {
                return get_cpu();
            } else {
                return get_cuda(idx);
            }
        }

        void cache() {
            if (!_block)
                throw std::runtime_error("cache: cache_tensor is not initialized");
            std::unique_lock<std::mutex> lock(_block->mutex);
            cache_disk();
        }

        void free_cpu() {
            if (!_block)
                throw std::runtime_error("free_cpu: cache_tensor is not initialized");
            std::unique_lock<std::mutex> lock(_block->mutex);
            _block->tensor_cpu = nullptr;
        }

        void free(device_idx idx) {
            if (!_block)
                throw std::runtime_error("free: cache_tensor is not initialized");
            std::unique_lock<std::mutex> lock(_block->mutex);
            _block->cuda_tensors[i32(idx)] = nullptr;
        }

        void uncache() {
            if (!_block)
                throw std::runtime_error("uncache: cache_tensor is not initialized");
            std::unique_lock<std::mutex> lock(_block->mutex);
            uncache_disk();
        }

    };


    /*
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
    */

}
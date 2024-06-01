module;

#include "pch.hpp"
#include <cuda_runtime.h>

export module threading;

import util;
import tensor;

namespace hasty {

    export std::vector<device_idx> get_cuda_devices(std::function<bool(const cudaDeviceProp&)> device_selector = 
        [](const cudaDeviceProp& prop)
        { 
            return true; 
        }
    ) 
    {

        i32 device_count = torch::cuda::device_count();
        std::vector<device_idx> devices;
        devices.reserve(device_count);
        for (int idx = 0; idx < device_count; idx++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, idx);
            if (device_selector(prop)) {
                devices.push_back(device_idx(idx));
            }

        }
        return devices;
    }

    export template<is_tensor_type TT> 
    struct cuda_context_holder {
        
        using tensor_type_t = TT;

        using TypeTuple = TupleTraits<
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,0>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,1>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,2>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,3>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,4>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,5>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,6>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,7>>
        >;

        TypeTuple::Tuple tensors;

        std::array<std::set<std::string>, 8> tensor_names;

    
        template<size_t RANK>
        void add(const std::string& name, cache_tensor<cuda_t,TT,RANK>&& tensor) {
            tensor_names[RANK].insert(name);
            std::get<RANK>(tensors).emplace(name, std::move(tensor));
        }

        template<size_t RANK>
        cache_tensor<cuda_t,TT,RANK>& get(const std::string& name) {
            return std::get<RANK>(tensors).at(name);
        }

        template<size_t RANK>
        void remove(const std::string& name) {
            tensor_names[RANK].erase(name);
            std::get<RANK>(tensors).erase(name);
        }

        void remove(const std::string& name) {
            for_sequence<TypeTuple::Size>([this,&name](auto i) {
                tensor_names[i].erase(name);
                std::get<i>(tensors).erase(name);
            });
        }

        template<size_t RANK>
        void clear() {
            tensor_names[RANK].clear();
            std::get<RANK>(tensors).clear();
        }

        void clear() {
            for_sequence<TypeTuple::Size>([this](auto i) {
                tensor_names[i].clear();
                std::get<i>(tensors).clear();
            });
        }

        std::set<std::string> names() {
            std::set<std::string> names;
            for_sequence<TypeTuple::Size>([&names, this](auto i) {
                for (const auto& name : tensor_names[i]) {
                    names.insert(name);
                }
            });
            return names;
        }

    };

    export struct cuda_context {

        using TypeTuple = TupleTraits<
            cuda_context_holder<f32_t>,
            cuda_context_holder<f64_t>,
            cuda_context_holder<c64_t>,
            cuda_context_holder<c128_t>,
            cuda_context_holder<i16_t>,
            cuda_context_holder<i32_t>,
            cuda_context_holder<i64_t>,
            cuda_context_holder<b8_t>
        >;
        
        TypeTuple::Tuple holders;


        template<is_tensor_type TT>
        constexpr cuda_context_holder<TT>& holder() {
            std::reference_wrapper<cuda_context_holder<TT>> retholder;
            for_sequence<TypeTuple::Size>([&retholder, this](auto i) {
                if constexpr(std::is_same_v<typename TypeTuple::Nth<i>::tensor_type_t, TT>) 
                {
                    retholder = std::get<i>(holders);
                }
            });
            return retholder.get();
        }

        template<is_tensor_type TT, size_t RANK>
        void add(const std::string& name, cache_tensor<cuda_t,TT,RANK>&& tensor) {
            holder<TT>().add(name, std::move(tensor));
        }

        template<is_tensor_type TT, size_t RANK>
        cache_tensor<cuda_t,TT,RANK>& get(const std::string& name) {
            return holder<TT>().template get<RANK>(name);
        }

        template<is_tensor_type TT, size_t RANK>
        void remove(const std::string& name) {
            holder<TT>().template remove<RANK>(name);
        }

        template<is_tensor_type TT>
        void remove(const std::string& name) {
            holder<TT>().remove(name);
        }

        void remove(std::string& name) {
            for_sequence<TypeTuple::Size>([this, &name](auto i) {
                std::get<i>(holders).remove(name);
            });
        }

        template<is_tensor_type TT, size_t RANK>
        void clear() {
            holder<TT>().template clear<RANK>();
        }

        template<is_tensor_type TT>
        void clear() {
            holder<TT>().clear();
        }

        void clear() {
            for_sequence<TypeTuple::Size>([this](auto i) {
                std::get<i>(holders).clear();
            });
        }

        std::set<std::string> names() {
            std::set<std::string> names;
            for_sequence<TypeTuple::Size>([this, &names](auto i) {
                auto innernames = std::get<i>(holders).names();
                for (const auto& name : innernames) {
                    names.insert(name);
                }
            });
            return names;
        }

    };


    export class storage {
    private:
        struct key {
            std::string name;
            std::type_index type;

            bool operator==(const key& other) const {
                return name == other.name && type == other.type;
            }

            auto operator()(const key& p) const -> size_t {
                return std::hash<std::string>()(p.name) ^ p.type.hash_code();
            }

        };
        
        std::unordered_map<key, std::shared_ptr<void>> _storage;
        std::unordered_multimap<std::string, key> _names;

    public:

        template<typename T>
        std::optional<std::reference_wrapper<T>> get_ref(const std::string& name) {
            key k = {name, typeid(T)};
            auto it = _storage.find(k);
            if (it != _storage.end()) {
                return std::optional<std::reference_wrapper<T>>(*(T*)it->second.get());
            } else {
                return std::nullopt;
            }
        }

        template<typename T>
        std::shared_ptr<T> get_ptr(const std::string& name) {
            key k = {name, typeid(T)};
            auto it = _storage.find(k);
            if (it != _storage.end()) {
                std::shared_ptr<T> ret = std::static_pointer_cast<T>(it->second);
                return ret;
            } else {
                return nullptr;
            }
        }

        template<typename T>
        void add(const std::string& name, std::shared_ptr<T> ptr) {
            key k = {name, typeid(T)};
            std::shared_ptr<void> inptr = std::static_pointer_cast<void>(ptr);
            _storage.insert(std::make_pair(k, inptr));
            _names.insert(std::make_pair(name, k));
        }

        template<typename T>
        void clear(const std::string& name) {
            key k = {name, typeid(T)};
            auto eqit = _names.equal_range(name);
            for (auto it = eqit.first; it != eqit.second; it++) {
                if (it->second == k) {
                    _storage.erase(it->second);
                    _names.erase(it);
                    break;
                }
            }
        }

        bool exist(const std::string& name) {
            return _names.contains(name);
        };

    };

    export class cuda_thread_pool {
    public:

        cuda_thread_pool(std::vector<cuda_context>&& contexts)
            : _stop(false), _work_length(0), _contexts(contexts)
        {
            _nthreads = _contexts.size();
            _threads.resize(_nthreads);
            try {
                for (size_t i = 0; i < _nthreads; i++) {
                    _threads.emplace_back([this, i]() { work(i); });
                }    
            }
            catch (...) {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stop = true;
                }
                _cv.notify_all();
                for (auto& thread : _threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
                throw;
            }
        }

        ~cuda_thread_pool() {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _stop = true;
            }
            _cv.notify_all();
            for (auto& thread : _threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

        template<class F, class... Args>
        std::future<typename std::invoke_result<F, cuda_context&, Args...>::type> enqueue(F&& f, std::set<std::string> names, Args&&... args) {
            using return_type = typename std::invoke_result<F, cuda_context&, Args...>::type;

            auto task = std::make_shared<std::packaged_task<return_type(cuda_context&)>>(
                std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();
            std::function<void(cuda_context&)> task_func = [task](cuda_context& context) { (*task)(context); };
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _work.push_back(std::make_pair(task, names));
            }
            _cv.notify_one();
            _work_length += 1;
            return res;
        }

    private:

        void work(size_t index) {
            auto& context = _contexts[index];

            i32 sleeps_since_work = 0;

            while (true) {
                
                std::optional<std::function<void(cuda_context&)>> task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    // We do quite alot of work here under a mutex lock, 
                    // idea is that we will likely have quite long running
                    // tasks, so overhead is small
                    while (_work.empty()) {
                        if (_stop) {
                            return;
                        }
                        _cv.wait(lock);
                    }

                    auto context_names = context.names();
                    
                    if (context_names.size() != 0) {
                        // Find most suitable work item
                        bool found = false;
                        for (auto workit = _work.begin(); workit != _work.end(); workit++) {
                            for (const auto& element : workit->second) {
                                if (context_names.count(element) > 0) {
                                    task = workit->first;
                                    _work.erase(workit);
                                    found = true;
                                    break;
                                }
                            }
                            if (found) {
                                break;
                            }
                        }

                        // If no suitable work item found, sleep and try again, when sleep count is too high,
                        // take the first item in the list
                        if (!found) {
                            sleeps_since_work += 1;
                            if (sleeps_since_work > 100) {
                                task = _work.front().first;
                                _work.pop_front();
                            }
                        }
                    } else {
                        task = _work.front().first;
                        _work.pop_front();
                    }

                }

                // If a suitable task was found, of if we have slept for too long, execute the task
                if (task.has_value()) {
                    task.value()(context);
                    sleeps_since_work = 0;
                    _work_length -= 1;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        }

    private:
        i32 _nthreads;
        std::vector<cuda_context> _contexts;
        std::vector<std::thread> _threads;

        std::condition_variable _cv;
        std::mutex _mutex;

        std::list<std::pair<std::function<void(cuda_context&)>, std::set<std::string>>> _work;

        bool _stop;
        std::atomic<int> _work_length;

    };


}
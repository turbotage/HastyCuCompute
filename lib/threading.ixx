module;

//#include "pch.hpp"
#include <cuda_runtime.h>

export module threading;

import pch;
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

    export class storage {
    private:

        struct key {
            std::string name;
            std::type_index type;

            bool operator==(const key& other) const {
                return name == other.name && type == other.type;
            }

        };

        struct key_hash {
            std::size_t operator()(const key& k) const {
                return std::hash<std::string>()(k.name) ^ std::hash<std::type_index>()(k.type);
            }
        };

        struct key_equal {
            bool operator()(const key& lhs, const key& rhs) const {
                return lhs == rhs;
            }
        };
        
        std::unordered_map<key, std::shared_ptr<void>, key_hash, key_equal> _storage;
        std::unordered_multimap<std::string, key> _names;

    public:

        template<typename T>
        auto get_ref(const std::string& name) -> std::optional<std::reference_wrapper<T>> {
            key k = {name, typeid(T)};
            auto it = _storage.find(k);
            if (it != _storage.end()) {
                return std::optional<std::reference_wrapper<T>>(*(T*)it->second.get());
            } else {
                return std::nullopt;
            }
        }

        template<typename T>
        auto get_ptr(const std::string& name) -> std::shared_ptr<T> {
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

        std::set<std::string> names() {
            std::set<std::string> names;
            for (const auto& [name, key] : _names) {
                names.insert(name);
            }
            return names;
        }

        std::vector<key> keys() {
            std::vector<key> keys;
            keys.reserve(_storage.size());
            for (const auto& [key, ptr] : _storage) {
                keys.push_back(key);
            }
            return keys;
        }

    };

    export class storage_thread_pool {
    public:

        storage_thread_pool(std::vector<storage>&& storages)
            : _stop(false), _work_length(0), _storages(storages)
        {
            _nthreads = _storages.size();
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

        ~storage_thread_pool() {
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
        std::future<typename std::invoke_result<F, storage&, Args...>::type> enqueue(F&& f, std::set<std::string> names, Args&&... args) {
            using return_type = typename std::invoke_result<F, storage&, Args...>::type;

            auto task = std::make_shared<std::packaged_task<return_type(storage&)>>(
                std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();
            std::function<void(storage&)> task_func = [task](storage& store) { (*task)(store); };
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
            auto& store = _storages[index];

            i32 sleeps_since_work = 0;

            while (true) {
                
                std::optional<std::function<void(storage&)>> task;
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

                    auto store_names = store.names();
                    
                    if (store_names.size() != 0) {
                        // Find most suitable work item
                        bool found = false;
                        for (auto workit = _work.begin(); workit != _work.end(); workit++) {
                            for (const auto& element : workit->second) {
                                if (store_names.count(element) > 0) {
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
                    task.value()(store);
                    sleeps_since_work = 0;
                    _work_length -= 1;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        }

    private:
        i32 _nthreads;
        std::vector<storage> _storages;
        std::vector<std::thread> _threads;

        std::condition_variable _cv;
        std::mutex _mutex;

        std::list<
            std::pair<
                std::function<void(storage&)>, 
                std::set<std::string>
            >
        > _work;

        bool _stop;
        std::atomic<int> _work_length;

    };

}


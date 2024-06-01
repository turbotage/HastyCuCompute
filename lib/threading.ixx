module;

#include "pch.hpp"

export module threading;

namespace hasty {

    export template<is_fp_complex_tensor_type TT> 
    struct cuda_context_holder {
        
        using tensor_type_t = TT;

        TupleTraits<
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,0>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,1>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,2>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,3>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,4>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,5>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,6>>,
            std::unordered_map<std::string, cache_tensor<cuda_t,TT,7>>
        > tensors;

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
            for_sequence<decltype(tensors)::Size>([](auto i) {
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
            for_sequence<decltype(tensors)::Size>([](auto i) {
                tensor_names[i].clear();
                std::get<i>(tensors).clear();
            });
        }

        std::set<std::string> names() {
            std::set<std::string> names;
            for_sequence<decltype(tensors)::Size>([&](auto i) {
                for (const auto& name : tensor_names[i]) {
                    names.insert(name);
                }
            });
            return names;
        }

    };

    export struct cuda_context {

        TupleTraits<
            cuda_context_holder<f32_t>,
            cuda_context_holder<f64_t>,
            cuda_context_holder<c64_t>,
            cuda_context_holder<c128_t>,
            cuda_context_holder<i16_t>,
            cuda_context_holder<i32_t>,
            cuda_context_holder<i64_t>,
            cuda_context_holder<b8_t>
        > holders;
        
        template<is_tensor_type TT>
        constexpr cuda_context_holder<TT>& holder() {
            cuda_context_holder<TT>& holder;
            using H = decltype(holder);
            for_sequence<H::Size>([](auto i) {
                if constexpr(std::is_same_v<H::Nth<i>::tensor_type_t,TT>) {
                    holder = std::get<i>(holders);
                }
            });
            return holder;
        }

        template<is_tensor_type TT, size_t RANK>
        void add(const std::string& name, cache_tensor<cuda_t,TT,RANK>&& tensor) {
            holder<TT>().add(name, std::move(tensor));
        }

        template<is_tensor_type TT, size_t RANK>
        cache_tensor<cuda_t,TT,RANK>& get(const std::string& name) {
            return holder<TT>().get<RANK>(name);
        }

        template<is_tensor_type TT, size_t RANK>
        void remove(const std::string& name) {
            holder<TT>().remove<RANK>(name);
        }

        template<is_tensor_type TT>
        void remove(const std::string& name) {
            holder<TT>().remove(name);
        }

        void remove(std::string& name) {
            using H = decltype(holder);
            for_sequence<H::Size>([&](auto i) {
                std::get<i>(holders).remove(name);
            });
        }

        template<is_tensor_type TT, size_t RANK>
        void clear() {
            holder<TT>().clear<RANK>();
        }

        void template<is_tensor_type TT>
        void clear() {
            holder<TT>().clear();
        }

        void clear() {
            using H = decltype(holder);
            for_sequence<H::Size>([&](auto i) {
                std::get<i>(holders).clear();
            });
        }

        std::set<std::string> names() {
            std::set<std::string> names;
            using H = decltype(holder);
            for_sequence<H::Size>([&](auto i) {
                auto innernames = std::get<i>(holders).names();
                for (const auto& name : innernames) {
                    names.insert(name);
                }
            });
            return names;
        }

    };

    export template<size_t NTHREADS>
    class cuda_thread_pool {
    public:

        device_thread_pool(std::array<cuda_context, NTHREADS>&& contexts)
            : _stop(false), _work_length(0), _contexts(contexts)
        {
            try {
                for (size_t i = 0; i < NTHREADS; i++) {
                    _threads[i] = std::thread([this, i]() { work(i); });
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

        ~device_thread_pool() {
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
        std::future<typename std::invoke_result<F, T&, Args...>::type> enqueue(F&& f, std::set<std::string> names, Args&&... args) {
            using return_type = typename std::invoke_result<F, T&, Args...>::type;

            auto task = std::make_shared<std::packaged_task<return_type(cuda_context&)>>(
                std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...)
            );

            std::future<return_type> res = task->get_future();
            std::function<void(cuda_context&)> task = [task](cuda_context& context) { (*task)(context); };
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

            size_t sleeps_since_work = 0;

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
                        for (auto& workit : _work) {
                            for (const auto& element : workit.second) {
                                if (context_names.count(element) > 0) {
                                    task = workit.first;
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
                                task = _work.pop_front().first;
                            }
                        }
                    } else {
                        task = _work.pop_front().first;
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
        std::array<cuda_context, NTHREADS> _contexts;
        std::array<std::thread, NTHREADS> _threads;

        std::condition_variable _cv;
        std::mutex _mutex;

        std::list<std::pair<std::function<void(cuda_context&)>, std::set<std::string>>> _work;

        bool _stop;
        std::atomic<int> _work_length;

    }


}
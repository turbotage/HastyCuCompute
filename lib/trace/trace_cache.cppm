module;

#include "pch.hpp"

export module trace_cache;

import util;

export import trace;

namespace hasty {
    namespace trace {

        
        export template<typename T>
        concept is_module_settings = requires(const T& t1, const T& t2) {
            { t.name() } -> std::convertible_to<std::string>;
            { t.to_string() } -> std::convertible_to<std::string>;
            { t1 == t2 } -> std::convertible_to<bool>;
            { std::hash<T>{}(t1) } -> std::convertible_to<std::size_t>;
        };
        
        export class trace_cache {
        public:

            trace_cache() = default;
            
            template<is_module_settings Settings>
            void cache_module(Settings&& settings, std::shared_ptr<CompilationModule> module) {
                std::lock_guard<std::mutex> lock(_cache_mutex);
                if (_module_cache.contains(settings)) {
                    _module_cache[settings] = std::move(module);
                } else {
                    _module_cache.insert({std::forward<Settings>(settings), std::move(module)});
                }
            }
            
            template<is_module_settings Settings>
            bool contains(const Settings& settings) const {
                std::lock_guard<std::mutex> lock(_cache_mutex);
                return _module_cache.contains(settings);
            }
            
            template<is_module_settings Settings>
            std::shared_ptr<CompilationModule> get_cached_module(const Settings& settings) {
                std::lock_guard<std::mutex> lock(_cache_mutex);
                if (_module_cache.contains(settings)) {
                    return _module_cache[settings];
                }
                return nullptr;
            }
            
            void save_modules() const {
                if (!std::filesystem::exists(module_cache_dir)) {
                    std::filesystem::create_directories(module_cache_dir);
                }
                
                {
                    std::lock_guard<std::mutex> lock(_cache_mutex);
                    for (const auto& [settings, module] : _module_cache) {
                        auto modpath = module_cache_dir / std::format("mod_{}.pt", settings.to_string());
                        module->save(modpath.string());
                    }
                }
            }
            
            template<is_module_settings Settings, is_trace_function Func>
            void load_module(const Settings& settings) {
                if (!std::filesystem::exists(module_cache_dir)) {
                    return;
                }
                
                auto modpath = module_cache_dir / std::format("mod_{}.pt", settings.to_string());
                if (!std::filesystem::exists(modpath)) {
                    return;
                }
                
                auto module = std::make_unique<CompilationModule>(torch::jit::load(modpath.string()));
                auto trace_func_ptr = std::make_unique<Func>(settings.name(), std::move(module));

                cache_module(settings, std::move(module));
            }
            
        private:
            struct Storage {
                std::type_info info;
                std::unique_ptr<void> data;
            };
            std::mutex _cache_mutex;
            std::unordered_map<Settings, Storage> _module_cache;
        };
        
        
    }
        
}
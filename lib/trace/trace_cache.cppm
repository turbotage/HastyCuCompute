module;

#include "pch.hpp"
#include <memory>

export module trace_cache;

import util;

export import trace;

namespace hasty {
	namespace trace {
		
		export extern std::filesystem::path module_cache_dir;

		export class trace_cache {
		public:

			trace_cache() = default;

			template<is_stringable Settings, is_trace_function Func>
			void cache_module(const Settings& settings, std::shared_ptr<Func> func) {
				std::lock_guard<std::mutex> lock(_cache_mutex);

				Key key{settings.to_string(), typeid(Func)};
				Storage storage;
				storage.trace_function = std::move(func);

				auto [it, inserted] = _module_cache.insert_or_assign(key, std::move(storage));

				Storage& stored_value = it->second;
				stored_value.module_getter = [&trace_func = stored_value.trace_function]<typename F = Func>() -> CompilationModule {
					return std::static_pointer_cast<F>(trace_func)->get_module();
				};
			}

			template<is_stringable Settings, is_trace_function Func>
			auto get_cached_trace_function(const Settings& settings) const -> trace_function<typename Func::ReturnTraits, typename Func::InputTraits> 
			{
				std::lock_guard<std::mutex> lock(_cache_mutex);

				Key key{settings.to_string(), typeid(Func)};
				if (_module_cache.contains(key)) {
					auto& storage = _module_cache.at(key);
					return std::static_pointer_cast<Func>(storage.trace_function);
				}
				throw std::runtime_error("No cached module for given settings");
			}

			template<is_stringable Settings, is_trace_function Func>
			auto get_cached_runnable_trace_function(const Settings& settings) const -> runnable_trace_function<typename Func::ReturnTraits, typename Func::InputTraits> 
			{
				std::lock_guard<std::mutex> lock(_cache_mutex);

				Key key{settings.to_string(), typeid(Func)};
				if (_module_cache.contains(key)) {
					auto& storage = _module_cache.at(key);
					return std::static_pointer_cast<Func>(storage.trace_function)->get_runnable();
				}
				throw std::runtime_error("No cached module for given settings");
			}

			template<is_stringable Settings, is_trace_function Func>
			bool contains_cached(const Settings& settings) const {
				std::lock_guard<std::mutex> lock(_cache_mutex);
				Key key{settings.to_string(), typeid(Func)};
				return _module_cache.contains(key);
			}

			template<is_stringable Settings, is_trace_function Func>
			bool contains_file(const Settings& settings) const {
				auto modpath = module_cache_dir / std::format("mod_{}_type_{}.txt", 
														settings.to_string(), 
														typeid(Func).name());
				if (!std::filesystem::exists(modpath)) {
					return false;
				}

				return true;
			}

			template<is_stringable Settings, is_trace_function Func>
			void load_module(const Settings& settings) {
				auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt", 
													settings.to_string(), 
													typeid(Func).name());
				if (!std::filesystem::exists(modpath)) {
					return;
				}
				
				auto module = std::make_unique<CompilationModule>(torch::load(modpath.string()));

				auto trace_func_ptr = std::make_shared<Func>(settings.name(), std::move(module));

				cache_module(settings, std::move(trace_func_ptr));
			}

			template<is_stringable Settings, is_trace_function Func>
			void save_module(const Settings& settings) const {
				if (!std::filesystem::exists(module_cache_dir)) {
					std::filesystem::create_directories(module_cache_dir);
				}
				auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt", 
													settings.to_string(), 
													typeid(Func).name());

				Settings key{settings.to_string(), typeid(Func)};
				{
					std::lock_guard<std::mutex> lock(_cache_mutex);
					auto it = _module_cache.find(key);
					if (it != _module_cache.end()) {
						auto& storage = it->second;
						storage.module_getter().save(modpath.string());
					} else {
						throw std::runtime_error("No cached module for given settings");
					}
				}
			}
			

			void save_modules() const {
			
				if (!std::filesystem::exists(module_cache_dir)) {
					std::filesystem::create_directories(module_cache_dir);
				}
				{
					std::lock_guard<std::mutex> lock(_cache_mutex);
					for (const auto& [key, storage] : _module_cache) {
						auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt", 
															key.settings_string, 
															key.info.name());
						storage.module_getter().save(modpath.string());
					}
				}
			}

		private:
			struct Key {
				std::string settings_string;
				std::type_info info;
			};

			struct KeyHash {
				std::size_t operator()(const Key& key) const {
					std::size_t h1 = std::hash<std::string>{}(key.settings_string);
					std::size_t h2 = key.info.hash_code();
					return h1 ^ (h2 << 1); // Combine the hashes
				}
			};

			struct KeyEqual {
				bool operator()(const Key& lhs, const Key& rhs) const {
					return (lhs.settings_string == rhs.settings_string) && (lhs.info == rhs.info);
				}
			};

			struct Storage {
				std::shared_ptr<void> trace_function;
				std::function<CompilationModule()> module_getter;
			};
			mutable std::mutex _cache_mutex;
			std::unordered_map<Key, Storage, KeyHash, KeyEqual> _module_cache;
		};
		
		export extern trace_cache global_trace_cache;

		export trace_cache global_trace_cache;

	}
		
}
module;

#include <pch.hpp>

export module script_cache;

import torch_base;
import util;
export import script;

namespace hasty {
namespace script {

export extern std::filesystem::path module_cache_dir;

export template<typename T>
concept is_sc_settings = requires(T t) {
    { t.to_string() } -> std::convertible_to<std::string_view>;
    { t.name() } -> std::convertible_to<std::string_view>;
};

export class script_cache {
public:

    script_cache() = default;

    template<is_runnable_script Script, is_sc_settings Settings>
    void cache(const Settings& settings, const Script& script) {
        std::lock_guard<std::mutex> lock(_cache_mutex);

        Key key{std::string(settings.to_string()), typeid(Script)};
        Storage storage;
        storage.module = std::make_unique<Module>(script.get_module().clone());

        _module_cache.insert_or_assign(key, std::move(storage));
    }

    template<is_runnable_script Script, is_sc_settings Settings>
    auto get_cached(const Settings& settings) const -> Script
    {
        std::lock_guard<std::mutex> lock(_cache_mutex);

        Key key{std::string(settings.to_string()), typeid(Script)};
        auto it = _module_cache.find(key);
        if (it != _module_cache.end()) {
            const auto& storage = it->second;
            return Script(
                settings.name(),
                std::make_unique<Module>(storage.module->clone())
            );
        }
        throw std::runtime_error("No cached module for given settings");
    }

    template<is_runnable_script Script, is_sc_settings Settings>
    bool contains_cached(const Settings& settings) const {
        std::lock_guard<std::mutex> lock(_cache_mutex);
        Key key{std::string(settings.to_string()), typeid(Script)};
        return _module_cache.contains(key);
    }

    template<is_runnable_script Script, is_sc_settings Settings>
    bool contains_file(const Settings& settings) const {
        auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt", 
                                                settings.to_string(), 
                                                typeid(Script).name());
        if (!std::filesystem::exists(modpath)) {
            return false;
        }
        return true;
    }

    template<is_runnable_script Script, is_sc_settings Settings>
    void load(const Settings& settings) {
        auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt",
                                            settings.to_string(),
                                            typeid(Script).name());
        if (!std::filesystem::exists(modpath)) {
            return;
        }

        Module m_module = htorch::jit::load(modpath.string());
        auto module_ptr = std::make_unique<Module>(std::move(m_module));

        auto script_ptr = Script(
                            settings.name(),
                            std::move(module_ptr)
        );

        cache(settings, std::move(script_ptr));
    }

    template<is_runnable_script Script, is_sc_settings Settings>
    void save(const Settings& settings) const {
        if (!std::filesystem::exists(module_cache_dir)) {
            std::filesystem::create_directories(module_cache_dir);
        }
        auto modpath = module_cache_dir / std::format("mod_{}_type_{}.pt",
                                            settings.to_string(),
                                            typeid(Script).name());

        Key key{std::string(settings.to_string()), typeid(Script)};
        {
            std::lock_guard<std::mutex> lock(_cache_mutex);
            auto it = _module_cache.find(key);
            if (it != _module_cache.end()) {
                auto& storage = it->second;
                storage.module->save(modpath.string());
            } else {
                throw std::runtime_error("No cached module for given settings");
            }
        }
    }

private:

    struct Key {
        std::string settings_str;
        const std::type_info& type;

        bool operator==(const Key& other) const {
            return settings_str == other.settings_str && type == other.type;
        }
    };

    struct KeyHasher {
        std::size_t operator()(const Key& k) const {
            return std::hash<std::string>()(k.settings_str) ^ std::hash<std::string>()(k.type.name());
        }
    };

    struct Storage {
        uptr<Module> module;
    };

    mutable std::mutex _cache_mutex;
    std::unordered_map<Key, Storage, KeyHasher> _module_cache;
};



}
}
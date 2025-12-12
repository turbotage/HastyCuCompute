module;

export module interface:cache;

import std;

namespace hasty {

export class InterfaceObject {
public:
	
	template<typename T>
	T& get() const {
		if (_type_index != std::type_index(typeid(T))) {
			throw std::runtime_error("InterfaceObject: bad cast in get");
		}
		return *std::static_pointer_cast<T>(_ptr);
	}

	template<typename T>
	std::shared_ptr<T> get_ptr() const {
		if (_type_index != std::type_index(typeid(T))) {
			throw std::runtime_error("InterfaceObject: bad cast in get_ptr");
		}
		return std::static_pointer_cast<T>(_ptr);
	}

	std::type_index type_info() const {
		return _type_index;
	}

	std::size_t id() const { return _id; }

private:

	template<typename T>
	InterfaceObject(std::shared_ptr<T> ptr,
					std::size_t id)
		:  _ptr(std::static_pointer_cast<void>(ptr)), _type_index(typeid(T)), _id(id)
	{
	}

	std::shared_ptr<void> _ptr;
	std::type_index _type_index;
	std::size_t _id;
};

export class InterfaceObjectCache {
public:
	InterfaceObjectCache() = default;

	template<typename T>
	InterfaceObject& add_object(std::shared_ptr<T> obj) {
		std::lock_guard<std::mutex> lock(_mutex);

		std::size_t id = _id_manager.get_new_id();

		auto interface_obj = std::make_shared<InterfaceObject>(
			obj,
			id
		);

		_cache.emplace(id, interface_obj);
		return interface_obj;
	}

	InterfaceObject& get_object(std::size_t id, std::type_index type_index) {
		auto it = _cache.find(id);
		if (it == _cache.end()) {
			throw std::runtime_error("InterfaceObjectCache: Object ID not found");
		}
		if (it->second->type_info() != type_index) {
			throw std::runtime_error("InterfaceObjectCache: Type mismatch in get_object");
		}
		return *(it->second);
	}

	void remove_object(std::size_t id, std::type_index type_index) {
		std::lock_guard<std::mutex> lock(_mutex);
		auto it = _cache.find(id);
		if (it != _cache.end()) {
			if (it->second->type_info() == type_index) {
				_cache.erase(it);
				_id_manager.release_id(id);
			}
		}
	}

private:
	std::mutex _mutex;
	std::unordered_map<std::size_t, std::unique_ptr<InterfaceObject>> _cache;

	struct IdManager {
		std::size_t get_new_id() {
			if (!free_ids.empty()) {
				std::size_t id = free_ids.back();
				free_ids.pop_back();
				return id;
			}
			return next_id++;
		}

		void release_id(std::size_t id) {
			free_ids.push_back(id);
		}

		std::vector<std::size_t> free_ids;
		std::size_t next_id = 0;
	};

	IdManager _id_manager;
};


}

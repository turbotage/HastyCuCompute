module;

#include "pch.hpp"
#include <cuda_runtime.h>

export module util;

//import pch;
export import std;

export import :concepts;
export import :containers;
export import :funcs;
export import :idx;
export import :meta;
export import :span;
export import :torch;
export import :typing;
export import :io;

namespace debug {

export void print_memory_usage(const std::string& prepend = "") {
	std::ifstream file("/proc/self/status");
	std::string line;
	while (std::getline(file, line)) {
		if (line.substr(0, 6) == "VmRSS:") {
			std::cout << prepend << " Resident Set Size: " << line.substr(6) << std::endl;
		} else if (line.substr(0, 6) == "VmSize:") {
			std::cout << prepend << " Virtual Memory Size: " << line.substr(6) << std::endl;
		}
	}
}

}

namespace hasty {

export template<typename T>
class move {
public:

	explicit move(T&& obj) : _obj(std::move(obj)) {}

	// Deleted copy constructor and copy assignment operator
	move(const move&) = delete;
	move& operator=(const move&) = delete;

	// Deleted move constructor and move assignment operator
	move(move&&) = delete;
	move& operator=(move&&) = delete;

	// Access the underlying object
	T& get() { return _obj; }
	const T& get() const { return _obj; }

private:
	T&& _obj;
};

export void synchronize() {
	htorch::cuda::synchronize();
}

export void synchronize(device_idx idx) {
	htorch::cuda::synchronize(i32(idx));
}

namespace util {

export template<typename T>
T future_catcher(std::future<T>& fut)
{
	try {
		return fut.get();
	}
	catch (hc10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}
	
export template<typename T>
T future_catcher(const std::function<T()>& func)
{
	try {
		return func();
	}
	catch (hc10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}
	
export void future_catcher(std::future<void>& fut)
{
	try {
		fut.get();
	}
	catch (hc10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}
	
export void future_catcher(const std::function<void()>& func)
{
	try {
		func();
	}
	catch (hc10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}

export std::vector<device_idx> get_cuda_devices(std::function<bool(const cudaDeviceProp&)> device_selector = 
	[](const cudaDeviceProp& prop)
	{ 
		return true; 
	}
) 
{
	i32 device_count = hat::cuda::device_count();
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

export inline void print_cuda_memory(device_idx idx, const std::string& prepend = "", bool empty_cache = false) {
	hasty::synchronize(idx);
	auto devicestats = hat::cuda::CUDACachingAllocator::getDeviceStats((int)idx);

	auto mb = [](size_t b){ return b / (1024.0 * 1024.0); };

	std::println("{}, device: {}", prepend, (int)idx);

	std::println("AGGREGATE: allocated: {}, reserved: {}",
			mb(devicestats.allocated_bytes[0].current),
			mb(devicestats.reserved_bytes[0].current)
	);

	std::println("SMALL_POOL: allocated: {}, reserved: {}",
			mb(devicestats.allocated_bytes[1].current),
			mb(devicestats.reserved_bytes[1].current)
	);

	std::println("LARGE_POOL: allocated: {}, reserved: {} \n",
			mb(devicestats.allocated_bytes[2].current),
			mb(devicestats.reserved_bytes[2].current)
	);

	if (empty_cache) {
		hat::cuda::CUDACachingAllocator::emptyCache();
	}
}

}
}
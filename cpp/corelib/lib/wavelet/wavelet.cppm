module;

export module wavelet;

import tensor;
import script_cache;

namespace hasty {

std::string get_wavelet_module_path(const std::string& wavelet, int dim, bool dec = true) {
	auto wavelet_dir = get_data_path() / "wavelets";

	std::string dec_dim_str = std::to_string(dim) + (dec ? "dec" : "rec");
	std::string filename;

	if (wavelet == "bior2.2") {
		filename = "wave"+dec_dim_str+"_bior2.2.pt";
	} else if (wavelet == "coif1") {
		filename = "wave"+dec_dim_str+"_coif1.pt";
	} else if (wavelet == "coif2") {
		filename = "wave"+dec_dim_str+"_coif2.pt";
	} else if (wavelet == "coif3") {
		filename = "wave"+dec_dim_str+"_coif3.pt";
	} else if (wavelet == "db2") {
		filename = "wave"+dec_dim_str+"_db2.pt";
	} else if (wavelet == "db4") {
		filename = "wave"+dec_dim_str+"_db4.pt";
	} else if (wavelet == "db6") {
		filename = "wave"+dec_dim_str+"_db6.pt";
	} else if (wavelet == "db8") {
		filename = "wave"+dec_dim_str+"_db8.pt";
	} else if (wavelet == "haar") {
		filename = "wave"+dec_dim_str+"_haar.pt";
	} else if (wavelet == "sym2") {
		filename = "wave"+dec_dim_str+"_sym2.pt";
	} else if (wavelet == "sym4") {
		filename = "wave"+dec_dim_str+"_sym4.pt";
	} else if (wavelet == "sym8") {
		filename = "wave"+dec_dim_str+"_sym8.pt";
	} else {
		throw std::runtime_error("Unsupported wavelet type: " + wavelet);
	}

	return (wavelet_dir / filename).string();
}

template<is_device D>
hc10::Device check_device(device_idx idx) {
	c10::Device device;
	if (std::is_same_v<D,cuda_t>) {
		if (idx == device_idx::CPU) {
			throw std::runtime_error("cuda_t device type cannot have CPU device_idx");
		}
		device = c10::Device(c10::kCUDA, (int)idx);
	} else {
		if (idx != device_idx::CPU) {
			throw std::runtime_error("cpu_t device type can only have CPU device_idx");
		}
		device = c10::Device(c10::kCPU, 0);
	}
	return device;
}

struct WaveletSettings {
	std::string wavelet_name;
	WaveletSettings(const std::string& name) : wavelet_name(name) {}
	auto to_string() const { return wavelet_name; }
	auto name() const { return wavelet_name; }
};

export template<is_device D, is_fp_tensor_type TT>
class WaveDec1D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:

	using RET_TYPE = std::tuple<tensor<D,TT,2>, std::vector<tensor<D,TT,2>>>;
	using INPUT_TYPE = tensor<D,TT,2>;

	WaveDec1D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("wavedec1d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 1, true);
			
			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveDec1D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,2>&& x) -> RET_TYPE
	{
		return _wavelet_script->run(std::move(x));
	}


private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};

export template<is_device D, is_fp_tensor_type TT>
class WaveDec2D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:
	using RET_TYPE = std::tuple<tensor<D,TT,3>, std::vector<tensor<D,TT,3>>>;
	using INPUT_TYPE = tensor<D,TT,3>;

	WaveDec2D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("wavedec2d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 2, true);

			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveDec2D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,3>&& x) -> RET_TYPE
	{
		return _wavelet_script->run(std::move(x));
	}

private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};

template<is_device D, is_fp_tensor_type TT>
class WaveDec3D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:
	using RET_TYPE = std::tuple<tensor<D,TT,4>, std::vector<tensor<D,TT,4>>>;
	using INPUT_TYPE = tensor<D,TT,4>;

	WaveDec3D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("wavedec3d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 3, true);
			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveDec3D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,4>&& x) -> RET_TYPE
	{
		return _wavelet_script->run(std::move(x));
	}

private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};


export template<is_device D, is_fp_tensor_type TT>
class WaveRec1D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:
	using RET_TYPE = tensor<D,TT,2>;
	using INPUT_TYPE = std::tuple<tensor<D,TT,2>, std::vector<tensor<D,TT,2>>>;

	WaveRec1D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("waverec1d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 1, false);

			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveRec1D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,2>&& coeffs, std::vector<tensor<D,TT,2>>&& details) -> RET_TYPE
	{
		return _wavelet_script->run(std::make_tuple(std::move(coeffs), std::move(details)));
	}

private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};

export template<is_device D, is_fp_tensor_type TT>
class WaveRec2D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:
	using RET_TYPE = tensor<D,TT,3>;
	using INPUT_TYPE = std::tuple<tensor<D,TT,3>, std::vector<tensor<D,TT,3>>>;

	WaveRec2D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("waverec2d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 2, false);

			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveRec2D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,3>&& coeffs, std::vector<tensor<D,TT,3>>&& details) -> RET_TYPE
	{
		return _wavelet_script->run(std::make_tuple(std::move(coeffs), std::move(details)));
	}

private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};

export template<is_device D, is_fp_tensor_type TT>
class WaveRec3D {
private:
	using RUNNABLE_TYPE = runnable_script<RET_TYPE,INPUT_TYPE>;
public:
	using RET_TYPE = tensor<D,TT,4>;
	using INPUT_TYPE = std::tuple<tensor<D,TT,4>, std::vector<tensor<D,TT,4>>>;

	WaveRec3D(std::string wavelet, int level, std::string mode, device_idx idx)
		: wavelet_(std::move(wavelet)), level_(level), mode_(std::move(mode))
	{
		WaveletSettings settings("waverec3d_"+_wavelet+"_level"+std::to_string(_level)+"_mode"+_mode);

		if (global_trace_cache.contains_cached<RUNNABLE_TYPE>(settings)) {
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				global_trace_cache.get_cached<RUNNABLE_TYPE>(settings)
			);
			_wavelet_script->to(idx);
		} else {
			auto device = check_device<D>(idx);
			auto module_path = get_wavelet_module_path(_wavelet, 3, false);
			
			auto module_ptr = std::make_unique<Module>(htorch::jit::load(module_path, device));
			_wavelet_script = std::make_unique<RUNNABLE_TYPE>(
				"WaveRec3D",
				std::move(module_ptr)
			);
			global_trace_cache.cache<RUNNABLE_TYPE>(settings, *_wavelet_script);
		}
	}

	auto operator()(tensor<D,TT,4>&& coeffs, std::vector<tensor<D,TT,4>>&& details) -> RET_TYPE
	{
		return _wavelet_script->run(std::make_tuple(std::move(coeffs), std::move(details)));
	}

private:
	std::string _wavelet;
	int _level;
	std::string _mode;
	uptr<RUNNABLE_TYPE> _wavelet_script;
};


} // namespace hasty

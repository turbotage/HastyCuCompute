module;

#include "pch.hpp"

export module mri:normal;

//import pch;
export import :trajectory;

import op;
import util;
import tensor;
import trace;
import trace_cache;
import nufft;
import threading;

namespace hasty {
	
	/**
	@brief	
	Performs the operation:
	\f[Bx\f] where \f[B = \sum_s \Psi_s^H F^H T F \Psi_s\f]
	
	@param kernel toeplitz kernel \f[T \f]
	@param stacked_diags stacked diagonals \f[\{\Psi_s\}\f].

	Example:
	The normal operator for a common sense operator can be written like this
	
	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_toeplitz_type1_operator {
	public:
		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		normal_toeplitz_type1_operator(
			cache_tensor<TT,DIM>&& kernel, 
			cache_tensor<TT,DIM+1>&& stacked_diags, 
			i32 fft_batch_size = 4, 
			bool store_module = false)
			: _kernel(std::move(kernel)), 
			_stacked_diags(std::move(stacked_diags)),
			_fft_batch_size(fft_batch_size), 
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& x)
		{
			auto didx = x.get_device_idx();

			constexpr size_t kernelsize = decltype(_kernel.template operator[]<D>(didx, 0, Slice{}))::size();
			static_assert(
					kernelsize == DIM, "Kernel was wrong size"
			);
			constexpr size_t stacked_diagsize = decltype(_stacked_diags.template get<D>(didx))::size();
			static_assert(
				stacked_diagsize == DIM+1, "Stacked diag was wrong size"
			);

			return std::get<0>(_runner.run(
				x,
				_kernel.template get<D>(didx),
				_stacked_diags.template get<D>(didx)
			));

		}

	protected:
		using INPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						STACKED_DIAG_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						STACKED_DIAG_PROTO_T>
		>;

		using THIS_TYPE_T = normal_toeplitz_type1_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NT_T1_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}
		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T
		{
			INPUT_PROTO_T			 input("input");
			KERNEL_PROTO_T			 kernel("kernel");
			STACKED_DIAG_PROTO_T	 stacked_diag("stacked_diag");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
				"normal_toeplitz_type1_operator",
				input, kernel, stacked_diag
			);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, input, kernel, stacked_diag):
	spatial_shp = input.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	ncoil = stacked_diag.shape[0]
	nrun = ncoil // {0}
	
	out = torch.zeros_like(input)
	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * input
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		d *= dmap.conj()
		out += torch.sum(d, 0)

	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM> _kernel;
		cache_tensor<TT,DIM+1> _stacked_diags;
		i32 _fft_batch_size;
		bool _store_module;

		RUNNABLE_TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NT_T1_OP = normal_toeplitz_type1_operator<D,TT,DIM>;

	/**
	@brief	
	Performs the operation:
	\f[B\Psi \f] where \f[(B\Psi)_i = X^H F^H T F X \Psi_i\f]
	
	@param kernel toeplitz kernel \f[T \f]
	@param stacked_diags stacked diagonals \f[\{\Psi_s\}\f].

	Example:
	The normal operator for a common sense operator can be written like this
	
	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_toeplitz_type2_operator {
	public:
		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		normal_toeplitz_type2_operator(
			cache_tensor<TT,DIM>&& kernel, 
			cache_tensor<TT,DIM>&& x, 
			i32 fft_batch_size = 4, 
			bool store_module = false)
			: _kernel(std::move(kernel)), 
			_x(std::move(x)),
			_fft_batch_size(fft_batch_size), 
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM+1>&& stacked_diags)
		{
			auto didx = stacked_diags.get_device_idx();

			return std::get<0>(_runner.run(
				stacked_diags,
				_kernel.template get<D>(didx),
				_x.template get<D>(didx)
			));

		}

	protected:
		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using X_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						X_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						X_PROTO_T>
		>;

		using THIS_TYPE_T = normal_toeplitz_type2_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NT_T2_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}
		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T
		{
			X_PROTO_T			 	x("x");
			KERNEL_PROTO_T			 kernel("kernel");
			STACKED_DIAG_PROTO_T	 stacked_diag("stacked_diag");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
				"normal_toeplitz_type1_operator",
				stacked_diag, kernel, x
			);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, stacked_diag, kernel, x):
	spatial_shp = x.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	ncoil = stacked_diag.shape[0]
	nrun = ncoil // {0}
	
	out = torch.empty_like(stacked_diag)
	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * x
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		out[bst:(bst+{0})] = d

	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	out *= x.conj()
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM> _kernel;
		cache_tensor<TT,DIM> _x;
		i32 _fft_batch_size;
		bool _store_module;

		RUNNABLE_TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NT_T2_OP = normal_toeplitz_type2_operator<D,TT,DIM>;





	/**
	@brief	
	Performs the operation:
	\f[Bx\f] where \f[B = \sum_l R_l^H \left[ \sum_s \Psi_s^H F^H T_l F \Psi_s \right] R_l\f]
	
	@param kernels toeplitz kernels \f[T_l \f]
	@param kerneldiags  kernel diagonals \f[R_l \f]
	@param stacked_diags stacked diagonals \f[\{\Psi_s\}\f].

	Example:
	The normal operator for a common sense operator with off-resonance correction and diagonal phase modulation
	can be written as:
	\f[A^HA = D_\phi^H \sum_i S_i^H  \left[ \sum_l R_l^H F^H T_l F R_l \right] S_i D_\phi\f]
	where \f[D_\phi\f] is the diagonal phase modulation matrix, \f[S_i\f] are the coil sensitivity matrices, \f[R_l\f] come from 
	the off-resonance ratemaps and \f[T_l\f] are the toeplitz matrices. This matrix vector with \f[A^HA\f] can be computed by calling this class operator()
	with the pairs \f[<T_l, D_lD_\phi>\f] and the stacked diagonals \f[S = \{S_i\}\f].
	
	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_innerlooped_diagonal_toeplitz_type1_operator {
	public:

		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		/**
		@param kernels_kerneldiags vector of pairs of kernels and kernel diagonals \f[<T_l,D_l>\f]
		@param stacked_diags stacked diagonals \f[\{D_s\}\f].

		@tparam D Device type.
		@tparam TT Tensor type.
		@tparam DIM Dimension.
		*/
		normal_innerlooped_diagonal_toeplitz_type1_operator(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM+1>&& stacked_diags, 
			i32 fft_batch_size = 4,
			bool store_module = false)
			: 
			_kernels(std::move(kernels)), 
			_kerneldiags(std::move(kerneldiags)),
			_stacked_diags(std::move(stacked_diags)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(fft_batch_size, store_module)),
			_fft_batch_size(fft_batch_size),
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM> operator()(
			tensor<D,TT,DIM>&& x)
		{
			
			auto didx = x.get_device_idx();

			constexpr size_t kernelsize = decltype(_kernels.template operator[]<D>(didx, 0, Slice{}))::size();
			static_assert(
					kernelsize == DIM, "Kernel was wrong size"
			);
			constexpr size_t kerneldiagsize = decltype(_kerneldiags.template operator[]<D>(didx, 0, Slice{}))::size();
			static_assert(
				kerneldiagsize == DIM, "Kerneldiag was wrong size"
			);
			constexpr size_t stacked_diagsize = decltype(_stacked_diags.template get<D>(didx))::size();
			static_assert(
				stacked_diagsize == DIM+1, "Stacked diag was wrong size"
			);

			// The first term in the sum
			tensor<D,TT,DIM> out = std::get<0>(_runner.run(
				x, 
				_kernels.template operator[]<D>(didx, 0, Slice{}),
				_kerneldiags.template operator[]<D>(didx, 0, Slice{}),
				_stacked_diags.template get<D>(didx)
			));
			
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}

			// Loop over off kernels >= 1
			for (int i = 1; i < _kernels.template shape<0>(); ++i) {

				std::tuple<tensor<D,TT,DIM>> tensortup = _runner.run(
					x,
					_kernels.template operator[]<D>(didx, i, Slice{}),
					_kerneldiags.template operator[]<D>(didx, i, Slice{}),
					_stacked_diags.template get<D>(didx)
				);

				out += std::get<0>(tensortup);
			}

			return out;
		}

	protected:

		using INPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						STACKED_DIAG_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						STACKED_DIAG_PROTO_T>
		>;

		using THIS_TYPE_T = normal_innerlooped_diagonal_toeplitz_type1_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T 
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NIDT_T1_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}

		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T 
		{
			INPUT_PROTO_T             input("input");
			KERNEL_PROTO_T            kernel("kernel");
			DIAG_PROTO_T              diag("diag");
			STACKED_DIAG_PROTO_T      stacked_diag("stacked_diag");
			
			OUTPUT_PROTO_T            output("output");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
									"normal_innerlooped_diagonal_toeplitz_type1_operator", 
									input, kernel, diag, stacked_diag);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, input, kernel, diag, stacked_diag):
	spatial_shp = input.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	nstack = stacked_diag.shape[0]
	nrun = nstack // {0}

	out = torch.zeros_like(input)

	input = input * diag

	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * input
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		d *= dmap.conj()
		out += torch.sum(d, 0)

	out *= diag.conj()
	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM+1> _stacked_diags;
		i32 _fft_batch_size;
		bool _store_module;

		RUNNABLE_TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NIDT_T1_OP = normal_innerlooped_diagonal_toeplitz_type1_operator<D,TT,DIM>;

	/**
	@brief	
	Performs the operation:
	\f[B\Psi \f] where \f[(B\Psi)_i = X^H \left[ \sum_l R_l^H F^H T_l F R_l \right] X \Psi_i \f]
	
	@param kernels toeplitz kernels \f[T_l \f]
	@param kerneldiags  kernel diagonals \f[R_l \f]
	@param stacked_diags stacked diagonals \f[\{\Psi_s\}\f].

	Example:
	Same as the type1 operator but used for gradients with respect to the stacked diags.

	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_innerlooped_diagonal_toeplitz_type2_operator {
	public:

		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM+1> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM+1> output_rank_t = {};

		/**
		@param kernels_kerneldiags vector of pairs of kernels and kernel diagonals \f[<T_l,D_l>\f]
		@param stacked_diags stacked diagonals \f[\{D_s\}\f].

		@tparam D Device type.
		@tparam TT Tensor type.
		@tparam DIM Dimension.
		*/
		normal_innerlooped_diagonal_toeplitz_type2_operator(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM>&& x, 
			i32 fft_batch_size = 4,
			bool store_module = false)
			: 
			_kernels(std::move(kernels)), 
			_kerneldiags(std::move(kerneldiags)),
			_x(std::move(x)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(fft_batch_size, store_module)),
			_fft_batch_size(fft_batch_size),
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& stacked_diag)
		{
			auto didx = stacked_diag.get_device_idx();

			// The first term in the sum
			tensor<D,TT,DIM> out = std::get<0>(_runner.run(
				stacked_diag, 
				_kernels.template operator[]<D>(didx, 0, Slice{}),
				_x.template operator[]<D>(didx),
				_kerneldiags.template operator[]<D>(didx, 0, Slice{})
			));
			
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}

			// Loop over off kernels >= 1
			for (int i = 1; i < _kernels.template shape<0>(); ++i) {

				std::tuple<tensor<D,TT,DIM>> tensortup = _runner.run(
					stacked_diag,
					_kernels.template operator[]<D>(didx, i, Slice{}),
					_x.template get<D>(didx),
					_kerneldiags.template operator[]<D>(didx, i, Slice{})
				);

				out += std::get<0>(tensortup);
			}

			return out;
		}

	protected:

		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using X_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						X_PROTO_T,
						DIAG_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						X_PROTO_T,
						DIAG_PROTO_T>
		>;

		using THIS_TYPE_T = normal_innerlooped_diagonal_toeplitz_type2_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T 
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NIDT_T2_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}

		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T 
		{
			STACKED_DIAG_PROTO_T      stacked_diag("stacked_diag");
			KERNEL_PROTO_T            kernel("kernel");
			X_PROTO_T                 x("x");
			DIAG_PROTO_T              diag("diag");
			
			OUTPUT_PROTO_T            output("output");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
									"normal_innerlooped_diagonal_toeplitz_type2_operator", 
									stacked_diag, kernel, x, diag);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, stacked_diag, kernel, x, diag):
	spatial_shp = x.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	nstack = stacked_diag.shape[0]
	nrun = nstack // {0}

	out = torch.empty_like(stacked_diag)

	x = (x * diag).unsqueeze(0)

	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * x
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		out[bst:(bst+{0})] = d

	out *= x.conj()
	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM> _x;
		i32 _fft_batch_size;
		bool _store_module;

		RUNNABLE_TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NIDT_T2_OP = normal_innerlooped_diagonal_toeplitz_type2_operator<D,TT,DIM>;




	/**
	@brief
	Performs the operation:
	\f[Bx\f] where \f[B = \sum_{i,j} W_{ij} \Psi_i^H \left[ \sum_l R_l^H F^H T_l F R_l \right] \Psi_j\f]
	

	@param kernels toeplitz kernels \f[T_l \f]
	@param kerneldiags  kernel diagonals \f[R_l \f]
	@param stacked_diags stacked diagonals \f[\{\Psi_s\}\f].
	@param weights Weights to use when combining stacked diagonals \f[\{\Psi_s\}\f] \f[W_{ij}\f]

	Example:
	The normal operator for a common sense operator with off-resonance correction and diagonal phase modulation and noise de-whitening
	can be written as:
	\f[A^HA = D_\phi^H \left[ \sum_{i,j} W_{ij} S_i^H  \left[ \sum_l R_l^H F^H T_l F R_l \right] S_j \right] D_\phi\f]
	where \f[D_\phi\f] is the diagonal phase modulation matrix, \f[S_i\f] are the coil sensitivity matrices, \f[R_l\f] come from
	the off-resonance interpolators and \f[T_l\f] are the toeplitz matrices. \f[W\f] is the inverse of the covariance noise matrix,
	or if coil-compression has been made \f[W = (K^H\Sigma K)^{-1}\f] where \f[K\f] is
	the coil compression matrix and \f[\Sigma\f] is the noise covariance matrix. Then \f[S_i\f] are the virtual
	coil sensitivities.

	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_innerlooped_diagonal_toeplitz_weighted_type1_operator {
	public:
 
		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		/**
		@param kernels_kerneldiags vector of pairs of kernels and kernel diagonals \f[<T_l,D_l>\f]
		@param stacked_diags stacked diagonals \f[\{D_s\}\f].

		@tparam D Device type.
		@tparam TT Tensor type.
		@tparam DIM Dimension.
		*/
		normal_innerlooped_diagonal_toeplitz_weighted_type1_operator(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM+1>&& stacked_diags, 
			cache_tensor<TT,2>&& weights, 
			i32 fft_batch_size = 4,
			bool store_module = false
		)
			: 
			_kernels(std::move(kernels)),
			_kerneldiags(std::move(kerneldiags)),
			_stacked_diags(std::move(stacked_diags)),
			_weights(std::move(weights)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(fft_batch_size)),
			_fft_batch_size(fft_batch_size),
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& x)
		{
			
			auto didx = x.get_device_idx();

			// The first term in the sum
			tensor<D,TT,DIM> out = std::get<0>(_runner.run(
				x,
				_kernels.template operator[]<D>(didx, 0, Slice{}),
				_kerneldiags.template operator[]<D>(didx, 0, Slice{}),
				_stacked_diags.template get<D>(didx),
				_weights.template get<D>(didx)
			));
			
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}

			// Loop over off kernels >= 1
			for (int i = 1; i < _kernels.template shape<0>(); ++i) {
				std::tuple<tensor<D,TT,DIM>> tensortup = _runner.run(
					x, 
					_kernels.template operator[]<D>(didx, i, Slice{}),
					_kerneldiags.template operator[]<D>(didx, i, Slice{}),
					_stacked_diags.template get<D>(didx),
					_weights.template get<D>(didx)
				);

				out += std::get<0>(tensortup);
			}

			return out;
		}

	protected:

		using INPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;
		using WEIGHTS_PROTO_T = trace::tensor_prototype<D,TT,2>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						STACKED_DIAG_PROTO_T,
						WEIGHTS_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	INPUT_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						STACKED_DIAG_PROTO_T,
						WEIGHTS_PROTO_T>
		>;

		using THIS_TYPE_T = normal_innerlooped_diagonal_toeplitz_weighted_type1_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T 
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NIDTW_T1_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}

		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T {

			INPUT_PROTO_T             input("input");
			KERNEL_PROTO_T            kernel("kernel");
			DIAG_PROTO_T              diag("diag");
			STACKED_DIAG_PROTO_T      stacked_diag("stacked_diag");
			WEIGHTS_PROTO_T           weights("weights");

			
			OUTPUT_PROTO_T            output("output");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
									"normal_innerlooped_diagonal_toeplitz_weighted_type1_operator", 
									input, kernel, diag, stacked_diag, weights);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, input, kernel, diag, stacked_diag, weights):
	spatial_shp = input.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	nstack = stacked_diag.shape[0]
	nrun = nstack // {0}

	out = torch.zeros_like(input)

	input = input * diag

	stack_out = torch.empty_like(stacked_diag)

	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * input
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		stack_out[bst:(bst+{0})] = d

	for i in range(nstack):
		for j in range(nstack):
			out += weights[i,j] * stacked_diag[i].conj() * stack_out[j]	

	out *= diag.conj()
	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM+1> _stacked_diags;
		cache_tensor<TT,2> _weights;
		i32 _fft_batch_size;
		bool _store_module;

		TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NIDTW_T1_OP = normal_innerlooped_diagonal_toeplitz_weighted_type1_operator<D,TT,DIM>;


	/**
	@brief
	Performs the operation:
	\f[(B\Psi)_i = X^H \sum_{j} W_{ij} \left[ \sum_l R_l^H F^H T_l F R_l \right]X\Psi_j\f]
	

	@param kernels toeplitz kernels \f[T_l \f]
	@param kerneldiags  kernel diagonals \f[R_l \f]
	@param x \f[X \f].
	@param weights Weights to use when combining stacked diagonals \f[\{\Psi_s\}\f] \f[W_{ij}\f]

	Example:
	Similar to the same type1 operator but used for gradients with respected to the stacked diagonal instead.

	@tparam D Device type.
	@tparam TT Tensor type.
	@tparam DIM Dimension.
	*/
	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	class normal_innerlooped_diagonal_toeplitz_weighted_type2_operator {
	public:
 
		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM+1> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM+1> output_rank_t = {};

		/**
		@tparam D Device type.
		@tparam TT Tensor type.
		@tparam DIM Dimension.
		*/
		normal_innerlooped_diagonal_toeplitz_weighted_type2_operator(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM>&& x, 
			cache_tensor<TT,2>&& weights, 
			i32 fft_batch_size = 4,
			bool store_module = false
		)
			: 
			_kernels(std::move(kernels)),
			_kerneldiags(std::move(kerneldiags)),
			_x(std::move(x)),
			_weights(std::move(weights)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(fft_batch_size)),
			_fft_batch_size(fft_batch_size),
			_store_module(store_module)
		{
		}

		tensor<D,TT,DIM+1> operator()(tensor<D,TT,DIM+1>&& stacked_diags)
		{
			auto didx = stacked_diags.get_device_idx();

			// The first term in the sum
			tensor<D,TT,DIM> out = std::get<0>(_runner.run(
				stacked_diags,
				_kernels.template operator[]<D>(didx, 0, Slice{}),
				_kerneldiags.template operator[]<D>(didx, 0, Slice{}),
				_x.template get<D>(didx),
				_weights.template get<D>(didx)
			));
			
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}

			// Loop over off kernels >= 1
			for (int i = 1; i < _kernels.template shape<0>(); ++i) {
				std::tuple<tensor<D,TT,DIM>> tensortup = _runner.run(
					stacked_diags,
					_kernels.template operator[]<D>(didx, i, Slice{}),
					_kerneldiags.template operator[]<D>(didx, i, Slice{}),
					_x.template get<D>(didx),
					_weights.template get<D>(didx)
				);

				out += std::get<0>(tensortup);
			}

			return out;
		}

	protected:

		using STACKED_DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM+1>;
		using KERNEL_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using DIAG_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using X_PROTO_T = trace::tensor_prototype<D,TT,DIM>;
		using WEIGHTS_PROTO_T = trace::tensor_prototype<D,TT,2>;

		using OUTPUT_PROTO_T = trace::tensor_prototype<D,TT,DIM>;

		using TRACE_FUNC_T = trace::trace_function<
			std::tuple<OUTPUT_PROTO_T>, 
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						X_PROTO_T,
						WEIGHTS_PROTO_T>
		>;

		using RUNNABLE_TRACE_FUNC_T = trace::runnable_trace_function<
			std::tuple<OUTPUT_PROTO_T>,
			std::tuple<	STACKED_DIAG_PROTO_T,
						KERNEL_PROTO_T,
						DIAG_PROTO_T,
						X_PROTO_T,
						WEIGHTS_PROTO_T>
		>;

		using THIS_TYPE_T = normal_innerlooped_diagonal_toeplitz_weighted_type2_operator<D,TT,DIM>;

		static auto build_runner(i32 fft_batch_size, bool store_module) -> RUNNABLE_TRACE_FUNC_T 
		{
			struct Settings {
				i32 _fft_batch_size;
				auto to_string() { return std::format("NIDTW_T2_OP<{}>", _fft_batch_size); }
			} settings;

			if (trace::global_trace_cache.template contains_cached<TRACE_FUNC_T>(settings)) {
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else if (trace::global_trace_cache.template contains_file<TRACE_FUNC_T>(settings)) {
				trace::global_trace_cache.template load_module<TRACE_FUNC_T>(settings);
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			} else {
				auto trace_func = std::make_shared<TRACE_FUNC_T>(THIS_TYPE_T::build_trace_function(fft_batch_size));
				trace::global_trace_cache.cache_module(settings, std::move(trace_func));
				if (store_module) {
					trace::global_trace_cache.template save_module<TRACE_FUNC_T>(settings);
				}
				return trace::global_trace_cache.template get_cached_runnable_trace_function<TRACE_FUNC_T>(settings);
			}

		}

		static auto build_trace_function(i32 fft_batch_size) -> TRACE_FUNC_T {

			STACKED_DIAG_PROTO_T      stacked_diag("stacked_diag");
			KERNEL_PROTO_T            kernel("kernel");
			DIAG_PROTO_T              diag("diag");
			X_PROTO_T             	  x("x");
			WEIGHTS_PROTO_T           weights("weights");

			
			OUTPUT_PROTO_T            output("output");

			TRACE_FUNC_T ret = trace::trace_function_factory<OUTPUT_PROTO_T>::make(
									"normal_innerlooped_diagonal_toeplitz_weighted_type2_operator", 
									stacked_diag, kernel, diag, x, weights
								);

			ret.add_lines(std::format(R"ts(
FORWARD_ENTRYPOINT(self, stacked_diag, kernel, diag, x, weights):
	spatial_shp = x.shape #shp[1:]
	expanded_shp = [2*s for s in spatial_shp]
	transform_dims = [i+1 for i in range(len(spatial_shp))]

	nstack = stacked_diag.shape[0]
	nrun = nstack // {0}

	out = torch.zeros_like(stacked_diag)

	x = (x * diag).unsqueeze(0)

	stack_out = torch.empty_like(stacked_diag)

	for run in range(nrun):
		bst = run*{0}
		dmap = stacked_diag[bst:(bst+{0})]
		d = dmap * x
		d = torch.fft_fftn(d, expanded_shp, transform_dims)
		d *= kernel
		d = torch.fft_ifftn(d, None, transform_dims)

		for dim in range(len(spatial_shp)):
			d = torch.slice(d, dim+1, spatial_shp[dim]-1, -1)

		stack_out[bst:(bst+{0})] = d

	for i in range(nstack):
		for j in range(nstack):
			out[i] += weights[i,j] * stack_out[j]	

	out *= x.conj()
	out *= (1 / torch.prod(torch.tensor(spatial_shp)))
	
	return (out,)
)ts", fft_batch_size));

			ret.compile();

			return ret;
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM> _x;
		cache_tensor<TT,2> _weights;
		i32 _fft_batch_size;
		bool _store_module;

		TRACE_FUNC_T _runner;
	};

	export template<is_device D, is_fp_complex_tensor_type TT, size_t DIM>
	using NIDTW_T2_OP = normal_innerlooped_diagonal_toeplitz_weighted_type2_operator<D,TT,DIM>;



}
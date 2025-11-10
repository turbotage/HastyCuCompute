module;

#include "pch.hpp"

export module mri:normal;

//import pch;
export import :trajectory;

import op;
import util;
import tensor;
import script_cache;
import threading;
import fft;

namespace hasty {
	
	template<typename D, typename TT, size_t DIM>
	concept is_normal_op_compatible = 
		is_device<D> &&
		is_fp_complex_tensor_type<TT> &&
		std::is_same_v<TT, c64_t> 	&& 
		(DIM >= 2) && (DIM <= 3) 	&&
		std::is_same_v<D, cuda_t>;

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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_T_T1_OP {
	public:

		static constexpr std::string_view class_name = "NORMAL_T_T1_OP";

		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		NORMAL_T_T1_OP(
			cache_tensor<TT,DIM>&& kernel, 
			cache_tensor<TT,DIM+1>&& stacked_diags)
			: 
			_kernel(std::move(kernel)), 
			_stacked_diags(std::move(stacked_diags))
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& x)
		{
			auto didx = x.get_device_idx();
			auto stacked_diags = _stacked_diags.template get<D>(didx);
			auto out = make_empty_tensor_like(stacked_diags);
			
			optcrefw<tensor<D,TT,DIM>> x_opt = std::make_optional(std::cref(x));
			fft::toeplitz_multiplication(
				stacked_diags,
				out,
				_kernel.template operator[]<D>(didx, Ellipsis{}),
				std::nullopt,
				std::move(x_opt),
				std::nullopt,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzAccumulateType::NONE
			);

			return hasty::sum<0>(out);
		}

	private:
		cache_tensor<TT,DIM> _kernel;
		cache_tensor<TT,DIM+1> _stacked_diags;
	};

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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_T_T2_OP {
	public:
	
		static constexpr std::string_view class_name = "NORMAL_T_T2_OP";

		using device_type_t = D;
		using input_tensor_type_t = TT;
		using output_tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, DIM> input_rank_t = {};
		static constexpr std::integral_constant<size_t, DIM> output_rank_t = {};

		NORMAL_T_T2_OP(
			cache_tensor<TT,DIM>&& kernel, 
			cache_tensor<TT,DIM>&& x)
			: 
			_kernel(std::move(kernel)), 
			_x(std::move(x))
		{
		}

		tensor<D,TT,DIM+1> operator()(tensor<D,TT,DIM+1>&& stacked_diags)
		{
			auto didx = stacked_diags.get_device_idx();
			auto out = make_empty_tensor_like(stacked_diags);

			auto x = _x.template get<D>(didx);

			fft::toeplitz_multiplication(
				stacked_diags,
				out,
				_kernel.template operator[]<D>(didx, Ellipsis{}),
				std::nullopt,
				std::make_optional(std::cref(x)),
				std::nullopt,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzAccumulateType::NONE
			);
			return out;
		}

	private:
		cache_tensor<TT,DIM> _kernel;
		cache_tensor<TT,DIM> _x;
	};

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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_IDT_T1_OP {
	public:

		using THIS_TYPE_T = NORMAL_IDT_T1_OP<D,TT,DIM>;

		static constexpr std::string_view class_name = "NORMAL_IDT_T1_OP";

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
		NORMAL_IDT_T1_OP(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM+1>&& stacked_diags,
			bool store_module = false)
			: 
			_kernels(std::move(kernels)), 
			_kerneldiags(std::move(kerneldiags)),
			_stacked_diags(std::move(stacked_diags))
		{
		}

		tensor<D,TT,DIM> operator()(
			tensor<D,TT,DIM>&& x)
		{
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}

			auto didx = x.get_device_idx();

			auto stacked_diags = _stacked_diags.template get<D>(didx);
			auto out = make_empty_tensor_like(stacked_diags);

			auto kerneldiag = _kerneldiags.template operator[]<D>(didx, 0, Ellipsis{});
			fft::toeplitz_multiplication(
				stacked_diags,
				out,
				_kernels.template operator[]<D>(didx, 0, Ellipsis{}),
				std::nullopt,
				std::make_optional(std::cref(x)),
				std::make_optional(std::cref(kerneldiag)),
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzAccumulateType::NONE
			);

			for (int i = 1; i < _kernels.template shape<0>(); ++i) {

				kerneldiag = hasty::move(_kerneldiags.template operator[]<D>(didx, i, Ellipsis{}));
				fft::toeplitz_multiplication(
					stacked_diags,
					out,
					_kernels.template operator[]<D>(didx, i, Ellipsis{}),
					std::nullopt,
					std::make_optional(std::cref(x)),
					std::make_optional(std::cref(kerneldiag)),
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::NONE,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzAccumulateType::ACCUMULATE
				);
			}

			return hasty::sum<0>(out);
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM+1> _stacked_diags;
	};

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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_IDT_T2_OP {
	public:

		static constexpr std::string_view class_name = "NORMAL_IDT_T2_OP";

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
		NORMAL_IDT_T2_OP(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM>&& x)
			: 
			_kernels(std::move(kernels)), 
			_kerneldiags(std::move(kerneldiags)),
			_x(std::move(x))
		{
		}

		tensor<D,TT,DIM+1> operator()(tensor<D,TT,DIM+1>&& stacked_diag)
		{
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}
			auto didx = stacked_diag.get_device_idx();

			// The first term in the sum
			auto out = make_empty_tensor_like(stacked_diag);

			auto x = _x.template get<D>(didx);
			auto kerneldiag = _kerneldiags.template operator[]<D>(didx, 0, Ellipsis{});

			fft::toeplitz_multiplication(
				stacked_diag,
				out,
				_kernels.template operator[]<D>(didx, 0, Ellipsis{}),
				std::nullopt,
				std::make_optional(std::cref(x)),
				std::make_optional(std::cref(kerneldiag)),
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzAccumulateType::NONE
			);
			
			for (int i = 1; i < _kernels.template shape<0>(); ++i) {
				kerneldiag = hasty::move(_kerneldiags.template operator[]<D>(didx, i, Ellipsis{}));
				fft::toeplitz_multiplication(
					stacked_diag,
					out,
					_kernels.template operator[]<D>(didx, i, Ellipsis{}),
					std::nullopt,
					std::make_optional(std::cref(x)),
					std::make_optional(std::cref(kerneldiag)),
					fft::ToeplitzMultType::NONE,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzAccumulateType::ACCUMULATE
				);
			}

			return out;
		}

	private:
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM> _x;
	};


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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_IDTW_T1_OP {
	public:
 
		static constexpr std::string_view class_name = "NORMAL_IDTW_T1_OP";

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
		NORMAL_IDTW_T1_OP(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM+1>&& stacked_diags, 
			cache_tensor<TT,2>&& weights, 
			bool store_module = false
		)
			: 
			_store_module(store_module),
			_kernels(std::move(kernels)),
			_kerneldiags(std::move(kerneldiags)),
			_stacked_diags(std::move(stacked_diags)),
			_weights(std::move(weights)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(store_module))
		{
		}

		tensor<D,TT,DIM> operator()(tensor<D,TT,DIM>&& x)
		{
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}
			auto didx = x.get_device_idx();

			auto stacked_diags = _stacked_diags.template operator[]<D>(didx, Ellipsis{});
			auto out = make_empty_tensor_like(stacked_diags);

			auto kerneldiag = _kerneldiags.template operator[]<D>(didx, 0, Ellipsis{});

			fft::toeplitz_multiplication(
				stacked_diags,
				out,
				_kernels.template operator[]<D>(didx, 0, Ellipsis{}),
				std::nullopt,
				std::make_optional(std::cref(x)),
				std::make_optional(std::cref(kerneldiag)),
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzAccumulateType::NONE
			);

			for (int i = 1; i < _kernels.template shape<0>(); ++i) {
				kerneldiag = hasty::move(_kerneldiags.template operator[]<D>(didx, i, Ellipsis{}));
				fft::toeplitz_multiplication(
					stacked_diags,
					out,
					_kernels.template operator[]<D>(didx, i, Ellipsis{}),
					std::nullopt,
					std::make_optional(std::cref(x)),
					std::make_optional(std::cref(kerneldiag)),
					fft::ToeplitzMultType::NONE,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::NONE,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzAccumulateType::ACCUMULATE
				);
			}

			return _runner.run(
				std::move(out),
				stacked_diags,
				_weights.template get<D>(didx)
			);
		}

	protected:

		using X_T = tensor<D,TT,DIM>;
		using STACKED_DIAG_T = tensor<D,TT,DIM+1>;
		using WEIGHTS_T = tensor<D,TT,2>;

		using RUNNABLE_SCRIPT_T = script::runnable_script<
			X_T,
			STACKED_DIAG_T,
			STACKED_DIAG_T,
			WEIGHTS_T
		>;

		using THIS_TYPE_T = NORMAL_IDTW_T1_OP<D,TT,DIM>;


		static auto build_runner(bool store_module) -> RUNNABLE_SCRIPT_T 
		{
			struct Settings{
				auto to_string() const { return class_name; }
				auto name() const { return class_name; }
			};
			Settings settings;

			if (script::global_trace_cache.template contains_cached<RUNNABLE_SCRIPT_T>(settings)) {
				return script::global_trace_cache.template get_cached<RUNNABLE_SCRIPT_T>(settings);
			} 
			else if (script::global_trace_cache.template contains_file<RUNNABLE_SCRIPT_T>(settings)) {
				script::global_trace_cache.template load<RUNNABLE_SCRIPT_T>(settings);
				return script::global_trace_cache.template get_cached<RUNNABLE_SCRIPT_T>(settings);
			} else {

				NT<STACKED_DIAG_T>	  stack_out("stack_out");
				NT<STACKED_DIAG_T>	  stacked_diag("stacked_diag");
				NT<WEIGHTS_T>         weights("weights");

				static constexpr std::string_view code = R"ts(
FORWARD_ENTRYPOINT(self, stack_out, stacked_diag, weights):
	nstack = stacked_diag.shape[0]
	stack_out[:] = torch.mm(weights, stack_out.view(nstack, -1)).view_as(stack_out)
	stack_out *= stacked_diag.conj()
	return stack_out.sum(dim=0)
		)ts";

				script::runnable_to_compiled_builder_t<RUNNABLE_SCRIPT_T> builder(
					class_name,
					code,
					std::move(stack_out), std::move(stacked_diag), std::move(weights)
				);

				builder.compile();
				auto runnable_script = builder.decay_to_runnable_script();

				script::global_trace_cache.cache(settings, runnable_script);
				if (store_module) {
					script::global_trace_cache.template save<RUNNABLE_SCRIPT_T>(settings);
				}
				return runnable_script;
			}

		}

	private:
		bool _store_module;
		
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM+1> _stacked_diags;
		cache_tensor<TT,2> _weights;

		RUNNABLE_SCRIPT_T _runner;
	};

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
	requires is_normal_op_compatible<D,TT,DIM>
	class NORMAL_IDTW_T2_OP {
	public:

		static constexpr std::string_view class_name = "NORMAL_IDTW_T2_OP";

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
		NORMAL_IDTW_T2_OP(
			cache_tensor<TT,DIM+1>&& kernels, 
			cache_tensor<TT,DIM+1>&& kerneldiags,
			cache_tensor<TT,DIM>&& x, 
			cache_tensor<TT,2>&& weights,
			bool store_module = false
		)
			: 
			_store_module(store_module),
			_kernels(std::move(kernels)),
			_kerneldiags(std::move(kerneldiags)),
			_x(std::move(x)),
			_weights(std::move(weights)),
			_runner(std::remove_reference_t<decltype(*this)>::build_runner(store_module))
		{
		}

		tensor<D,TT,DIM+1> operator()(tensor<D,TT,DIM+1>&& stacked_diags)
		{
			if (_kernels.template shape<0>() != _kerneldiags.template shape<0>()) {
				throw std::runtime_error("Number of kernels and kernel diagonals must match.");
			}
			auto didx = stacked_diags.get_device_idx();

			auto out = make_empty_tensor_like(stacked_diags);

			auto x = _x.template get<D>(didx);
			auto kerneldiag = _kerneldiags.template operator[]<D>(didx, 0, Ellipsis{});

			fft::toeplitz_multiplication(
				stacked_diags,
				out,
				_kernels.template operator[]<D>(didx, 0, Ellipsis{}),
				std::nullopt,
				std::make_optional(std::cref(x)),
				std::make_optional(std::cref(kerneldiag)),
				fft::ToeplitzMultType::NONE,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzMultType::MULT,
				fft::ToeplitzMultType::MULT_CONJ,
				fft::ToeplitzAccumulateType::NONE
			);

			for (int i = 1; i < _kernels.template shape<0>(); ++i) {
				kerneldiag = hasty::move(_kerneldiags.template operator[]<D>(didx, i, Ellipsis{}));
				fft::toeplitz_multiplication(
					stacked_diags,
					out,
					_kernels.template operator[]<D>(didx, i, Ellipsis{}),
					std::nullopt,
					std::make_optional(std::cref(x)),
					std::make_optional(std::cref(kerneldiag)),
					fft::ToeplitzMultType::NONE,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzMultType::MULT,
					fft::ToeplitzMultType::MULT_CONJ,
					fft::ToeplitzAccumulateType::ACCUMULATE
				);
			}

			return _runner.run(
				std::move(out),
				_weights.template get<D>(didx)
			);
		}

	protected:

		using STACKED_DIAG_T = tensor<D,TT,DIM+1>;
		using KERNEL_T = tensor<D,TT,DIM>;
		using DIAG_T = tensor<D,TT,DIM>;
		using X_T = tensor<D,TT,DIM>;
		using WEIGHTS_T = tensor<D,TT,2>;

		using RUNNABLE_SCRIPT_T = script::runnable_script<
			STACKED_DIAG_T,
			STACKED_DIAG_T,
			WEIGHTS_T
		>;

		using THIS_TYPE_T = NORMAL_IDTW_T2_OP<D,TT,DIM>;

		static auto build_runner(bool store_module) -> RUNNABLE_SCRIPT_T 
		{
			struct Settings{
				auto to_string() const { return class_name; }
				auto name() const { return class_name; }
			};
			Settings settings;

			if (script::global_trace_cache.template contains_cached<RUNNABLE_SCRIPT_T>(settings)) {
				return script::global_trace_cache.template get_cached<RUNNABLE_SCRIPT_T>(settings);
			} else if (script::global_trace_cache.template contains_file<RUNNABLE_SCRIPT_T>(settings)) {
				script::global_trace_cache.template load<RUNNABLE_SCRIPT_T>(settings);
				return script::global_trace_cache.template get_cached<RUNNABLE_SCRIPT_T>(settings);
			} else {
				NT<STACKED_DIAG_T>	  	stack_out("stack_out");
				NT<WEIGHTS_T>           weights("weights");

				static constexpr std::string_view code = R"ts(
FORWARD_ENTRYPOINT(self, stack_out, weights):
	nstack = stack_out.shape[0]
	stack_out_shape = stack_out.shape
	stack_out[:] = torch.mm(weights, stack_out.view(nstack, -1)).view(stack_out_shape)
	return (stack_out,)
)ts";
				script::runnable_to_compiled_builder_t<RUNNABLE_SCRIPT_T> builder(
					class_name,
					code,
					std::move(stack_out), std::move(weights)
				);
		
				builder.compile();
				auto runnable_script = builder.decay_to_runnable_script();
				script::global_trace_cache.cache(settings, runnable_script);
				if (store_module) {
					script::global_trace_cache.template save<RUNNABLE_SCRIPT_T>(settings);
				}
				return runnable_script;
			}
		}

	private:
		bool _store_module;
		
		cache_tensor<TT,DIM+1> _kernels;
		cache_tensor<TT,DIM+1> _kerneldiags;
		cache_tensor<TT,DIM> _x;
		cache_tensor<TT,2> _weights;

		RUNNABLE_SCRIPT_T _runner;
	};

}
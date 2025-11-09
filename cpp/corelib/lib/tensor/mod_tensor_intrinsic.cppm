module;

#include "pch.hpp"

export module tensor:intrinsic;

//import pch;

import torch_base;
import util;
export import :base;

namespace hasty {


	export template<is_device D, is_tensor_type TT, size_t RANK>
	class tensor {
	private:

		struct tensor_base {

			tensor_base(const std::array<i64, RANK>& input_shape, TensorBackend input);

			tensor_base(span<RANK> input_shape, TensorBackend input);

			~tensor_base();

			base_t<TT>* mutable_data_ptr();

			const base_t<TT>* const_data_ptr() const;

			device_idx get_device_idx() const;

			std::array<i64, RANK> shape;
			TensorBackend underlying_tensor;
		};

		std::shared_ptr<tensor_base> _pimpl;

		template<is_device DO1, is_tensor_type TTO1, size_t RO1>
		friend class tensor;

	public:

		using device_type_t = D;
		using tensor_type_t = TT;
		static constexpr std::integral_constant<size_t, RANK> size = {};

		auto& get_pimpl();
		const auto& get_pimpl() const;

		auto get_tensor() -> TensorBackend&;
		auto get_tensor() const -> const TensorBackend&;

		auto decay_to_tensor();

		size_t ninstances() const;

		std::string str() const;

		base_t<TT> item() const requires (RANK == 0);

		template<size_t R>
		requires less_than<R, RANK>
		i64 shape() const;

		std::array<i64, RANK> shape() const;

		// Don't use this, use ::size() instead
		constexpr i64 ndim() const;

		i64 numel() const;

		std::string devicestr() const;

		device_idx get_device_idx() const;

		base_t<TT>* mutable_data_ptr();
		const base_t<TT>* const_data_ptr() const;

		// I know what I am doing...
		base_t<TT>* unconsted_data_ptr() const;

		tensor<D, TT, RANK> clone() const;

		tensor<D, TT, RANK+1> unsqueeze(i64 dim);

		template<size_t R>
		tensor<D, TT, R> view(span<R> shape);

		tensor<D, TT, 1> flatview();

		tensor<D, TT, 1> flatslice(index_type auto& idx);

		tensor<D, TT, RANK> contiguous();

		template<is_device DN>
		tensor<DN, TT, RANK> to(device_idx idx);

		template<is_tensor_type TTN>
		tensor<D,TTN,RANK> to();

		tensor<D, TT, 1> masked_select(const tensor<D,b8_t,RANK>& mask) const;

		void assign(std::array<i64, RANK>& input_shape, TensorBackend input);

		void assign(span<RANK> input_shape, TensorBackend input);

		template<size_t DIM, is_device D1, is_tensor_type TT1, size_t RANK1>
		requires less_than_or_equal<DIM, RANK1>
		friend tensor<D1,TT1,RANK1+1> stack(const std::vector<tensor<D1,TT1,RANK1>>& tensors);




		tensor();

		//tensor(const tensor&) = delete;
		tensor(const tensor& other);

		//tensor(tensor&&) = delete;
		tensor(tensor&& other);

		template<is_tensor_type TTN>
		tensor(tensor<D, TTN, RANK>&& other);

		template<is_device DN, is_tensor_type TTN>
		requires (!std::is_same_v<DN, D> || !std::is_same_v<TTN, TT>)
		tensor(tensor<DN, TTN, RANK>&& other, device_idx idx);

		tensor(const std::array<i64, RANK>& input_shape, TensorBackend input);

		tensor(span<RANK> input_shape, TensorBackend input);




		~tensor();

		// <============= OPERATORS =============>

		template<size_t R>
		requires less_than<R, RANK>
		tensor<D, TT, RANK>& operator=(const tensor<D, TT, R>& other);

		tensor<D, TT, RANK>& operator=(const tensor<D, TT, RANK>& other);

		tensor<D,TT,RANK>& operator=(move<tensor<D,TT,RANK>>&& other);

		tensor<D, TT, RANK>& operator=(tensor<D,TT,RANK>&& other);

		template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
		friend auto operator+(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator+(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1>;

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator+(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1>;


		template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
		friend auto operator-(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator-(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1>;

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator-(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1>;


		template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
		friend auto operator*(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator*(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1>;

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator*(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1>;


		template<is_device D1, is_tensor_type TT1, is_tensor_type TT2, size_t R1, size_t R2>
		friend auto operator/(const tensor<D1,TT1,R1>& lhs, const tensor<D1,TT2,R2>& rhs);

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator/(const tensor<D1,TT1,R1>& lhs, base_t<TT1> rhs) -> tensor<D1,TT1,R1>;

		template<is_device D1, is_tensor_type TT1, size_t R1>
		friend auto operator/(base_t<TT1> lhs, const tensor<D1,TT1,R1>& rhs) -> tensor<D1,TT1,R1>;


		template<size_t R>
		requires less_than_or_equal<R, RANK>
		void operator+=(const tensor<D,TT,R>& other);

		void operator+=(base_t<TT> val);

		template<size_t R>
		requires less_than_or_equal<R, RANK>
		void operator-=(const tensor<D,TT,R>& other);

		void operator-=(base_t<TT> val);

		template<size_t R>
		requires less_than_or_equal<R, RANK>
		void operator*=(const tensor<D,TT,R>& other);

		void operator*=(base_t<TT> val);

		template<size_t R>
		requires less_than_or_equal<R, RANK>
		void operator/=(const tensor<D,TT,R>& other);

		void operator/=(base_t<TT> val);

		/* must return a view */
		template<size_t N>
		requires less_than_or_equal<N, RANK> && less_than<0,RANK>
		auto operator[](const std::array<Slice, N>& slices) const -> tensor<D, TT, RANK>;

		/* must return a view */
		template<index_type ...Idx>
		requires less_than<0,RANK>
		auto operator[](std::tuple<Idx...> indices) const;

		/* must return a view */
		template<index_type ...Idx>
		requires less_than<0,RANK>
		auto operator[](Idx... indices) const;

		/* will not return a view */
		auto operator[](const tensor<D,b8_t,RANK>& mask) const -> tensor<D,TT,1>;


		// <=============== INPLACE OPERATIONS ================>


		template<is_device DN>
		tensor<D, TT, RANK>& copy_(tensor<DN,TT,RANK>& other, bool non_blocking=false);

		template<is_device DN>
		tensor<D, TT, RANK>& copy_(tensor<DN,TT,RANK>&& other, bool non_blocking=false);

		void fill_(base_t<TT> val);

		void zero_();

		void contiguous_();

		void set_(const tensor<D, TT, RANK>& other);

		void assign_(tensor<D, TT, RANK>&& other);

		void masked_scatter_(const tensor<D,b8_t,RANK>& mask, const tensor<D, TT, 1>& src);

		void masked_fill_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val);

		void masked_add_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val);

		void masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src);

		void masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src, base_t<TT> alpha);

		template<size_t R>
		requires less_than<R, RANK>
		tensor<D,TT,RANK>& add_(const tensor<D,TT,R>& other);

		template<typename T>
		requires std::integral<T> || std::floating_point<T>
		tensor<D,TT,RANK>& add_(T val);

		template<size_t R>
		requires less_than<R, RANK>
		tensor<D,TT,RANK>& sub_(const tensor<D,TT,R>& other);

		template<typename T>
		requires std::integral<T> || std::floating_point<T>
		tensor<D,TT,RANK>& sub_(T val);

		template<size_t R>
		requires less_than<R, RANK>
		tensor<D,TT,RANK>& mul_(const tensor<D,TT,R>& other);

		template<typename T>
		requires std::integral<T> || std::floating_point<T>
		tensor<D,TT,RANK>& mul_(T val);

		template<size_t R>
		requires less_than_or_equal<R, RANK>
		tensor<D,TT,RANK>& div_(const tensor<D,TT,R>& other);

		template<typename T>
		requires std::integral<T> || std::floating_point<T>
		tensor<D,TT,RANK>& div_(T val);

		tensor<D,TT,RANK>& exp_();
		
		// <================= MATH OPERATIONS ===============>

		auto norm() const -> std::conditional_t<is_fp32_tensor_type<TT>, float, double> 
		requires is_fp_tensor_type<TT>;

		auto abs() const;

		auto max() const;

		auto min() const;

		auto real() const;


		template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
		requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
		friend tensor<D1,TT1,R> fftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
			opt<fft_norm> norm);

		template<is_device D1, is_fp_complex_tensor_type TT1, size_t R, size_t R1, size_t R2>
		requires less_than_or_equal<R1, R> && less_than_or_equal<R2, R> && equal_or_left_zero<R1, R2>
		friend tensor<D1,TT1,R> ifftn(const tensor<D1,TT1,R>& t, span<R1> s, span<R2> dim,
			opt<fft_norm> norm);  

		template<is_device D1, is_tensor_type TT1, size_t R>
		friend tensor<D,TT1,0> vdot(const tensor<D1,TT1,R>& lhs, const tensor<D1,TT1,R>& rhs);

		template<is_device D1, is_tensor_type TT1, size_t R>
		friend tensor<D1,TT1,R> exp(const tensor<D1,TT1,R>& t);

		template<size_t SUMDIM, is_device D1, is_tensor_type TT1, size_t R>
		requires less_than<SUMDIM, R>
		friend tensor<D1,TT1,R-1> sum(const tensor<D1,TT1,R>& t);

		template<is_device D1, is_tensor_type TT1, size_t R>
		friend tensor<D1,TT1,0> sum(const tensor<D1,TT1,R>& t);

	};
	
	export template<typename T>
	concept is_tensor = requires(T t) {
		[]<is_device D2, is_tensor_type TT2, size_t RANK2>(tensor<D2,TT2,RANK2>&){}(t);
	};


	export template<is_tensor_type TT>
	class scalar {
	public:

		scalar(base_t<TT> val) : _val(val) {}

		template<is_device D>
		tensor<D, TT, 0> to_tensor() {
			return tensor<D, TT, 0>({}, hat::scalar_tensor(_val));
		}

	private:
		base_t<TT> _val;
	};
 
	// Define the is_vector_of_tensors concept
	export template<typename T>
	concept is_vector_of_tensors = requires(T t) {
		[]<is_device D2, is_tensor_type TT2, size_t RANK2>(std::vector<tensor<D2,TT2,RANK2>>&){}(t);
	};

	export template<typename T>
	concept is_tensor_or_vector_of_tensors = is_tensor<T> || is_vector_of_tensors<T>;

	// We also create a concept for being a tensor container. This is a type that is
	// mirrored by the tensor_prototype types

	template<typename T, int Depth = 0>
	struct is_tensor_container_impl {
		static constexpr bool value = is_tensor<T>;
	};

	// Specialization for std::vector
	template<typename T, int Depth>
	requires (Depth < 10)
	struct is_tensor_container_impl<std::vector<T>, Depth> {
		static constexpr bool value = is_tensor_container_impl<T, Depth+1>::value;
	};

	// Specialization for std::unordered_map
	template<typename K, typename V, int Depth>
	requires (Depth < 10) && (std::same_as<K, std::string> || std::same_as<K, std::size_t>)
	struct is_tensor_container_impl<std::unordered_map<K, V>, Depth> {
		static constexpr bool value = is_tensor_container_impl<V, Depth+1>::value;
	};

	// Specialization for std::tuple
	template<typename... Ts, int Depth>
	requires (Depth < 10)
	struct is_tensor_container_impl<std::tuple<Ts...>, Depth> {
		static constexpr bool value = (is_tensor_container_impl<Ts, Depth+1>::value && ...);
	};

	export template<typename T>
	concept is_tensor_container = is_tensor_container_impl<T,0>::value;

	export template<typename T, int Depth>
	concept is_tensor_container_depthlimited = is_tensor_container_impl<T, Depth>::value;

	/*
	// Explicit instantiations

	// f32_t
	template class tensor<cpu_t, f32_t, 1>;
	template class tensor<cpu_t, f32_t, 2>;
	template class tensor<cpu_t, f32_t, 3>;
	template class tensor<cpu_t, f32_t, 4>;

	template class tensor<cuda_t, f32_t, 1>;
	template class tensor<cuda_t, f32_t, 2>;
	template class tensor<cuda_t, f32_t, 3>;
	template class tensor<cuda_t, f32_t, 4>;

	// f64_t
	template class tensor<cpu_t, f64_t, 1>;
	template class tensor<cpu_t, f64_t, 2>;
	template class tensor<cpu_t, f64_t, 3>;
	template class tensor<cpu_t, f64_t, 4>;

	template class tensor<cuda_t, f64_t, 1>;
	template class tensor<cuda_t, f64_t, 2>;
	template class tensor<cuda_t, f64_t, 3>;
	template class tensor<cuda_t, f64_t, 4>;

	// c64_t
	template class tensor<cpu_t, c64_t, 1>;
	template class tensor<cpu_t, c64_t, 2>;
	template class tensor<cpu_t, c64_t, 3>;
	template class tensor<cpu_t, c64_t, 4>;

	template class tensor<cuda_t, c64_t, 1>;
	template class tensor<cuda_t, c64_t, 2>;
	template class tensor<cuda_t, c64_t, 3>;
	template class tensor<cuda_t, c64_t, 4>;

	// c128_t
	template class tensor<cpu_t, c128_t, 1>;
	template class tensor<cpu_t, c128_t, 2>;
	template class tensor<cpu_t, c128_t, 3>;
	template class tensor<cpu_t, c128_t, 4>;

	template class tensor<cuda_t, c128_t, 1>;
	template class tensor<cuda_t, c128_t, 2>;
	template class tensor<cuda_t, c128_t, 3>;
	template class tensor<cuda_t, c128_t, 4>;

	// i16_t
	template class tensor<cpu_t, i16_t, 1>;
	template class tensor<cpu_t, i16_t, 2>;
	template class tensor<cpu_t, i16_t, 3>;
	template class tensor<cpu_t, i16_t, 4>;

	template class tensor<cuda_t, i16_t, 1>;
	template class tensor<cuda_t, i16_t, 2>;
	template class tensor<cuda_t, i16_t, 3>;
	template class tensor<cuda_t, i16_t, 4>;

	// i32_t
	template class tensor<cpu_t, i32_t, 1>;
	template class tensor<cpu_t, i32_t, 2>;
	template class tensor<cpu_t, i32_t, 3>;
	template class tensor<cpu_t, i32_t, 4>;

	template class tensor<cuda_t, i32_t, 1>;
	template class tensor<cuda_t, i32_t, 2>;
	template class tensor<cuda_t, i32_t, 3>;
	template class tensor<cuda_t, i32_t, 4>;

	// i64_t
	template class tensor<cpu_t, i64_t, 1>;
	template class tensor<cpu_t, i64_t, 2>;
	template class tensor<cpu_t, i64_t, 3>;
	template class tensor<cpu_t, i64_t, 4>;

	template class tensor<cuda_t, i64_t, 1>;
	template class tensor<cuda_t, i64_t, 2>;
	template class tensor<cuda_t, i64_t, 3>;
	template class tensor<cuda_t, i64_t, 4>;

	// b8_t
	template class tensor<cpu_t, b8_t, 1>;
	template class tensor<cpu_t, b8_t, 2>;
	template class tensor<cpu_t, b8_t, 3>;
	template class tensor<cpu_t, b8_t, 4>;

	template class tensor<cuda_t, b8_t, 1>;
	template class tensor<cuda_t, b8_t, 2>;
	template class tensor<cuda_t, b8_t, 3>;
	template class tensor<cuda_t, b8_t, 4>;
	*/


}




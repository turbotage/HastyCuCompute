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
 
	template<typename T>
	struct is_tensor_impl : std::false_type {};

	template<is_device D, is_tensor_type TT, size_t RANK>
	struct is_tensor_impl<tensor<D,TT,RANK>> : std::true_type {};

	export template<typename T>
	concept is_tensor = is_tensor_impl<T>::value;

	export template<typename T>
	concept is_tensor_vector = 
		is_specialization_of<T, std::vector> &&
		is_tensor<typename T::value_type>;

	export template<typename T>
	concept is_tensor_dict_keytype =
		std::same_as<T, std::string> || std::same_as<T, i64>;

	export template<typename T>
	concept is_tensor_dict =
		is_specialization_of<T, std::unordered_map> &&
		is_tensor_dict_keytype<typename T::key_type> &&
		is_tensor<typename T::mapped_type>;

	export template<typename T>
	concept is_tensor_tuple = 
		is_specialization_of<T, std::tuple> &&
		[]<std::size_t... Is>(std::index_sequence<Is...>) {
			return (is_tensor<std::tuple_element_t<Is, T>> && ...);
		}(std::make_index_sequence<std::tuple_size_v<T>>{});


	// We also create a concept for being a tensor container. This is a type that is
	// mirrored by the tensor_prototype types, note that being a tensor container
	// is not the same as fullfilling one of the concepts above, tensor containers may be nested

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
	requires (Depth < 10) && is_tensor_dict_keytype<K>
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

}




module;

#include "pch.hpp"

export module tensor:impl_intrinsic_operator;
//module tensor:impl_intrinsic_operator;

import util;
import :intrinsic;

namespace hasty {

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than<R, RANK>
tensor<D,TT,RANK>& tensor<D,TT,RANK>::operator=(const tensor<D, TT, R>& other) {
	//debug::print_memory_usage("Before tensor::operator=(const tensor&): (with smalled rank) ");
	if (!_pimpl) {
		throw std::runtime_error("tensor::operator=: this tensor was not initialized");
	} else {
		_pimpl->underlying_tensor.copy_(other.get_tensor());
	}
	return *this;
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, RANK>& tensor<D,TT,RANK>::operator=(const tensor<D, TT, RANK>& other) {
	//debug::print_memory_usage("Before tensor::operator=(const tensor&): ");
	if (!_pimpl) {
		throw std::runtime_error("tensor::operator=: this tensor was not initialized");
	} else {
		_pimpl->underlying_tensor.copy_(other.get_tensor());
	}
	return *this;
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>& tensor<D,TT,RANK>::operator=(move<tensor<D,TT,RANK>>&& other) {
	auto& other_pimpl = other.get()._pimpl;
	_pimpl = std::move(other_pimpl);
	other_pimpl = nullptr;
	return *this;   
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, RANK>& tensor<D,TT,RANK>::operator=(tensor<D,TT,RANK>&& other) {
	//debug::print_memory_usage("Before tensor::operator=(const tensor&): ");
	if (!_pimpl) {
		throw std::runtime_error("tensor::operator=: this tensor was not initialized");
	} else {
		//_pimpl->underlying_tensor.copy_(other.get_tensor());
		_pimpl->underlying_tensor.copy_(other.get_tensor());
	}
	return *this;
}



template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than_or_equal<R, RANK>
void tensor<D,TT,RANK>::operator+=(const tensor<D,TT,R>& other) {
	_pimpl->underlying_tensor.add_(other.get_tensor());
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::operator+=(base_t<TT> val) {
	_pimpl->underlying_tensor.add_(val);
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than_or_equal<R, RANK>
void tensor<D,TT,RANK>::operator-=(const tensor<D,TT,R>& other) {
	_pimpl->underlying_tensor.sub_(other.get_tensor());
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::operator-=(base_t<TT> val) {
	_pimpl->underlying_tensor.sub_(val);
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than_or_equal<R, RANK>
void tensor<D,TT,RANK>::operator*=(const tensor<D,TT,R>& other) {
	_pimpl->underlying_tensor.mul_(other.get_tensor());
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::operator*=(base_t<TT> val) {
	_pimpl->underlying_tensor.mul_(val);
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than_or_equal<R, RANK>
void tensor<D,TT,RANK>::operator/=(const tensor<D,TT,R>& other) {
	_pimpl->underlying_tensor.div_(other.get_tensor());
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::operator/=(base_t<TT> val) {
	_pimpl->underlying_tensor.div_(val);
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t N>
requires less_than_or_equal<N, RANK> && less_than<0,RANK>
auto tensor<D,TT,RANK>::operator[](const std::array<Slice, N>& slices) const -> tensor<D, TT, RANK> {
	
	//debug::print_memory_usage("Before tensor::operator[array<Slice>]: ");

	TensorBackend tensorview = _pimpl->underlying_tensor.index(tch::torchidx(std::tuple_cat(slices)));

	// In inference views are not tracked
	//if (!tensorview.is_view())
	//    throw std::runtime_error("tensor::operator[]: tensorview is not a view");

	auto pimpl_dptr = _pimpl->underlying_tensor.data_ptr();
	auto byte_length = _pimpl->underlying_tensor.numel() * _pimpl->underlying_tensor.element_size();
	auto view_dptr = tensorview.data_ptr();

	if ((view_dptr < pimpl_dptr) || (view_dptr > (pimpl_dptr + byte_length))) {
		throw std::runtime_error("tensor::operator[]: tensorview data not pointing into underlying tensor");
	}

	if (tensorview.ndimension() != RANK)
		throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");

	std::array<i64, RANK> new_shape;
	for_sequence<RANK>([&](auto i) {
		new_shape[i] = tensorview.size(i);
	});

	//debug::print_memory_usage("After tensor::operator[array<Slice>]: ");

	return tensor<D, TT, RANK>(new_shape, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<index_type ...Idx>
requires less_than<0,RANK>
auto tensor<D,TT,RANK>::operator[](std::tuple<Idx...> indices) const {
	//debug::print_memory_usage("Before tensor::operator[tuple<Idx...>]: ");
	
	constexpr auto RETRANK = get_slice_rank<RANK, Idx...>();

	TensorBackend tensorview = _pimpl->underlying_tensor.index(tch::torchidx(indices));

	// In inference views are not tracked
	//if (!tensorview.is_view())
	//    throw std::runtime_error("tensor::operator[]: tensorview is not a view");

	auto pimpl_dptr = _pimpl->underlying_tensor.data_ptr();
	auto byte_length = _pimpl->underlying_tensor.numel() * _pimpl->underlying_tensor.element_size();
	auto view_dptr = tensorview.data_ptr();

	if ((view_dptr < pimpl_dptr) || (view_dptr > (pimpl_dptr + byte_length))) {
		throw std::runtime_error("tensor::operator[]: tensorview data not pointing into underlying tensor");
	}

	if (tensorview.ndimension() != RETRANK) {
		throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");
	}

	std::array<i64, RETRANK> new_shape;
	for_sequence<RETRANK>([&](auto i) {
		new_shape[i] = tensorview.size(i);
	});

	//debug::print_memory_usage("After tensor::operator[tuple<Idx...>]: ");

	//return tensor_factory<D, TT, RETRANK>::make(new_shape, std::move(tensorview));
	return tensor<D,TT,RETRANK>(new_shape, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<index_type ...Idx>
requires less_than<0,RANK>
auto tensor<D,TT,RANK>::operator[](Idx... indices) const {
	//debug::print_memory_usage("Before tensor::operator[Idx...]: ");
	
	constexpr auto RETRANK = get_slice_rank<RANK, Idx...>();

	auto torch_indices = {tch::torchidx(indices)...};

	TensorBackend tensorview = _pimpl->underlying_tensor.index(torch_indices);

	// In inference views are not tracked
	//if (!tensorview.is_view())
	//    throw std::runtime_error("tensor::operator[]: tensorview is not a view");

	size_t pimpl_dptr = (size_t)_pimpl->underlying_tensor.data_ptr();
	size_t byte_length = _pimpl->underlying_tensor.numel() * _pimpl->underlying_tensor.element_size();
	size_t view_dptr = (size_t)tensorview.data_ptr();

	if ((view_dptr < pimpl_dptr) || (view_dptr > (pimpl_dptr + byte_length))) {
		throw std::runtime_error("tensor::operator[]: tensorview data not pointing into underlying tensor");
	}

	if (tensorview.ndimension() != RETRANK) {
		throw std::runtime_error("tensor::operator[]: tensorview.ndimension() did not match RETRANK");
	}

	std::array<i64, RETRANK> new_shape;
	for_sequence<RETRANK>([&](auto i) {
		new_shape[i] = tensorview.size(i);
	});

	//debug::print_memory_usage("After tensor::operator[Idx...]: ");

	return tensor<D, TT, RETRANK>(new_shape, std::move(tensorview));
	//return tensor<D,TT,RETRANK>(new_shape, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
auto tensor<D,TT,RANK>::operator[](const tensor<D,b8_t,RANK>& mask) const -> tensor<D,TT,1> {
	TensorBackend ret = _pimpl->underlying_tensor.index(mask.get_tensor());
	std::array<i64, 1> new_shape = {ret.size(0)};
	return tensor<D,TT,1>(new_shape, std::move(ret));
}


}
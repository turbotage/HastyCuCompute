module;

#include "pch.hpp"

export module tensor:impl_intrinsic;
//module tensor:impl_intrinsic;

import util;
import :intrinsic;

namespace hasty {


template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor_base::tensor_base(const std::array<i64, RANK>& input_shape, TensorBackend input)
	: shape(input_shape), underlying_tensor(std::move(input))
{}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor_base::tensor_base(span<RANK> input_shape, TensorBackend input)
	: shape(input_shape.to_arr()), underlying_tensor(std::move(input))
{}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor_base::~tensor_base() {
	//debug::print_memory_usage("~tensor_impl: ");
}

template<is_device D, is_tensor_type TT, size_t RANK>
base_t<TT>* tensor<D,TT,RANK>::tensor_base::mutable_data_ptr() { 
	return static_cast<base_t<TT>*>(underlying_tensor.data_ptr()); 
}

template<is_device D, is_tensor_type TT, size_t RANK>
const base_t<TT>* tensor<D,TT,RANK>::tensor_base::const_data_ptr() const { 
	return static_cast<base_t<TT>*>(underlying_tensor.data_ptr()); 
}

template<is_device D, is_tensor_type TT, size_t RANK>
device_idx tensor<D,TT,RANK>::tensor_base::get_device_idx() const {
	return static_cast<device_idx>(underlying_tensor.device().index());
}




template<is_device D, is_tensor_type TT, size_t RANK>
auto& tensor<D,TT,RANK>::get_pimpl() { return _pimpl; }

template<is_device D, is_tensor_type TT, size_t RANK>
const auto& tensor<D,TT,RANK>::get_pimpl() const { return _pimpl; }

template<is_device D, is_tensor_type TT, size_t RANK>
auto tensor<D,TT,RANK>::get_tensor() -> TensorBackend& { return _pimpl->underlying_tensor; }

template<is_device D, is_tensor_type TT, size_t RANK>
auto tensor<D,TT,RANK>::get_tensor() const -> const TensorBackend& { return _pimpl->underlying_tensor; }

template<is_device D, is_tensor_type TT, size_t RANK>
auto tensor<D,TT,RANK>::decay_to_tensor() { 
	TensorBackend ten = std::move(_pimpl->underlying_tensor); 
	_pimpl = nullptr;
	return std::move(ten);
}

template<is_device D, is_tensor_type TT, size_t RANK>
size_t tensor<D,TT,RANK>::ninstances() const { return _pimpl.use_count(); }

template<is_device D, is_tensor_type TT, size_t RANK>
std::string tensor<D,TT,RANK>::str() const { return _pimpl->underlying_tensor.toString(); }

template<is_device D, is_tensor_type TT, size_t RANK>
base_t<TT> tensor<D,TT,RANK>::item() const requires (RANK == 0) {
	return _pimpl->underlying_tensor.template item<base_t<TT>>();
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
requires less_than<R, RANK>
i64 tensor<D,TT,RANK>::shape() const { 
	if (_pimpl->shape[R] != _pimpl->underlying_tensor.size(R)) {
		throw std::runtime_error("tensor::shape(): shape does not match underlying tensor size");
	}
	return _pimpl->shape[R]; 
}

template<is_device D, is_tensor_type TT, size_t RANK>
std::array<i64, RANK> tensor<D,TT,RANK>::shape() const { 
	return _pimpl->shape; 
}

template<is_device D, is_tensor_type TT, size_t RANK>
constexpr i64 tensor<D,TT,RANK>::ndim() const { 
	assert(RANK == _pimpl->underlying_tensor.dim()); 
	return RANK; 
}

template<is_device D, is_tensor_type TT, size_t RANK>
i64 tensor<D,TT,RANK>::numel() const { 
	i64 nelem = 1; 
	for_sequence<RANK>([&](auto i) { nelem *= _pimpl->shape[i]; });
	assert(nelem == _pimpl->underlying_tensor.numel());
	return nelem;
}

template<is_device D, is_tensor_type TT, size_t RANK>
std::string tensor<D,TT,RANK>::devicestr() const { return _pimpl->underlying_tensor.device().str(); }

template<is_device D, is_tensor_type TT, size_t RANK>
device_idx tensor<D,TT,RANK>::get_device_idx() const { return _pimpl->get_device_idx(); }

template<is_device D, is_tensor_type TT, size_t RANK>
base_t<TT>* tensor<D,TT,RANK>::mutable_data_ptr() { return _pimpl->mutable_data_ptr(); }

template<is_device D, is_tensor_type TT, size_t RANK>
const base_t<TT>* tensor<D,TT,RANK>::const_data_ptr() const { return _pimpl->const_data_ptr(); }

template<is_device D, is_tensor_type TT, size_t RANK>
base_t<TT>* tensor<D,TT,RANK>::unconsted_data_ptr() const { return const_cast<base_t<TT>*>(_pimpl->const_data_ptr()); }

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, RANK> tensor<D,TT,RANK>::clone() const {
	TensorBackend newtensor = _pimpl->underlying_tensor.clone();
	return tensor<D, TT, RANK>(_pimpl->shape, std::move(newtensor));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, RANK+1> tensor<D,TT,RANK>::unsqueeze(i64 dim) {
	//debug::print_memory_usage("Before unsqueeze(): ");

	TensorBackend tensorview = _pimpl->underlying_tensor.unsqueeze(dim);
	if (tensorview.data_ptr() != _pimpl->underlying_tensor.data_ptr()) {
		throw std::runtime_error("tensor::unsqueeze: tensorview data not pointing into underlying tensor");
	}
	std::array<i64, RANK+1> new_shape;
	for_sequence<RANK+1>([&](auto i) {
		if (i < dim) {
			new_shape[i] = _pimpl->shape[i];
		} else if (i == dim) {
			new_shape[i] = 1;
		} else {
			new_shape[i] = _pimpl->shape[i-1];
		}
	});

	//debug::print_memory_usage("After unsqueeze(): ");

	return tensor<D, TT, RANK+1>(new_shape, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<size_t R>
tensor<D, TT, R> tensor<D,TT,RANK>::view(span<R> shape) {
	TensorBackend tensorview = _pimpl->underlying_tensor.view(shape.to_torch_arr());
	return tensor<D, TT, R>(shape, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, 1> tensor<D,TT,RANK>::flatview() {
	TensorBackend tensorview = _pimpl->underlying_tensor.view(-1);
	return tensor<D, TT, 1>({tensorview.size(0)}, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, 1> tensor<D,TT,RANK>::flatslice(index_type auto& idx) {
	TensorBackend tensorview = _pimpl->underlying_tensor.view(-1);
	tensorview = tensorview.index(tch::torchidx(idx));
	return tensor<D, TT, 1>({tensorview.size(0)}, std::move(tensorview));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, RANK> tensor<D,TT,RANK>::contiguous() {
	return tensor<D, TT, RANK>(_pimpl->shape, _pimpl->underlying_tensor.contiguous());
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<is_device DN>
tensor<DN, TT, RANK> tensor<D,TT,RANK>::to(device_idx idx) {
	static_assert(!(std::is_same_v<D, cpu_t> && std::is_same_v<DN, cpu_t>), "don't move from cpu to cpu");

	if (idx == get_device_idx())
		throw std::runtime_error("tensor::to: tensor already on device");
	hat::Device didx = get_backend_device(idx);
	return tensor<DN,TT,RANK>(_pimpl->shape, 
		std::move(_pimpl->underlying_tensor.to(didx)));
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<is_tensor_type TTN>
tensor<D,TTN,RANK> tensor<D,TT,RANK>::to() {
	hat::ScalarType stype = scalar_type_func<TTN>();
	return tensor<D,TTN,RANK>(_pimpl->shape, 
		std::move(_pimpl->underlying_tensor.to(stype)));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D, TT, 1> tensor<D,TT,RANK>::masked_select(const tensor<D,b8_t,RANK>& mask) const {
	hat::Tensor ret = _pimpl->underlying_tensor.masked_select(mask.get_tensor());
	std::array<i64, 1> new_shape = {ret.size(0)};
	return tensor<D,TT,1>(new_shape, std::move(ret));
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::assign(std::array<i64, RANK>& input_shape, TensorBackend input) {
	_pimpl = std::make_shared<tensor<D,TT,RANK>::tensor_base>(input_shape, std::move(input));
}

template<is_device D, is_tensor_type TT, size_t RANK>
void tensor<D,TT,RANK>::assign(span<RANK> input_shape, TensorBackend input) {
	_pimpl = std::make_shared<tensor<D,TT,RANK>::tensor_base>(input_shape, std::move(input));
}

export template<size_t DIM, is_device D1, is_tensor_type TT1, size_t RANK1>
requires less_than_or_equal<DIM, RANK1>
tensor<D1,TT1,RANK1+1> stack(const std::vector<tensor<D1,TT1,RANK1>>& tensors)
{
	if (tensors.empty()) {
		throw std::runtime_error("tensor::stack: cannot stack empty vector");
	}

	std::array<i64, RANK1> oneshape;
	std::array<i64, RANK1+1> new_shape;
	for_sequence<RANK1>([&](auto i) { 
		if (i == DIM) {
			new_shape[i] = tensors.size();
		} else {
			new_shape[i] = tensors[0].template shape<i>();
		}
	});

	std::vector<TensorBackend> tensor_views;
	tensor_views.reserve(tensors.size());
	for (const auto& t : tensors) {
		tensor_views.emplace_back(t.get_tensor());
	}

	TensorBackend stacked_tensor = hat::stack(std::move(tensor_views), DIM);
	return tensor<D1,TT1,RANK1+1>(new_shape, std::move(stacked_tensor));
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor() {}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor(const tensor& other) { _pimpl = other._pimpl; }

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor(tensor&& other) { _pimpl = std::move(other._pimpl); }

template<is_device D, is_tensor_type TT, size_t RANK>
template<is_tensor_type TTN>
tensor<D,TT,RANK>::tensor(tensor<D, TTN, RANK>&& other) {
	if (!_pimpl) {
		_pimpl = std::make_shared<tensor<D,TT,RANK>::tensor_base>(
				std::move(other._pimpl->shape), 
				std::move(other._pimpl->underlying_tensor)
			);
	} else {
		_pimpl->shape = std::move(other._pimpl->shape);
		_pimpl->underlying_tensor = std::move(other._pimpl->underlying_tensor);
	}
	other._pimpl = nullptr;

	if (!std::is_same_v<TT, TTN>) {
		_pimpl->underlying_tensor = _pimpl->underlying_tensor.to(scalar_type_func<TT>());
	}
}

template<is_device D, is_tensor_type TT, size_t RANK>
template<is_device DN, is_tensor_type TTN>
requires (!std::is_same_v<DN, D> || !std::is_same_v<TTN, TT>)
tensor<D,TT,RANK>::tensor(tensor<DN, TTN, RANK>&& other, device_idx idx) {
	if (!device_match<D>(idx))
		throw std::runtime_error("tensor::tensor: device mismatch");

	if (!_pimpl) {
		_pimpl = std::make_shared<tensor<D,TT,RANK>::tensor_base>(
				std::move(other._pimpl->shape), 
				std::move(other._pimpl->underlying_tensor)
			);
	} else {
		_pimpl->shape = std::move(other._pimpl->shape);
		_pimpl->underlying_tensor = std::move(other._pimpl->underlying_tensor);
	}
	other._pimpl = nullptr;

	if (!(std::is_same_v<D, cpu_t> && std::is_same_v<DN, cpu_t>)) {
		if (get_device_idx() != idx) {
			_pimpl->underlying_tensor = _pimpl->underlying_tensor.to(get_backend_device(idx));
		}
	}

	if (!std::is_same_v<TT, TTN>) {
		_pimpl->underlying_tensor = _pimpl->underlying_tensor.to(scalar_type_func<TT>());
	}
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor(const std::array<i64, RANK>& input_shape, TensorBackend input) 
	: _pimpl(std::make_shared<tensor<D,TT,RANK>::tensor_base>(input_shape, std::move(input)))
{
	//debug::print_memory_usage("tensor::tensor(std::array, TensorBackend): ");
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::tensor(span<RANK> input_shape, TensorBackend input)
	: _pimpl(std::make_shared<tensor<D,TT,RANK>::tensor_base>(input_shape, std::move(input)))
{
	//debug::print_memory_usage("tensor::tensor(span, TensorBackend): ");
}

template<is_device D, is_tensor_type TT, size_t RANK>
tensor<D,TT,RANK>::~tensor() {}


}
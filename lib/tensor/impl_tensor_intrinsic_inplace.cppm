module;

#include "pch.hpp"

export module tensor:impl_intrinsic_inplace;
//module tensor:impl_intrinsic_inplace;

import torch_base;
import util;
import :intrinsic;

namespace hasty {

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<is_device DN>
    tensor<D, TT, RANK>& tensor<D,TT,RANK>::copy_(tensor<DN,TT,RANK>& other, bool non_blocking) {
        _pimpl->underlying_tensor.copy_(other.get_tensor(), non_blocking);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<is_device DN>
    tensor<D, TT, RANK>& tensor<D,TT,RANK>::copy_(tensor<DN,TT,RANK>&& other, bool non_blocking) {
        _pimpl->underlying_tensor.copy_(other.get_tensor(), non_blocking);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::fill_(base_t<TT> val) {
        if constexpr(!is_fp_complex_tensor_type<TT>) {
            _pimpl->underlying_tensor.fill_(val);
        }
        else {
            auto cten = hat::complex(hat::scalar_tensor(val.real()), hat::scalar_tensor(val.imag()));
            _pimpl->underlying_tensor.fill_(cten);
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::zero_() {
        _pimpl->underlying_tensor.zero_();
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::contiguous_() {
        _pimpl->underlying_tensor.contiguous_();
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::set_(const tensor<D, TT, RANK>& other) {
        if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
            throw std::runtime_error("tensor::set_: other tensor shape did not match this tensor shape");
        _pimpl->underlying_tensor.set_(other._pimpl->underlying_tensor);
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::assign_(tensor<D, TT, RANK>&& other) {
        if (other._pimpl->underlying_tensor.sizes() != _pimpl->underlying_tensor.sizes())
            throw std::runtime_error("tensor::assign_: other tensor shape did not match this tensor shape");
        _pimpl->underlying_tensor = std::move(other._pimpl->underlying_tensor);
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::masked_scatter_(const tensor<D,b8_t,RANK>& mask, const tensor<D, TT, 1>& src) {
        _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), src.get_tensor());
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::masked_fill_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val) {
        if constexpr(!is_fp_complex_tensor_type<TT>) {
            _pimpl->underlying_tensor.masked_fill_(mask.get_tensor(), val);
        }
        else {
            auto cten = hat::complex(hat::scalar_tensor(val.real()), hat::scalar_tensor(val.imag()));
            _pimpl->underlying_tensor.masked_fill_(mask.get_tensor(), cten);
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::masked_add_(const tensor<D,b8_t,RANK>& mask, base_t<TT> val) {
        if constexpr(!is_fp_complex_tensor_type<TT>) {
            auto adder = _pimpl->underlying_tensor.masked_select(mask.get_tensor()) + hat::scalar_tensor(val);
            _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), adder);
        }
        else {
            auto cten = hat::complex(hat::scalar_tensor(val.real()), hat::scalar_tensor(val.imag()));
            auto adder = _pimpl->underlying_tensor.masked_select(mask.get_tensor()) + cten;
            _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), adder);
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src) {
        auto adder = _pimpl->underlying_tensor.masked_select(mask.get_tensor()) + src.get_tensor();
        _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), adder);
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    void tensor<D,TT,RANK>::masked_add_(const tensor<D,b8_t,RANK>& mask, const tensor<D,TT,1>& src, base_t<TT> alpha) {
        if constexpr(!is_fp_complex_tensor_type<TT>) {
            auto adder = _pimpl->underlying_tensor.masked_select(mask.get_tensor()) + src.get_tensor() * hat::scalar_tensor(alpha);
            _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), adder);
        }
        else {
            auto cten = hat::complex(hat::scalar_tensor(alpha.real()), hat::scalar_tensor(alpha.imag()));
            auto adder = _pimpl->underlying_tensor.masked_select(mask.get_tensor()) + src.get_tensor() * cten;
            _pimpl->underlying_tensor.masked_scatter_(mask.get_tensor(), adder);
        }
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<size_t R>
    requires less_than<R, RANK>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::add_(const tensor<D,TT,R>& other) {
        _pimpl->underlying_tensor.add_(other.get_tensor());
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<typename T>
    requires std::integral<T> || std::floating_point<T>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::add_(T val) {
        _pimpl->underlying_tensor.add_(val);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<size_t R>
    requires less_than<R, RANK>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::sub_(const tensor<D,TT,R>& other) {
        _pimpl->underlying_tensor.sub_(other.get_tensor());
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<typename T>
    requires std::integral<T> || std::floating_point<T>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::sub_(T val) {
        _pimpl->underlying_tensor.sub_(val);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<size_t R>
    requires less_than<R, RANK>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::mul_(const tensor<D,TT,R>& other) {
        _pimpl->underlying_tensor.mul_(other.get_tensor());
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<typename T>
    requires std::integral<T> || std::floating_point<T>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::mul_(T val) {
        _pimpl->underlying_tensor.mul_(val);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<size_t R>
    requires less_than_or_equal<R, RANK>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::div_(const tensor<D,TT,R>& other) {
        _pimpl->underlying_tensor.div_(other.get_tensor());
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    template<typename T>
    requires std::integral<T> || std::floating_point<T>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::div_(T val) {
        _pimpl->underlying_tensor.div_(val);
        return *this;
    }

    template<is_device D, is_tensor_type TT, size_t RANK>
    tensor<D,TT,RANK>& tensor<D,TT,RANK>::exp_() {
        _pimpl->underlying_tensor.exp_();
        return *this;
    }


}




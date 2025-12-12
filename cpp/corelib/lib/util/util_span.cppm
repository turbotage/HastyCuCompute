module;

#include "pch.hpp"

export module util:span;

//import pch;

import std;

import :meta;
import :typing;

namespace hasty {

export template<std::integral T, size_t N>
struct arbspan {

	//nullspan
	arbspan() : _data(nullptr) {};

	arbspan(T const (&list)[N]) 
		: _data(list) {}

	arbspan(const T* listptr)
		: _data(listptr) {}

	arbspan(hat::ArrayRef<T> arr)
		: _data(arr.data())
	{}

	arbspan(const std::array<T, N>& arr)
		: _data(arr.data())
	{}

	/*
	span(std::span<const T, N> span) 
		: _data(span.data()) {}
	*/
	hat::ArrayRef<T> to_arr_ref() {
		return hat::ArrayRef<T>(_data, N);
	}

	std::array<T, N> to_arr() {
		std::array<T, N> arr;
		for (size_t i = 0; i < N; i++) {
			arr[i] = _data[i];
		}
		return arr;
	}

	arbspan(std::nullopt_t)
		: _data(nullptr) {}

	const T& operator[](size_t index) const {
		if (index >= N) {
			throw std::out_of_range("Index out of range");
		}
		return _data[index];
	}

	template<size_t I>
	requires less_than<I,N>
	const T& get() {
		return _data[I];
	}

	constexpr size_t size() const { return N; }

	bool has_value() const {
		return _data != nullptr;
	}

private:
	const T* _data;
};

export template<std::integral I, size_t R>
constexpr std::string span_to_str(arbspan<I,R> arr, bool as_tuple = true) {
	std::string retstr = as_tuple ? "(" : "[";
	
	for_sequence<R>([&](auto i) {
		retstr += std::to_string(arr.template get<i>());
		if constexpr(i < R - 1) {
			retstr += ",";
		}
	});
	retstr += as_tuple ? ")" : "]";
	return retstr;
}

export template<size_t N>
struct span {

	//nullspan
	span() : _data(nullptr) {};

	span(i64 const (&list)[N]) 
		: _data(list) {}

	span(const i64* listptr)
		: _data(listptr) {}

	span(hat::ArrayRef<i64> arr)
		: _data(arr.data())
	{}

	span(const std::array<i64, N>& arr)
		: _data(arr.data())
	{}

	span(const std::array<i64, N>& arr, i32 offset)
		: _data(arr.data() + offset)
	{}

	std::array<i64,N> operator*(i64 m) const {
		std::array<i64, N> arr;
		for_sequence<N>([&](auto i) {
			arr[i] = _data[i] * m;
		});
		return arr;
	}

	hat::ArrayRef<i64> to_arr_ref() {
		return hat::ArrayRef<i64>(_data, N);
	}

	hat::OptionalArrayRef<i64> to_opt_arr_ref() {
		if (N == 0) {
			return std::nullopt;
		}
		return hat::ArrayRef<i64>(_data, N);
	}

	std::array<i64, N> to_arr() {
		std::array<i64, N> arr;
		for (size_t i = 0; i < N; i++) {
			arr[i] = _data[i];
		}
		return arr;
	}

	span(std::nullopt_t)
		: _data(nullptr) {}

	const i64& operator[](size_t index) const {
		if (index >= N) {
			throw std::out_of_range("Index out of range");
		}
		return _data[index];
	}

	template<size_t I>
	requires less_than<I,N>
	const i64& get() const {
		return _data[I];
	}

	constexpr size_t size() const { return N; }

	bool has_value() const {
		return _data != nullptr;
	}

	template<size_t R1, size_t R2>
	friend std::array<i64,R1+R2> operator+(const span<R1>& s1, const span<R2>& s2);

private:
	const i64* _data;
};

export template<size_t R1, size_t R2>
std::array<i64,R1+R2> operator+(const span<R1>& s1, const span<R2>& s2) {
	std::array<i64,R1+R2> ret;
	for_sequence<R1>([&](auto i) {
		ret[i] = s1.template get<i>();
	});
	for_sequence<R2>([&](auto i) {
		ret[i+R1] = s2.template get<i>();
	});
	return ret;
}

export using nullspan = span<0>;

export template<size_t R>
constexpr std::string span_to_str(span<R> arr, bool as_tuple = true) {
	std::string retstr = as_tuple ? "(" : "[";
	
	for_sequence<R>([&](auto i) {
		retstr += std::to_string(arr.template get<i>());
		if constexpr(i < R - 1) {
			retstr += ",";
		}
	});

	retstr += as_tuple ? ")" : "]";
	return retstr;
}

}
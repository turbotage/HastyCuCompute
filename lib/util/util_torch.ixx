module;
 
#include <ostream>
#include <sstream>
#include "c10/cuda/CUDAStream.h"
#include "../pch.hpp"

export module util:torch;

import :idx;

namespace hasty {

    namespace tch {
        
        export template<typename T>
        c10::optional<T> torch_optional(const std::optional<T>& opt)
        {
            if (opt.has_value()) {
                return c10::optional(opt.value());
            }
            return c10::nullopt;
        }

        export template<typename R, typename T>
        c10::optional<R> torch_optional(const std::optional<T>& opt)
        {
            if (opt.has_value()) {
                return c10::optional<R>(opt.value());
            }
            return c10::nullopt;
        }

        export template<index_type Idx>
        at::indexing::TensorIndex torchidx(Idx idx) {
            if constexpr(std::is_same_v<Idx, None>) {
                return at::indexing::None;
            } 
            else if constexpr(std::is_same_v<Idx, Ellipsis>) {
                return at::indexing::Ellipsis;
            }
            else if constexpr(std::is_same_v<Idx, Slice>) {
                return at::indexing::Slice(
                    torch_optional<c10::SymInt>(idx.start),
                    torch_optional<c10::SymInt>(idx.end),
                    torch_optional<c10::SymInt>(idx.step));
            } else if constexpr(std::is_integral_v<Idx>) {
                return idx;
            } else {
                static_assert(false);
            }
        }

        template<index_type... Idx, size_t... Is>
        auto torchidx_impl(std::tuple<Idx...> idxs, std::index_sequence<Is...>) {
            return std::array<at::indexing::TensorIndex, sizeof...(Idx)>{torchidx(std::get<Is>(idxs))...};
        }

        export template<index_type... Idx>
        auto torchidx(std::tuple<Idx...> idxs) {
            return torchidx_impl(idxs, std::make_index_sequence<sizeof...(Idx)>{});
        }

        export template<index_type Idx>
        std::string torchidxstr(Idx idx) {
            if constexpr(std::is_same_v<Idx, None>) {
                return "None";
            } 
            else if constexpr(std::is_same_v<Idx, Ellipsis>) {
                return "...";
            }
            else if constexpr(std::is_same_v<Idx, Slice>) {
                // If the Slice has start, end, and step values, format them as "start:end:step"
                if (idx.start.has_value() && idx.end.has_value() && idx.step.has_value()) {
                    return std::format("{}:{}:{}", idx.start.value(), idx.end.value(), idx.step.value());
                } 
                // If the Slice has only start and end values, format them as "start:end"
                else if (idx.start.has_value() && idx.end.has_value()) {
                    return std::format("{}:{}", idx.start.value(), idx.end.value());
                } 
                // If the Slice has only start and step values, format them as "start::step"
                else if (idx.start.has_value() && idx.step.has_value()) {
                    return std::format("{}::{}", idx.start.value(), idx.step.value());
                } 
                // If the Slice has only end and step values, format them as ":end:step"
                else if (idx.end.has_value() && idx.step.has_value()) {
                    return std::format(":{}:{}", idx.end.value(), idx.step.value());
                } 
                // If the Slice has only a start value, format it as "start:"
                else if (idx.start.has_value()) {
                    return std::format("{}:", idx.start.value());
                } 
                // If the Slice has only an end value, format it as ":end"
                else if (idx.end.has_value()) {
                    return std::format(":{}", idx.end.value());
                } 
                // If the Slice has only a step value, format it as "::step"
                else if (idx.step.has_value()) {
                    return std::format("::{}", idx.step.value());
                }
            } 
            else if constexpr(std::is_integral_v<Idx>) {
                return std::to_string(idx);
            }
        }

        export std::vector<at::Stream> get_streams(const at::optional<std::vector<at::Stream>>& streams)
        {
            if (streams.has_value()) {
                return *streams;
            }
            else {
                return { at::cuda::getDefaultCUDAStream() };
            }
        }

		export std::vector<at::Stream> get_streams(const at::optional<at::ArrayRef<at::Stream>>& streams)
        {
            if (streams.has_value()) {
                return (*streams).vec();
            }
            else {
                return { c10::cuda::getDefaultCUDAStream() };
            }
        }

		export std::stringstream print_4d_xyz(const at::Tensor& toprint)
        {
            std::stringstream printer;
            printer.precision(2);
            printer << std::scientific;
            int tlen = toprint.size(0);
            int zlen = toprint.size(1);
            int ylen = toprint.size(2);
            int xlen = toprint.size(3);

            auto closer = [&printer](int iter, int length, bool brackets, int spacing)
            {
                if (brackets) {
                    for (int i = 0; i < spacing; ++i)
                        printer << " ";
                    printer << "]";
                }
                if (iter + 1 != length)
                    printer << ",";
                if (brackets)
                    printer << "\n";
            };

            auto value_printer = [&printer]<typename T>(T val)
            {
                if (val < 0.0)
                    printer << val;
                else
                    printer << " " << val;
            };

            for (int t = 0; t < tlen; ++t) {
                printer << "[\n";
                for (int z = 0; z < zlen; ++z) {
                    printer << " [\n";
                    for (int y = 0; y < ylen; ++y) {
                        printer << "  [";
                        for (int x = 0; x < xlen; ++x) {
                            switch (toprint.dtype().toScalarType()) {
                            case c10::ScalarType::Float:
                                value_printer(toprint.index({ t,z,y,x }).item<float>());
                            break;
                            case c10::ScalarType::Double:
                                value_printer(toprint.index({ t,z,y,x }).item<float>());
                            break;
                            case c10::ScalarType::ComplexFloat:
                            {
                                float val;
                                val = at::real(toprint.index({ t,z,y,x })).item<float>();
                                printer << "("; value_printer(val); printer << ",";
                                val = at::imag(toprint.index({ t,z,y,x })).item<float>();
                                value_printer(val); printer << ")";
                            }
                            break;
                            case c10::ScalarType::ComplexDouble:
                            {
                                double val;
                                val = at::real(toprint.index({ t,z,y,x })).item<double>();
                                printer << "("; value_printer(val); printer << ",";
                                val = at::imag(toprint.index({ t,z,y,x })).item<double>();
                                value_printer(val); printer << ")";
                            }
                            break;
                            default:
                                printer << toprint.index({ t,z,y,x }).toString();
                            }

                            closer(x, xlen, false, 0);
                        }
                        closer(y, ylen, true, 0);
                    }
                    closer(z, zlen, true, 1);
                }
                closer(t, tlen, true, 0);
            }

            return printer;
        }

		export std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor)
        {
            auto ret = tensor.sizes().vec();
            ret[0] = 1;
            return ret;
        }

		export template<typename T>
		std::vector<int64_t> argsort(const std::vector<T>& array) {
			std::vector<int64_t> indices(array.size());
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(),
				[&array](int64_t left, int64_t right) -> bool {
					// sort indices according to corresponding array element
					return array[left] < array[right];
				});

			return indices;
		}

		export at::ScalarType complex_type(at::ScalarType real_type, std::initializer_list<at::ScalarType> allowed_types)
        {
            at::ScalarType complex_type;
            if (real_type == at::ScalarType::Float)
                complex_type = at::ScalarType::ComplexFloat;
            else if (real_type == at::ScalarType::Double)
                complex_type = at::ScalarType::ComplexDouble;
            else if (real_type == at::ScalarType::Half)
                complex_type = at::ScalarType::ComplexHalf;
            else
                throw std::runtime_error("Type not implemented complex_type()");

            if (allowed_types.size() != 0) {
                for (auto& atype : allowed_types) {
                    if (complex_type == atype)
                        return complex_type;
                }
                throw std::runtime_error("complex_type converted to non allowable type");
            }
            return complex_type;
        }


		export at::ScalarType real_type(at::ScalarType complex_type, std::initializer_list<at::ScalarType> allowed_types)
        {
            at::ScalarType real_type;
            if (complex_type == at::ScalarType::ComplexFloat)
                real_type = at::ScalarType::Float;
            else if (complex_type == at::ScalarType::ComplexDouble)
                real_type = at::ScalarType::Double;
            else if (complex_type == at::ScalarType::ComplexHalf)
                real_type = at::ScalarType::Half;
            else
                throw std::runtime_error("Type not implemented complex_type()");

            if (allowed_types.size() != 0) {
                for (auto& atype : allowed_types) {
                    if (real_type == atype)
                        return real_type;
                }
                throw std::runtime_error("complex_type converted to non allowable type");
            }
            return real_type;
        }

		export template<typename T>
		std::vector<T> apply_permutation(const std::vector<T>& v, const std::vector<int64_t>& indices)
		{
			std::vector<T> v2(v.size());
			for (size_t i = 0; i < v.size(); i++) {
				v2[i] = v[indices[i]];
			}
			return v2;
		}

		export template<typename T>
		T future_catcher(std::future<T>& fut)
		{
			try {
				return fut.get();
			}
			catch (c10::Error& e) {
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
			catch (c10::Error& e) {
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
            catch (c10::Error& e) {
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
            catch (c10::Error& e) {
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

		export at::Tensor upscale_with_zeropad(const at::Tensor& input, const std::vector<int64_t>& newsize)
        {

            std::vector<at::indexing::TensorIndex> slices;
            for (int i = 0; i < input.ndimension(); ++i) {
                auto inpsize = input.size(i);
                if (inpsize > newsize[i]) {
                    throw std::runtime_error("Cannot upscale to smaller image in dim " + std::to_string(i));
                }
                slices.push_back(at::indexing::Slice(at::indexing::None, inpsize));
            }

            at::Tensor output = at::zeros(at::makeArrayRef(newsize), input.options());
            output.index_put_(at::makeArrayRef(slices), input);

            return output;
        }

		export at::Tensor upscale_with_zeropad(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize)
        {

            std::vector<at::indexing::TensorIndex> slices;
            for (int i = 0; i < input.ndimension(); ++i) {
                auto inpsize = input.size(i);
                if (inpsize > newsize[i]) {
                    throw std::runtime_error("Cannot upscale to smaller image in dim " + std::to_string(i));
                }
                slices.push_back(at::indexing::Slice(at::indexing::None, inpsize));
            }

            at::Tensor output = at::zeros(at::makeArrayRef(newsize), input.options());
            output.index_put_(at::makeArrayRef(slices), input);

            return output;
        }

		export at::Tensor resize(const at::Tensor& input, const std::vector<int64_t>& newsize)
        {

            std::vector<at::indexing::TensorIndex> slices;
            for (int i = 0; i < input.ndimension(); ++i) {
                slices.push_back(at::indexing::Slice(at::indexing::None, std::min(newsize[i], input.size(i))));
            }

            at::Tensor output = at::zeros(at::makeArrayRef(newsize), input.options());
            output.index_put_(at::makeArrayRef(slices), input);

            return output;
        }

		export at::Tensor resize(const at::Tensor& input, const at::ArrayRef<int64_t>& newsize)
        {

            std::vector<at::indexing::TensorIndex> slices;
            for (int i = 0; i < input.ndimension(); ++i) {
                slices.push_back(at::indexing::Slice(at::indexing::None, std::min(newsize[i], input.size(i))));
            }

            at::Tensor output = at::zeros(at::makeArrayRef(newsize), input.options());
            output.index_put_(at::makeArrayRef(slices), input);

            return output;
        }


    }
}


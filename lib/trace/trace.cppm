module;

#include "pch.hpp"
#include <pstl/pstl_config.h>
#include <ratio>
#include <type_traits>

export module trace;

import util;
import tensor;

namespace hasty {
	namespace trace {
		
		export using CompilationUnit = torch::jit::CompilationUnit;
    	export using CompilationModule = torch::jit::Module;


		// TENSOR PROTOTYPE

		export template<is_device D, is_tensor_type TT, size_t RANK>
		class tensor_prototype {
		public:

			using device_type_t = D;
			using tensor_type_t = TT;
			static constexpr std::integral_constant<size_t, RANK> size = {};

			tensor_prototype(std::string name)
				: _tracestr(name)
			{};

			std::string str() const {
				return _tracestr;
			}

		private:
			std::string _tracestr;
		};

		
		export template<typename T>
		concept is_tensor_prototype = requires(T t) {
			[]<is_device D, is_tensor_type TT, size_t RANK>(tensor_prototype<D,TT,RANK>&){}(t);
		};
		/*
		export template<typename T>
		concept is_tensor_prototype = requires(T t) {
			typename T::device_type_t;
			typename T::tensor_type_t;
			{ T::size() } -> std::convertible_to<size_t>;
			requires std::is_same_v<T,
				tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()>
			>;
		};
		*/


		template<typename T>
		struct tensor_prototype_conversion;

		template<is_tensor_prototype T>
		struct tensor_prototype_conversion<T> {
			tensor<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(T t);
		};

		template<is_tensor T>
		struct tensor_prototype_conversion<T> {
			tensor_prototype<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(T t);
		};

		export template<is_tensor_prototype T>
		using tensor_prototype_conversion_t = std::invoke_result_t<tensor_prototype_conversion<T>, T>;


		// TENSOR PROTOTYPE VECTOR

		export template<is_device D, is_tensor_type TT, size_t RANK>
		class tensor_prototype_vector {
		public:

			using device_type_t = D;
			using tensor_type_t = TT;
			static constexpr std::integral_constant<size_t, RANK> size = {};

			tensor_prototype_vector(const std::string& str)
				: _str(str)
			{}

			tensor_prototype_vector(const std::string& str, std::vector<tensor_prototype<D,TT,RANK>>&& tps)
				: _str(str), _tps(std::move(tps))
			{}

			std::string str() const {
				return _str;
			}

			std::string _str;
			std::vector<tensor_prototype<D,TT,RANK>> _tps;

			using value_type = tensor_prototype<D,TT,RANK>;
			using non_prototype_value_type = tensor<D,TT,RANK>;
		};

		
		export template<typename T>
		concept is_tensor_prototype_vector = requires(T t) {
			[]<is_device D, is_tensor_type TT, size_t RANK>(tensor_prototype_vector<D,TT,RANK>&){}(t);
		};
		/*
		export template<typename T>
		concept is_tensor_prototype_vector = requires(T t) {
			typename T::device_type_t;
			typename T::tensor_type_t;
			{ T::size() } -> std::convertible_to<size_t>;
			requires std::is_same_v<T,
				tensor_prototype_vector<typename T::device_type_t, typename T::tensor_type_t, T::size()>
			>;
		};
		*/

		template <typename T>
		struct tensor_prototype_vector_conversion;

		template<is_tensor_prototype_vector T>
		struct tensor_prototype_vector_conversion<T> {
			std::vector<typename T::non_prototype_value_type> operator()(T t);
		};

		template<is_tensor T>
		struct tensor_prototype_vector_conversion<T> {
			tensor_prototype_vector<typename T::device_type_t, typename T::tensor_type_t, T::size()> operator()(
				std::vector<T> t);
		};

		export template<is_tensor_prototype_vector T>
		using tensor_prototype_vector_conversion_t = std::invoke_result_t<tensor_prototype_vector_conversion<T>, T>;

		// ANY PROTOTYPE

		export template<typename T>
		concept is_any_tensor_prototype = is_tensor_prototype<T> || is_tensor_prototype_vector<T>;

		template<typename T, typename = void>
		struct any_tensor_prototype_conversion;

		template<typename T>
		struct any_tensor_prototype_conversion<T, std::enable_if_t<is_tensor_prototype<T>>> {
			using type = tensor_prototype_conversion_t<T>;
		};

		template<typename T>
		struct any_tensor_prototype_conversion<T, std::enable_if_t<is_tensor_prototype_vector<T>>> {
			using type = tensor_prototype_vector_conversion_t<T>;
		};

		export template<is_any_tensor_prototype T>
		using any_tensor_prototype_conversion_t = typename any_tensor_prototype_conversion<T>::type;

		export template<typename... U>
		struct trace_function;

		export template<typename... U>
		struct runnable_trace_function;

		export template<typename... U>
		struct trace_function_builder;

		export template<is_any_tensor_prototype... ReturnTt, is_any_tensor_prototype... InputTt>
		struct trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>> {
		private:
			std::string _funcname;
			std::unique_ptr<CompilationModule> _mod;
		public:

			using ReturnTraits = TupleTraits<ReturnTt...>;
			using InputTraits = TupleTraits<InputTt...>;

			trace_function(std::string_view funcname, std::unique_ptr<CompilationModule>&& mod)
				: _funcname(funcname),
				_mod(std::move(mod))
			{
			}

			const CompilationModule& get_module() const {
				return *_mod;
			}

			auto get_runnable() const 
				-> runnable_trace_function<typename ReturnTraits::Tuple, typename InputTraits::Tuple> 
			{
				if (_mod == nullptr) {
					throw std::runtime_error("Module not compiled");
				}
				return runnable_trace_function<
							typename ReturnTraits::Tuple, 
							typename InputTraits::Tuple
							>(*this);
			}

			template<is_tensor_or_vector_of_tensors ...Ts>
			auto run(Ts&&... tts) const -> std::tuple<any_tensor_prototype_conversion_t<ReturnTt>...> {
				
				//auto ttstup = std::make_tuple(tts...);
				using TsTraits = TupleTraits<Ts...>;

				for_sequence<sizeof...(Ts)>([](auto i) {
					using TS_TN = std::remove_reference_t<typename TsTraits::template Nth<i>>;
					using INPUT_TN = std::remove_reference_t<typename InputTraits::template Nth<i>>;
					
					using INPUT_DEVICE_T = typename INPUT_TN::device_type_t;
					using INPUT_TENSOR_T = typename INPUT_TN::tensor_type_t;
					
					if constexpr (is_tensor_prototype<INPUT_TN>) {
						static_assert(is_tensor<TS_TN>, "Input must be a tensor");
						using TS_DEVICE_T = typename TS_TN::device_type_t;
						using TS_TENSOR_T = typename TS_TN::tensor_type_t;
						static_assert(std::is_same_v<TS_DEVICE_T, INPUT_DEVICE_T>, "device_types must be the same");
						static_assert(std::is_same_v<TS_TENSOR_T, INPUT_TENSOR_T>, "tensor_types must be the same");
						static_assert(TS_TN::size() == INPUT_TN::size(), "Sizes must be the same");
					} else if constexpr (is_tensor_prototype_vector<INPUT_TN>) {
						static_assert(is_vector_of_tensors<TS_TN>, "Input must be a vector of tensors");
						using TS_VALUE_T = typename TS_TN::value_type;
						using TS_DEVICE_T = typename TS_VALUE_T::device_type_t;
						using TS_TENSOR_T = typename TS_VALUE_T::tensor_type_t;
						static_assert(std::is_same_v<TS_DEVICE_T, INPUT_DEVICE_T>, "device_types must be the same");
						static_assert(std::is_same_v<TS_TENSOR_T, INPUT_TENSOR_T>, "tensor_types must be the same");
						static_assert(TS_VALUE_T::size() == INPUT_TN::size(), "Sizes must be the same");
					} else {
						static_assert(false, "Invalid type");
					}

				});

				auto gettensorfunc = []<typename T>(T&& t) -> torch::jit::IValue {
					if constexpr (is_vector_of_tensors<T>) {
						std::vector<at::Tensor> rettensors;
						rettensors.reserve(t.size());
						for (auto& tt : t) {
							if (tt.ninstances() == 1) {
								rettensors.emplace_back(tt.decay_to_tensor());
							} else {
								rettensors.emplace_back(tt.get_tensor());
							}
						}
						return torch::jit::IValue(std::move(rettensors));
					}
					else if constexpr (is_tensor<T>) {
						if (t.ninstances() == 1) {
							return torch::jit::IValue(t.decay_to_tensor());
						}
						return torch::jit::IValue(t.get_tensor());
					} else if constexpr (!is_tensor_or_vector_of_tensors<T>) {
						static_assert(false, ""); // Why do this trigger?
					}
				};

				auto ttscopy = std::tuple(Ts(std::forward<Ts>(tts))...);
				
				std::vector<torch::jit::IValue> inputs;
				inputs.reserve(sizeof...(Ts));
				for_sequence<sizeof...(Ts)>([&](auto i) {
					inputs.emplace_back(gettensorfunc(std::get<i>(ttscopy)));
				});

				torch::jit::IValue ret_ivalue = (*_mod)(std::move(inputs));

				std::tuple<any_tensor_prototype_conversion_t<ReturnTt>...> rets;

				auto check_device_type = [&]<typename DEVICE_T>(TensorBackend t) {
					if constexpr(std::is_same_v<DEVICE_T, cpu_t>) {
						if (t.device().type() != TensorBackendDeviceType::CPU) {
							throw std::runtime_error("Tensor was not on CPU");
						}
					} else if constexpr(std::is_same_v<DEVICE_T, cuda_t>) {
						if (t.device().type() != TensorBackendDeviceType::CUDA) {
							throw std::runtime_error("Tensor was not on CUDA");
						}
					} else {
						static_assert(false, "Invalid device type");
					}
				};

				if (ret_ivalue.isTensor()) {
					auto& ret = std::get<0>(rets);
					using RET_T = std::remove_reference_t<decltype(ret)>;

					if (sizeof...(ReturnTt) > 1) {
						throw std::runtime_error("ReturnTt indicated multiple return values, but only one was returned");
					}
					if (!is_tensor_prototype<RET_T>) {
						throw std::runtime_error("ReturnTt was not a tensor prototype when return value was tensor");
					}

					TensorBackend ret_tensor = std::move(ret_ivalue.toTensor());
					check_device_type.template operator()<typename RET_T::device_type_t>(ret_tensor);
					ret.assign(span<RET_T::size()>(ret_tensor.sizes()), std::move(ret_tensor));
				} else if (ret_ivalue.isTuple()) {
					auto tuple_ptr = std::move(ret_ivalue.toTuple());
					auto& elements = tuple_ptr->elements();
					
					if (sizeof...(ReturnTt) != elements.size()) {
						throw std::runtime_error("Number of return values did not match number of elements in return type tuple");
					}

					for_sequence<sizeof...(ReturnTt)>([&](auto i) {
						auto& ret = std::get<i>(rets);
						using RET_T = std::remove_reference_t<decltype(ret)>;

						auto& tup_ret = elements[i];

						if constexpr(is_vector_of_tensors<RET_T>) {
							if (!tup_ret.isTensorList()) {
								throw std::runtime_error("Expected list, got something else");
							}
							auto tensor_list = tup_ret.toTensorList();
							ret.resize(tensor_list.size());

							for (int j = 0; j < tensor_list.size(); ++j) {
								TensorBackend ret_tensor = std::move(tensor_list[j]);
								check_device_type.template operator()<typename RET_T::value_type::device_type_t>(ret_tensor);
								ret[j].assign(span<RET_T::value_type::size()>(ret_tensor.sizes()), std::move(ret_tensor));
							}
						} else if constexpr(is_tensor<RET_T>) {
							if (!tup_ret.isTensor()) {
								throw std::runtime_error("Expected tensor, got something else");
							}

							TensorBackend ret_tensor = std::move(tup_ret.toTensor());
							check_device_type.template operator()<typename RET_T::device_type_t>(ret_tensor);
							ret.assign(span<RET_T::size()>(ret_tensor.sizes()), std::move(ret_tensor));
						} else if constexpr(!is_tensor_or_vector_of_tensors<RET_T>) {
							/*
							using TP = tensor_prototype_vector<cuda_t, f32_t, 2>;
							using TR = any_tensor_prototype_conversion_t<TP>;
							using H0 = TR::something;
							using H = RET_T::something;
							*/
							static_assert(false, ""); // This should never happen...
						}
					});
				} else if (ret_ivalue.isNone()) {
					if (sizeof...(ReturnTt) != 0) {
						throw std::runtime_error("Expected return values, got None");
					}
				} else {
					throw std::runtime_error("Invalid return type");
				}
				
				return rets;
			}

		};

		export template<is_any_tensor_prototype... ReturnTt, is_any_tensor_prototype... InputTt>
		struct runnable_trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>> {
		private:
			const trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>>& _trace_func;
		public:

			runnable_trace_function(const trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>>& trace_func)
				: _trace_func(trace_func)
			{}

			template<is_tensor_or_vector_of_tensors ...Ts>
			auto run(Ts&&... tts) const -> std::tuple<any_tensor_prototype_conversion_t<ReturnTt>...> {
				return _trace_func.run(std::forward<Ts>(tts)...);
			}
		};

		export template<is_any_tensor_prototype... ReturnTt, is_any_tensor_prototype... InputTt>
		struct trace_function_builder<std::tuple<ReturnTt...>, std::tuple<InputTt...>> {
		private:
			std::string _funcname;
			std::string _tracestr;
			std::string _forwardname;
			std::string _compiled;

			std::unique_ptr<CompilationModule> _mod;

			std::tuple<InputTt...> _trace_tensors;
		public:
			using ReturnTraits = TupleTraits<ReturnTt...>;
			using InputTraits = TupleTraits<InputTt...>;

			trace_function_builder(
				std::string_view funcname,
				std::string_view code,
				InputTt&&... tts
			)
				: 
				_funcname(funcname),
				_tracestr(code),
				_trace_tensors(
					std::forward<InputTt>(tts)...
				)
			{
				reset();
			}

			trace_function_builder(
				std::string_view funcname,
				std::string_view code,
				const InputTt&... tts
			)
				: 
				_funcname(funcname),
				_tracestr(code),
				_trace_tensors(tts...)
			{
				reset();
			}
			

			void reset() {
				_tracestr = "";
				_mod = nullptr;

				std::string variables = "self, " + for_sequence<sizeof...(InputTt)>([this](auto i, std::string& currentstr){
					auto& itt = std::get<i>(_trace_tensors);

					currentstr += itt.str() + ": ";

					if constexpr(is_tensor_prototype<decltype(itt)>) {
						currentstr += "Tensor";
					} else if constexpr(is_tensor_prototype_vector<decltype(itt)>) {
						currentstr += "List[Tensor]";
					} else if constexpr(!is_any_tensor_prototype<decltype(itt)>) {
						static_assert(false, "Invalid type");
					}

					if constexpr(i < sizeof...(InputTt) - 1) {
						currentstr += ",";
					}
				}, std::string(""));

				if constexpr(sizeof...(ReturnTt) == 0) {
					_tracestr = std::format("def forward({}):", variables);
					return;
				}

				std::string returnvars = "";
				if constexpr(sizeof...(ReturnTt) >= 1) {

					returnvars += " -> Tuple[";

					returnvars += for_sequence<sizeof...(ReturnTt)>([this](auto i, std::string& currentstr){
						
						using RET_T = typename ReturnTraits::template Nth<i>;

						if constexpr(is_tensor_prototype<RET_T>) {
							currentstr += "Tensor";
						} else if constexpr(is_tensor_prototype_vector<RET_T>) {
							currentstr += "List[Tensor]";
						} else if constexpr(!is_any_tensor_prototype<RET_T>) {
							static_assert(false, "Invalid type");
						}

						if constexpr((i < sizeof...(ReturnTt) - 1) || (sizeof...(ReturnTt) == 1)) {
							currentstr += ",";
						}
					}, std::string(""));

					returnvars += "]";

				}

				_forwardname = std::format("def forward({}){}:", variables, returnvars);

			}

			const std::string& uncompiled_str() const {
				return _tracestr;
			}

			const std::string& compiled_str() const {
				return _compiled;
			}

			void compile() {
				if (_mod != nullptr) {
					return;
				}

				_compiled = util::replace_line(_tracestr, "FORWARD_ENTRYPOINT", _forwardname);

				_mod = std::make_unique<CompilationModule>(_funcname);
				_mod->define(_compiled);

				*_mod = torch::jit::freeze(*_mod);
				*_mod = torch::jit::optimize_for_inference(*_mod);
			}

			const CompilationModule& get_module() const {
				if (_mod == nullptr) {
					throw std::runtime_error("Module not compiled");
				}
				return *_mod;
			}

			std::unique_ptr<CompilationModule> release_module() {
				return std::move(_mod);
			}

			auto build_trace_function()	
				-> trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>> 
			{
				if (_mod == nullptr) {
					throw std::runtime_error("Module not compiled");
				}
				return trace_function<std::tuple<ReturnTt...>, std::tuple<InputTt...>>(_funcname, std::move(_mod));
			}

		};

		export template<is_any_tensor_prototype... ReturnTt>
		struct trace_function_builder_factory {

			template<is_any_tensor_prototype... InputTt>
			static auto make(std::string_view funcname, std::string_view code, InputTt&... tts)
				-> trace_function_builder<std::tuple<ReturnTt...>, std::tuple<std::decay_t<InputTt>...>> 
			{
				return trace_function_builder<std::tuple<ReturnTt...>, std::tuple<std::decay_t<InputTt>...>>(
					funcname, code, std::forward<const InputTt>(tts)...
				);
			}

			template<is_any_tensor_prototype... InputTt>
			static auto make_unique(std::string_view funcname, std::string_view code, InputTt&&... tts)
				-> uptr<trace_function_builder<std::tuple<ReturnTt...>, std::tuple<std::decay_t<InputTt>...>>>
			{
				return std::make_unique<trace_function_builder<std::tuple<ReturnTt...>, std::tuple<std::decay_t<InputTt>...>>>(
					funcname, code, std::forward<InputTt>(tts)...
				);
			}
		};


		export template<typename T>
		concept is_trace_function = requires {
			std::derived_from<T, hasty::trace::trace_function<typename T::ReturnTraits::Tuple, typename T::InputTraits::Tuple>>;
		};

		export template<typename T>
		concept is_runnable_trace_function = requires {
			std::derived_from<T, hasty::trace::runnable_trace_function<typename T::ReturnTraits::Tuple, typename T::InputTraits::Tuple>>;
		};

	}


}

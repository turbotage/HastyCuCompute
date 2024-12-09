module;

#include "pch.hpp"

export module util:funcs;

//import pch;

namespace hasty {

	namespace util {

		export std::size_t replace_all(std::string& inout, std::string_view what, std::string_view with)
		{
			std::size_t count{};
			for (std::string::size_type pos{};
				inout.npos != (pos = inout.find(what.data(), pos, what.length()));
				pos += with.length(), ++count) {
				inout.replace(pos, what.length(), with.data(), with.length());
			}
			return count;
		}

		export std::size_t remove_all(std::string& inout, std::string_view what) {
			return replace_all(inout, what, "");
		}

		export template<typename KeyType, typename HashFunc = std::hash<KeyType>>
		concept Hashable = std::regular_invocable<HashFunc, KeyType>
		&& std::convertible_to<std::invoke_result_t<HashFunc, KeyType>, std::size_t>;

		export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
			inline std::size_t hash_combine(const std::size_t& seed, const KeyType& v)
		{
			HashFunc hasher;
			std::size_t ret = seed;
			ret ^= hasher(v) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
			return ret;
		}

		export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
			std::size_t hash_combine(const std::vector<KeyType>& hashes)
		{
			std::size_t ret = hashes.size() > 0 ? hashes.front() : throw std::runtime_error("Can't hash_combine an empty vector");
			for (int i = 1; i < hashes.size(); ++i) {
				ret = hash_combine(ret, hashes[i]);
			}
			return ret;
		}

		export std::string to_lower_case(const std::string& str) {
			std::string ret = str;
			std::transform(ret.begin(), ret.end(), ret.begin(), [](unsigned char c) { return std::tolower(c); });
			return ret;
		}

		export std::string add_whitespace_until(const std::string& str, int until) {
			if (str.size() > until) {
				return std::string(str.begin(), str.begin() + until);
			}

			std::string ret = str;
			ret.reserve(until);
			for (int i = str.size(); i <= until; ++i) {
				ret += ' ';
			}
			return ret;
		}

		export std::string add_after_newline(const std::string& str, const std::string& adder, bool add_start = true)
		{
			std::string ret = str;
			if (add_start) {
				ret.insert(0, adder);
			}
			for (int i = 0; i < ret.size(); ++i) {
				if (ret[i] == '\n') {
					if (i + 1 > ret.size())
						return ret;
					ret.insert(i + 1, adder);
					i += adder.size() + 2;
				}
			}
			return ret;
		}

		export std::string add_line_numbers(const std::string& str, int max_number_length = 5) {
			int until = max_number_length;
			std::string ret = str;
			ret.insert(0, util::add_whitespace_until(std::to_string(1), until) + "\t|");
			int k = 2;
			for (int i = until; i < ret.size(); ++i) {
				if (ret[i] == '\n') {
					if (i + 1 > ret.size())
						return ret;
					ret.insert(i + 1, util::add_whitespace_until(std::to_string(k), until) + "\t|");
					i += until + 2;
					++k;
				}
			}
			return ret;
		}

		export std::string remove_whitespace(const std::string& str) {
			std::string ret = str;
			ret.erase(std::remove_if(ret.begin(), ret.end(),
				[](char& c) {
					return std::isspace<char>(c, std::locale::classic());
				}),
				ret.end());
			return ret;
		}

		export template<typename Container> requires std::ranges::range<Container>
		bool container_contains(const Container& c, typename Container::const_reference v)
		{
			return std::find(c.begin(), c.end(), v) != c.end();
		}

		export std::string hash_string(size_t num, size_t len)
		{
			std::string numstr = std::to_string(num);
			static std::string lookup("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
			std::string ret;
			for (int i = 0; i < numstr.size(); i += 2) {
				std::string substr = numstr.substr(i, 2);
				int index = std::atoi(substr.c_str());
				index = index % lookup.size();
				ret += lookup[index];
			}
			return ret.substr(0, len);
		}

		export std::string stupid_compress(std::uint64_t num)
		{
			std::string basec = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
			std::string ret;

			auto powlam = [](std::uint64_t base, std::uint32_t exponent) {
				uint64_t retnum = 1;
				for (int i = 0; i < exponent; ++i) {
					retnum *= base;
				}
				return retnum;
			};

			uint64_t base = std::numeric_limits<uint64_t>::max();
			uint64_t c = (uint64_t)num / base;
			uint64_t rem = num % base;

			for (int i = 10; i >= 0; --i) {
				base = powlam(basec.size(), i);
				c = (uint64_t)num / base;
				rem = num % base;

				if (c > 0)
					ret += basec[c];
				num = rem;
			}

			return ret;
		}

		export template<typename T>
		std::vector<T> vec_concat(const std::vector<T>& left, const std::vector<T>& right)
		{
			std::vector<T> ret;
			ret.reserve(left.size() + right.size());
			ret.insert(ret.end(), left.begin(), left.end());
			ret.insert(ret.end(), right.begin(), right.end());
			return ret;
		}

		export void add_n_str(std::string& str, const std::string& adder, int n)
		{
			for (int i = 0; i < n; ++i) {
				str += adder;
			}
		}

		export std::vector<std::int64_t> broadcast_tensor_shapes(const std::vector<std::int64_t>& shape1, const std::vector<std::int64_t>& shape2)
		{
			if (shape1.size() == 0 || shape2.size() == 0)
				throw std::runtime_error("shapes must have atleast one dimension to be broadcastable");

			auto& small = (shape1.size() > shape2.size()) ? shape2 : shape1;
			auto& big = (shape1.size() > shape2.size()) ? shape1 : shape2;

			std::vector<int64_t> ret(big.size());

			auto retit = ret.rbegin();
			auto smallit = small.rbegin();
			for (auto bigit = big.rbegin(); bigit != big.rend(); ) {
				if (smallit != small.rend()) {
					if (*smallit == *bigit) {
						*retit = *bigit;
					}
					else if (*smallit > *bigit && *bigit == 1) {
						*retit = *smallit;
					}
					else if (*bigit > *smallit && *smallit == 1) {
						*retit = *bigit;
					}
					else {
						throw std::runtime_error("shapes where not broadcastable");
					}
					++smallit;
				}
				else {
					*retit = *bigit;
				}

				++bigit;
				++retit;
			}

			return ret;
		}

		export template<typename RetType, typename InpType>
		std::vector<RetType> vector_cast(const std::vector<InpType>& input)
		{
			std::vector<RetType> ret;
			ret.reserve(input.size());
			std::for_each(input.begin(), input.end(), [&](const InpType& i) {
				ret.push_back(static_cast<RetType>(i));
			});
			return ret;
		}

		

	}

}


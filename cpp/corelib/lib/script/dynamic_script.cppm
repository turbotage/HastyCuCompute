module;

export module dynamic_script;

import std;
import util;
import torch_base;
import generic_value;

namespace hasty {
namespace script {

class runnable_dynamic_script {
public:

	runnable_dynamic_script(
		std::string&& funcname,
		std::string&& code,
		std::string&& return_type_string,
		std::vector<std::pair<std::string,std::string>>&& input_type_strings
	) :
		_funcname(std::move(funcname)),
		_code(std::move(code)),
		_return_type_string(std::move(return_type_string)),
		_input_type_strings(std::move(input_type_strings))
	{

	}

	generic_value run(std::vector<generic_value>&& inputs) {
		if (_mod == nullptr) {
			compile();
		}

		std::vector<htorch::jit::IValue> ivalue_inputs;
		ivalue_inputs.reserve(inputs.size());

		for (auto& gv : inputs) {
			ivalue_inputs.push_back(generic_to_ivalue(std::move(gv)));
		}

		htorch::jit::IValue ret_ivalue = _mod->forward(ivalue_inputs);

		return ivalue_to_generic(std::move(ret_ivalue));
	}

	void compile() {
		_mod = nullptr;
		
		std::string inputvars = "self";

		for (const auto& [name, type] : _input_type_strings) {
			inputvars += ", " + name + ": " + type;
		}

		std::string returnvars = " -> " + _return_type_string;

		_forwardname = std::format("def foward({}){}:", inputvars, returnvars);
		_compiled_str = util::replace_line(_code, "FORWARD_ENTRYPOINT", _forwardname);

		if (_mod != nullptr) {
			throw std::runtime_error("Script already compiled");
		}

		_mod = std::make_unique<htorch::jit::Module>(_funcname);
		_mod->define(_compiled_str);

		if (_freeze) {
			*_mod = htorch::jit::freeze(*_mod);
		}
		if (_optimize_for_inference) {
			*_mod = htorch::jit::optimize_for_inference(*_mod);
		}
	}

	void freeze(bool v) {
		_freeze = v;
	}

	void optimize_for_inference(bool v) {
		_optimize_for_inference = v;
	}


private:
	std::string _funcname;
	std::string _code;
	std::string _forwardname;
	std::string _compiled_str;
	std::string _return_type_string;
	std::vector<std::pair<std::string,std::string>> _input_type_strings;

	bool _freeze = true;
	bool _optimize_for_inference = true;

	uptr<htorch::jit::Module> _mod;
};

}
}
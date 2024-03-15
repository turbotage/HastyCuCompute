module;

#include "pch.hpp"
#include <highfive/H5Easy.hpp>

export module hdf5;

import util;

HighFive::CompoundType make_complex_float() {
	return {
		{"r", HighFive::AtomicType<float>{}},
		{"i", HighFive::AtomicType<float>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(std::complex<float>, make_complex_float);

HighFive::CompoundType make_complex_double() {
	return {
		{"r", HighFive::AtomicType<double>{}},
		{"i", HighFive::AtomicType<double>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(std::complex<double>, make_complex_double);

namespace hasty {

	export at::Tensor import_tensor(const std::string& filepath, const std::string& dataset)
	{
		HighFive::File file(filepath, HighFive::File::ReadOnly);
		HighFive::DataSet dset = file.getDataSet(dataset);

		HighFive::DataType dtype = dset.getDataType();
		std::string dtype_str = dtype.string();
		size_t dtype_size = dtype.getSize();
		std::vector<int64_t> dims = hasty::util::vector_cast<int64_t>(dset.getDimensions());
		size_t nelem = dset.getElementCount();

		if (dtype_str == "Float32") {
			std::vector<float> data(nelem);
			dset.read(data.data());
			return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::Float).detach().clone();
		}
		else if (dtype_str == "Float64") {
			std::vector<double> data(nelem);
			dset.read(data.data());
			return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::Double).detach().clone();
		}
		else if (dtype_str == "Compound64") {
			HighFive::CompoundType ctype(std::move(dtype));
			auto members = ctype.getMembers();
			if (members.size() != 2)
				throw std::runtime_error("HighFive reported an Compound64 type");
			std::vector<std::complex<float>> data(nelem);
			dset.read(data.data());
			return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::ComplexFloat).detach().clone();
		}
		else if (dtype_str == "Compound128") {
			HighFive::CompoundType ctype(std::move(dtype));
			auto members = ctype.getMembers();
			if (members.size() != 2)
				throw std::runtime_error("HighFive reported an Compound64 type");
			std::vector<std::complex<double>> data(nelem);
			dset.read(data.data());
			return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::ComplexDouble).detach().clone();
		}
		else {
			throw std::runtime_error("disallowed dtype");
		}

	}

    export void export_tensor(const at::Tensor& tensor, const std::string& filepath, const std::string& dataset)
    {
        HighFive::File file(filepath, HighFive::File::Overwrite);

		if (!tensor.is_contiguous()) {
			throw std::runtime_error("tensor must be contiguous");
		}

		std::vector<hsize_t> dims(tensor.sizes().begin(), tensor.sizes().end());
		at::ScalarType dtype = tensor.scalar_type();
		auto dataspace = HighFive::DataSpace(dims);

		if (dtype == at::ScalarType::Float) {
        	HighFive::DataSet dset = file.createDataSet<float>(dataset, dataspace);
			dset.write(static_cast<float*>(tensor.data_ptr()));
		} else if (dtype == at::ScalarType::Double) {
			HighFive::DataSet dset = file.createDataSet<double>(dataset, dataspace);
			dset.write(static_cast<double*>(tensor.data_ptr()));
		} else if (dtype == at::ScalarType::ComplexFloat) {
			HighFive::DataSet dset = file.createDataSet<std::complex<float>>(dataset, dataspace);
			dset.write(static_cast<std::complex<float>*>(tensor.data_ptr()));
		} else if (dtype == at::ScalarType::ComplexDouble) {
			HighFive::DataSet dset = file.createDataSet<std::complex<double>>(dataset, dataspace);
			dset.write(static_cast<std::complex<double>*>(tensor.data_ptr()));
		}

    }

}

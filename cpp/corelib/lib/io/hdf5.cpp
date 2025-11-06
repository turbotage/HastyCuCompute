module;

#include "pch.hpp"
#include <highfive/highfive.hpp>
#include <H5Cpp.h>

module hdf5;

struct Complex64 {
    float real;
    float imag;
};

HighFive::CompoundType make_complex_float_struct() {
	return {
		{"real", HighFive::AtomicType<float>{}},
		{"imag", HighFive::AtomicType<float>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(Complex64, make_complex_float_struct);

struct Complex128 {
    double real;
    double imag;
};

HighFive::CompoundType make_complex_double_struct() {
	return {
		{"real", HighFive::AtomicType<double>{}},
		{"imag", HighFive::AtomicType<double>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(Complex128, make_complex_double_struct);


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



hat::Tensor import_tensor(const std::string& filepath, const std::string& dataset)
{
    HighFive::File file(filepath, HighFive::File::ReadOnly);
    HighFive::DataSet dset = file.getDataSet(dataset);

    HighFive::DataType dtype = dset.getDataType();
    std::string dtype_str = dtype.string();
    size_t dtype_size = dtype.getSize();
    std::vector<int64_t> dims = hasty::util::vector_cast<int64_t>(dset.getDimensions());
    size_t nelem = dset.getElementCount();
    //auto flat_dset = dset.reshapeMemSpace({nelem});
    dtype.getClass();

    if (dtype_str == "Float32") {
        std::vector<float> data(nelem);
        //flat_dset.read(data);
        //dset.read<float>(data.data());
        dset.read_raw<float>(data.data());
        return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::Float).detach().clone();
    }
    else if (dtype_str == "Float64") {
        std::vector<double> data(nelem);
        //flat_dset.read(data);
        //dset.read<double>(data.data());
        dset.read_raw<double>(data.data());
        return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::Double).detach().clone();
    }
    else if (dtype_str == "Compound64") {
        HighFive::CompoundType ctype(std::move(dtype));
        auto members = ctype.getMembers();
        if (members.size() != 2)
            throw std::runtime_error("HighFive reported an Compound64 type");
        std::vector<std::complex<float>> data(nelem);
        //flat_dset.read(data);
        //dset.read<std::complex<float>>(data.data());
        dset.read_raw<std::complex<float>>(data.data());
        return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::ComplexFloat).detach().clone();
    }
    else if (dtype_str == "Compound128") {
        HighFive::CompoundType ctype(std::move(dtype));
        auto members = ctype.getMembers();
        if (members.size() != 2)
            throw std::runtime_error("HighFive reported an Compound128 type");
        std::vector<std::complex<double>> data(nelem);
        //flat_dset.read(data);
        //dset.read<std::complex<double>>(data.data());
        dset.read_raw<std::complex<double>>(data.data());
        return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::ComplexDouble).detach().clone();
    }
    else {
        throw std::runtime_error("disallowed dtype");
    }

}



namespace hasty {

    auto import_tensors(const HighFive::DataSet& dataset) ->
        std::variant<hat::Tensor, std::vector<hat::Tensor>>
    {

        HighFive::DataType dtype = dataset.getDataType();
        std::string dtype_str = dtype.string();
        size_t dtype_size = dtype.getSize();
        std::vector<int64_t> dims = hasty::util::vector_cast<int64_t>(dataset.getDimensions());
        size_t nelem = dataset.getElementCount();
        //auto flat_dset = dataset.reshapeMemSpace({nelem});
        
        if (dtype_str == "Float32") {
            std::vector<float> data(nelem);
            //flat_dset.read(data);
            //dataset.read<float>(data.data());
            dataset.read_raw<float>(data.data());
            return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::Float).detach().clone();
        }
        else if (dtype_str == "Float64") {
            std::vector<double> data(nelem);
            //flat_dset.read(data);
            //dataset.read<double>(data.data());
            dataset.read_raw<double>(data.data());
            return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::Double).detach().clone();
        }
        else if (dtype_str == "Compound64") {
            HighFive::CompoundType ctype(std::move(dtype));
            auto members = ctype.getMembers();
            if (members.size() != 2)
                throw std::runtime_error("HighFive reported an Compound64 type");

            std::vector<Complex64> data(nelem);
            //flat_dset.read<Complex64>(data);
            //dataset.read<std::complex<float>>(data.data());
            dataset.read_raw<Complex64>(data.data());
            return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::ComplexFloat).detach().clone();
        }
        else if (dtype_str == "Compound128") {
            HighFive::CompoundType ctype(std::move(dtype));
            auto members = ctype.getMembers();
            if (members.size() != 2)
                throw std::runtime_error("HighFive reported an Compound64 type");
            std::vector<Complex128> data(nelem);
            //flat_dset.read(data);
            //dataset.read<std::complex<double>>(data.data());
            dataset.read_raw<Complex128>(data.data());
            return hat::from_blob(data.data(), hat::makeArrayRef(dims), hat::ScalarType::ComplexDouble).detach().clone();
        }
        else {
            throw std::runtime_error("disallowed dtype");
        }

    }

    auto import_tensor(const std::string& filepath, const std::string& dataset) -> 
        std::variant<hat::Tensor, std::vector<hat::Tensor>> 
    {
        HighFive::File file(filepath, HighFive::File::ReadOnly);
        HighFive::DataSet dset = file.getDataSet(dataset);

        return import_tensors(dset);
    }


    auto import_tensors(const std::string& filepath, const std::optional<std::vector<std::regex>>& matchers) ->
        std::unordered_map<std::string, std::variant<hat::Tensor, std::vector<hat::Tensor>>>
    {
        HighFive::File file(filepath, HighFive::File::ReadOnly);
        std::unordered_map<std::string, std::variant<hat::Tensor, std::vector<hat::Tensor>>> tensors;

        std::function<void(const std::string&, const HighFive::Group&)> grouplam;

        grouplam = [&](const std::string& groupname, const HighFive::Group& group) {
            std::vector<std::string> names;

            names = group.listObjectNames(HighFive::IndexType(H5_INDEX_NAME));
            for (const auto& name : names) {
                HighFive::ObjectType objtype = group.getObjectType(name);
                if (objtype == HighFive::ObjectType::Dataset) {
                    auto datasetname = groupname + "/" + name;

                    // If matcher regexes were provided the dataset path must match at least one of them
                    if (matchers.has_value()) {
                        bool matched = false;

                        auto& matchersval = *matchers;
                        for (auto it = matchersval.begin(); it != matchersval.end(); ++it) {
                            if (std::regex_match(datasetname, *it)) {
                                matched = true;
                                break;
                            }
                        }
                        // Skip this dataset if it didn't match any of the regexes
                        if (!matched) {
                            continue;
                        }
                    }

                    // No matcher regexes were provided or the dataset path matched at least one of them, we can insert
                    tensors.insert({datasetname, import_tensors(group.getDataSet(name)) });

                } else if (objtype == HighFive::ObjectType::Group) {
                    grouplam(groupname + "/" + name, group.getGroup(name));
                } else {
                    throw std::runtime_error("unsupported object type");
                }
            }
        };

        grouplam("", file.getGroup("/"));

        return tensors;
    }


    void export_tensor(const hat::Tensor& tensor, const std::string& filepath, const std::string& dataset)
    {
        HighFive::File file(filepath, HighFive::File::Overwrite);

        if (!tensor.is_contiguous()) {
            throw std::runtime_error("tensor must be contiguous");
        }

        std::vector<size_t> dims(tensor.sizes().begin(), tensor.sizes().end());
        hat::ScalarType dtype = tensor.scalar_type();
        auto dataspace = HighFive::DataSpace(dims);

        if (dtype == hat::ScalarType::Float) {
            HighFive::DataSet dset = file.createDataSet<float>(dataset, dataspace);
            dset.write_raw(static_cast<float*>(tensor.data_ptr()));
        } else if (dtype == hat::ScalarType::Double) {
            HighFive::DataSet dset = file.createDataSet<double>(dataset, dataspace);
            dset.write_raw(static_cast<double*>(tensor.data_ptr()));
        } else if (dtype == hat::ScalarType::ComplexFloat) {
            HighFive::DataSet dset = file.createDataSet<std::complex<float>>(dataset, dataspace);
            dset.write_raw(static_cast<std::complex<float>*>(tensor.data_ptr()));
        } else if (dtype == hat::ScalarType::ComplexDouble) {
            HighFive::DataSet dset = file.createDataSet<std::complex<double>>(dataset, dataspace);
            dset.write_raw(static_cast<std::complex<double>*>(tensor.data_ptr()));
        }

    }

} // namespace hasty
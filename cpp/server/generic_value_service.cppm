export module generic_value_service;

import <grpcpp/grpcpp.h>;
import "protos/generic_value.grpc.pb.h";
// ...import interface_cache.cppm and other necessary modules...

using grpc::ServerContext;
using grpc::ServerReader;
using grpc::Status;
using hastycu::GenericValue;
using hastycu::GenericValueAck;
using hastycu::GenericValueService;

export class GenericValueServiceImpl final : public GenericValueService::Service {
	Status PushGenericValue(ServerContext* context, ServerReader<GenericValue>* reader, GenericValueAck* ack) override {
		std::vector<std::string> values;
		GenericValue value;
		while (reader->Read(&value)) {
			values.push_back(value.data());
		}
		// Add to interface cache and get unique id
		std::string unique_id = add_to_interface_cache(values); // Implement this using interface_cache.cppm
		ack->set_unique_id(unique_id);
		return Status::OK;
	}
};

export module stream_service;

import <grpcpp/grpcpp.h>;
import "protos/stream_service.grpc.pb.h";
// ...import interface_cache.cppm and other necessary modules...

using grpc::ServerContext;
using hastycu::StreamService;
using hastycu::StreamInitRequest;
using hastycu::StreamInitResponse;
using hastycu::StreamChunk;
using hastycu::StreamChunkAck;

export class StreamServiceImpl final : public StreamService::Service {
	Status InitStream(ServerContext* context, const StreamInitRequest* request, StreamInitResponse* response) override {
		// Generate a unique stream/session id and store metadata
		std::string stream_id = create_stream_session(request->operation_type(), request->metadata());
		response->set_stream_id(stream_id);
		return Status::OK;
	}

	Status SendChunk(ServerContext* context, const StreamChunk* chunk, StreamChunkAck* ack) override {
		// Store chunk data for the given stream_id and chunk_index
		bool ok = store_chunk(chunk->stream_id(), chunk->chunk_index(), chunk->flags(), chunk->data());
		ack->set_success(ok);
		ack->set_message(ok ? "Chunk received" : "Chunk error");
		return Status::OK;
	}
};

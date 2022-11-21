// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/clients/c++/perf_client/triton_client_wrapper.h"

#include "src/clients/c++/examples/json_utils.h"
#include <random>
//==============================================================================

nic::Error
TritonClientFactory::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<nic::Headers> http_headers, const bool verbose,
    std::shared_ptr<TritonClientFactory>* factory)
{
  factory->reset(new TritonClientFactory(url, protocol, http_headers, verbose));
  return nic::Error::Success;
}

nic::Error
TritonClientFactory::CreateTritonClient(
    std::unique_ptr<TritonClientWrapper>* client)
{
  RETURN_IF_ERROR(TritonClientWrapper::Create(
      url_, protocol_, http_headers_, verbose_, client));
  return nic::Error::Success;
}

//==============================================================================

nic::Error
TritonClientWrapper::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<nic::Headers> http_headers, const bool verbose,
    std::unique_ptr<TritonClientWrapper>* triton_client_wrapper)
{
  triton_client_wrapper->reset(new TritonClientWrapper(protocol, http_headers));
  if (protocol == ProtocolType::HTTP) {
    RETURN_IF_ERROR(nic::InferenceServerHttpClient::Create(
        &((*triton_client_wrapper)->client_.http_client_), url, verbose));
  } else {
    RETURN_IF_ERROR(nic::InferenceServerGrpcClient::Create(
        &((*triton_client_wrapper)->client_.grpc_client_), url, verbose));
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string metadata;
    RETURN_IF_ERROR(client_.http_client_->ModelMetadata(
        &metadata, model_name, model_version, *http_headers_));
    RETURN_IF_ERROR(nic::ParseJson(model_metadata, metadata));
  } else {
    return nic::Error("gRPC can not return model metadata as json");
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelMetadata(
    ni::ModelMetadataResponse* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->ModelMetadata(
        model_metadata, model_name, model_version, *http_headers_));
  } else {
    return nic::Error("HTTP can not return model metadata as protobuf message");
  }

  return nic::Error::Success;
}


nic::Error
TritonClientWrapper::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string config;
    RETURN_IF_ERROR(client_.http_client_->ModelConfig(
        &config, model_name, model_version, *http_headers_));
    RETURN_IF_ERROR(nic::ParseJson(model_config, config));
  } else {
    return nic::Error("gRPC can not return model config as json");
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelConfig(
    ni::ModelConfigResponse* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->ModelConfig(
        model_config, model_name, model_version, *http_headers_));
  } else {
    return nic::Error("HTTP can not return model config as protobuf message");
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::Infer(
nic::InferResult** result, const nic::InferOptions& options,
const std::vector<nic::InferInput*>& inputs,
const std::vector<const nic::InferRequestedOutput*>& outputs)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->Infer(
					result, options, inputs, outputs, *http_headers_));
	} else {
		
		// rand() % (last value - first value + 1) + first value	
		//int randindex = rand() % (3-0+1) + 0;
		std::vector<std::vector<int64_t>> shapes;
		std::vector<float> index_set;

		std::string model_name = options.model_name_;

		if(model_name == "b0")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,32,112,112};
			std::vector<int64_t> shape3{1,32,112,112};
			std::vector<int64_t> shape4{1,16,112,112};
			std::vector<int64_t> shape5{1,24,56,56};
			std::vector<int64_t> shape6{1,24,56,56};
			std::vector<int64_t> shape7{1,40,28,28};
			std::vector<int64_t> shape8{1,40,28,28};
			std::vector<int64_t> shape9{1,80,14,14};
			std::vector<int64_t> shape10{1,80,14,14};
			std::vector<int64_t> shape11{1,80,14,14};
			std::vector<int64_t> shape12{1,112,14,14};
			std::vector<int64_t> shape13{1,112,14,14};
			std::vector<int64_t> shape14{1,112,14,14};
			std::vector<int64_t> shape15{1,192,7,7};
			std::vector<int64_t> shape16{1,192,7,7};
			std::vector<int64_t> shape17{1,192,7,7};
			std::vector<int64_t> shape18{1,192,7,7};
			std::vector<int64_t> shape19{1,320,7,7};
			std::vector<int64_t> shape20{1,1280,7,7};
			std::vector<int64_t> shape21{1,1280,7,7};
			std::vector<int64_t> shape22{1,1280,1,1};
			std::vector<int64_t> shape23{1,1280};
			std::vector<int64_t> shape24{1,1280};
			std::vector<int64_t> shape25{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);

		}
		else if(model_name == "b1")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,32,112,112};
			std::vector<int64_t> shape3{1,32,112,112};
			std::vector<int64_t> shape4{1,16,112,112};
			std::vector<int64_t> shape5{1,16,112,112};
			std::vector<int64_t> shape6{1,24,56,56};
			std::vector<int64_t> shape7{1,24,56,56};
			std::vector<int64_t> shape8{1,24,56,56};
			std::vector<int64_t> shape9{1,40,28,28};
			std::vector<int64_t> shape10{1,40,28,28};
			std::vector<int64_t> shape11{1,40,28,28};
			std::vector<int64_t> shape12{1,80,14,14};
			std::vector<int64_t> shape13{1,80,14,14};
			std::vector<int64_t> shape14{1,80,14,14};
			std::vector<int64_t> shape15{1,80,14,14};
			std::vector<int64_t> shape16{1,112,14,14};
			std::vector<int64_t> shape17{1,112,14,14};
			std::vector<int64_t> shape18{1,112,14,14};
			std::vector<int64_t> shape19{1,112,14,14};
			std::vector<int64_t> shape20{1,192,7,7};
			std::vector<int64_t> shape21{1,192,7,7};
			std::vector<int64_t> shape22{1,192,7,7};
			std::vector<int64_t> shape23{1,192,7,7};
			std::vector<int64_t> shape24{1,192,7,7};
			std::vector<int64_t> shape25{1,320,7,7};
			std::vector<int64_t> shape26{1,320,7,7};
			std::vector<int64_t> shape27{1,1280,7,7};
			std::vector<int64_t> shape28{1,1280,7,7};
			std::vector<int64_t> shape29{1,1280,1,1};
			std::vector<int64_t> shape30{1,1280};
			std::vector<int64_t> shape31{1,1280};
			std::vector<int64_t> shape32{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
		}
		else if(model_name == "b2")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,32,112,112};
			std::vector<int64_t> shape3{1,32,112,112};
			std::vector<int64_t> shape4{1,16,112,112};
			std::vector<int64_t> shape5{1,16,112,112};
			std::vector<int64_t> shape6{1,24,56,56};
			std::vector<int64_t> shape7{1,24,56,56};
			std::vector<int64_t> shape8{1,24,56,56};
			std::vector<int64_t> shape9{1,48,28,28};
			std::vector<int64_t> shape10{1,48,28,28};
			std::vector<int64_t> shape11{1,48,28,28};
			std::vector<int64_t> shape12{1,88,14,14};
			std::vector<int64_t> shape13{1,88,14,14};
			std::vector<int64_t> shape14{1,88,14,14};
			std::vector<int64_t> shape15{1,88,14,14};
			std::vector<int64_t> shape16{1,120,14,14};
			std::vector<int64_t> shape17{1,120,14,14};
			std::vector<int64_t> shape18{1,120,14,14};
			std::vector<int64_t> shape19{1,120,14,14};
			std::vector<int64_t> shape20{1,208,7,7};
			std::vector<int64_t> shape21{1,208,7,7};
			std::vector<int64_t> shape22{1,208,7,7};
			std::vector<int64_t> shape23{1,208,7,7};
			std::vector<int64_t> shape24{1,208,7,7};
			std::vector<int64_t> shape25{1,352,7,7};
			std::vector<int64_t> shape26{1,352,7,7};
			std::vector<int64_t> shape27{1,1408,7,7};
			std::vector<int64_t> shape28{1,1408,7,7};
			std::vector<int64_t> shape29{1,1408,1,1};
			std::vector<int64_t> shape30{1,1408};
			std::vector<int64_t> shape31{1,1408};
			std::vector<int64_t> shape32{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
		}

		else if(model_name == "b3")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,40,112,112};
			std::vector<int64_t> shape3{1,40,112,112};
			std::vector<int64_t> shape4{1,24,112,112};
			std::vector<int64_t> shape5{1,24,112,112};
			std::vector<int64_t> shape6{1,32,56,56};
			std::vector<int64_t> shape7{1,32,56,56};
			std::vector<int64_t> shape8{1,32,56,56};
			std::vector<int64_t> shape9{1,48,28,28};
			std::vector<int64_t> shape10{1,48,28,28};
			std::vector<int64_t> shape11{1,48,28,28};
			std::vector<int64_t> shape12{1,96,14,14};
			std::vector<int64_t> shape13{1,96,14,14};
			std::vector<int64_t> shape14{1,96,14,14};
			std::vector<int64_t> shape15{1,96,14,14};
			std::vector<int64_t> shape16{1,96,14,14};
			std::vector<int64_t> shape17{1,136,14,14};
			std::vector<int64_t> shape18{1,136,14,14};
			std::vector<int64_t> shape19{1,136,14,14};
			std::vector<int64_t> shape20{1,136,14,14};
			std::vector<int64_t> shape21{1,136,14,14};
			std::vector<int64_t> shape22{1,232,7,7};
			std::vector<int64_t> shape23{1,232,7,7};
			std::vector<int64_t> shape24{1,232,7,7};
			std::vector<int64_t> shape25{1,232,7,7};
			std::vector<int64_t> shape26{1,232,7,7};
			std::vector<int64_t> shape27{1,232,7,7};
			std::vector<int64_t> shape28{1,384,7,7};
			std::vector<int64_t> shape29{1,384,7,7};
			std::vector<int64_t> shape30{1,1536,7,7};
			std::vector<int64_t> shape31{1,1536,7,7};
			std::vector<int64_t> shape32{1,1536,1,1};
			std::vector<int64_t> shape33{1,1536};
			std::vector<int64_t> shape34{1,1536};
			std::vector<int64_t> shape35{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
		}
		else if(model_name == "b4")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,48,112,112};
			std::vector<int64_t> shape3{1,48,112,112};
			std::vector<int64_t> shape4{1,24,112,112};
			std::vector<int64_t> shape5{1,24,112,112};
			std::vector<int64_t> shape6{1,32,56,56};
			std::vector<int64_t> shape7{1,32,56,56};
			std::vector<int64_t> shape8{1,32,56,56};
			std::vector<int64_t> shape9{1,32,56,56};
			std::vector<int64_t> shape10{1,56,28,28};
			std::vector<int64_t> shape11{1,56,28,28};
			std::vector<int64_t> shape12{1,56,28,28};
			std::vector<int64_t> shape13{1,56,28,28};
			std::vector<int64_t> shape14{1,112,14,14};
			std::vector<int64_t> shape15{1,112,14,14};
			std::vector<int64_t> shape16{1,112,14,14};
			std::vector<int64_t> shape17{1,112,14,14};
			std::vector<int64_t> shape18{1,112,14,14};
			std::vector<int64_t> shape19{1,112,14,14};
			std::vector<int64_t> shape20{1,160,14,14};
			std::vector<int64_t> shape21{1,160,14,14};
			std::vector<int64_t> shape22{1,160,14,14};
			std::vector<int64_t> shape23{1,160,14,14};
			std::vector<int64_t> shape24{1,160,14,14};
			std::vector<int64_t> shape25{1,160,14,14};
			std::vector<int64_t> shape26{1,272,7,7};
			std::vector<int64_t> shape27{1,272,7,7};
			std::vector<int64_t> shape28{1,272,7,7};
			std::vector<int64_t> shape29{1,272,7,7};
			std::vector<int64_t> shape30{1,272,7,7};
			std::vector<int64_t> shape31{1,272,7,7};
			std::vector<int64_t> shape32{1,272,7,7};
			std::vector<int64_t> shape33{1,272,7,7};
			std::vector<int64_t> shape34{1,448,7,7};
			std::vector<int64_t> shape35{1,448,7,7};
			std::vector<int64_t> shape36{1,1792,7,7};
			std::vector<int64_t> shape37{1,1792,7,7};
			std::vector<int64_t> shape38{1,1792,1,1};
			std::vector<int64_t> shape39{1,1792};
			std::vector<int64_t> shape40{1,1792};
			std::vector<int64_t> shape41{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			shapes.push_back(shape41);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);
			index_set.push_back((float)40);
		}
		else if(model_name == "b5")
		{

			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,48,112,112};
			std::vector<int64_t> shape3{1,48,112,112};
			std::vector<int64_t> shape4{1,24,112,112};
			std::vector<int64_t> shape5{1,24,112,112};
			std::vector<int64_t> shape6{1,24,112,112};
			std::vector<int64_t> shape7{1,40,56,56};
			std::vector<int64_t> shape8{1,40,56,56};
			std::vector<int64_t> shape9{1,40,56,56};
			std::vector<int64_t> shape10{1,40,56,56};
			std::vector<int64_t> shape11{1,40,56,56};
			std::vector<int64_t> shape12{1,64,28,28};
			std::vector<int64_t> shape13{1,64,28,28};
			std::vector<int64_t> shape14{1,64,28,28};
			std::vector<int64_t> shape15{1,64,28,28};
			std::vector<int64_t> shape16{1,64,28,28};
			std::vector<int64_t> shape17{1,128,14,14};
			std::vector<int64_t> shape18{1,128,14,14};
			std::vector<int64_t> shape19{1,128,14,14};
			std::vector<int64_t> shape20{1,128,14,14};
			std::vector<int64_t> shape21{1,128,14,14};
			std::vector<int64_t> shape22{1,128,14,14};
			std::vector<int64_t> shape23{1,128,14,14};
			std::vector<int64_t> shape24{1,176,14,14};
			std::vector<int64_t> shape25{1,176,14,14};
			std::vector<int64_t> shape26{1,176,14,14};
			std::vector<int64_t> shape27{1,176,14,14};
			std::vector<int64_t> shape28{1,176,14,14};
			std::vector<int64_t> shape29{1,176,14,14};
			std::vector<int64_t> shape30{1,176,14,14};
			std::vector<int64_t> shape31{1,304,7,7};
			std::vector<int64_t> shape32{1,304,7,7};
			std::vector<int64_t> shape33{1,304,7,7};
			std::vector<int64_t> shape34{1,304,7,7};
			std::vector<int64_t> shape35{1,304,7,7};
			std::vector<int64_t> shape36{1,304,7,7};
			std::vector<int64_t> shape37{1,304,7,7};
			std::vector<int64_t> shape38{1,304,7,7};
			std::vector<int64_t> shape39{1,304,7,7};
			std::vector<int64_t> shape40{1,512,7,7};
			std::vector<int64_t> shape41{1,512,7,7};
			std::vector<int64_t> shape42{1,512,7,7};
			std::vector<int64_t> shape43{1,2048,7,7};
			std::vector<int64_t> shape44{1,2048,7,7};
			std::vector<int64_t> shape45{1,2048,1,1};
			std::vector<int64_t> shape46{1,2048};
			std::vector<int64_t> shape47{1,2048};
			std::vector<int64_t> shape48{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			shapes.push_back(shape41);
			shapes.push_back(shape42);
			shapes.push_back(shape43);
			shapes.push_back(shape44);
			shapes.push_back(shape45);
			shapes.push_back(shape46);
			shapes.push_back(shape47);
			shapes.push_back(shape48);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);
			index_set.push_back((float)40);
			index_set.push_back((float)41);
			index_set.push_back((float)42);
			index_set.push_back((float)43);
			index_set.push_back((float)44);
			index_set.push_back((float)45);
			index_set.push_back((float)46);
			index_set.push_back((float)47);

		}


		else if(model_name == "b6")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,56,112,112};
			std::vector<int64_t> shape3{1,56,112,112};
			std::vector<int64_t> shape4{1,32,112,112};
			std::vector<int64_t> shape5{1,32,112,112};
			std::vector<int64_t> shape6{1,32,112,112};
			std::vector<int64_t> shape7{1,40,56,56};
			std::vector<int64_t> shape8{1,40,56,56};
			std::vector<int64_t> shape9{1,40,56,56};
			std::vector<int64_t> shape10{1,40,56,56};
			std::vector<int64_t> shape11{1,40,56,56};
			std::vector<int64_t> shape12{1,40,56,56};
			std::vector<int64_t> shape13{1,72,28,28};
			std::vector<int64_t> shape14{1,72,28,28};
			std::vector<int64_t> shape15{1,72,28,28};
			std::vector<int64_t> shape16{1,72,28,28};
			std::vector<int64_t> shape17{1,72,28,28};
			std::vector<int64_t> shape18{1,72,28,28};
			std::vector<int64_t> shape19{1,144,14,14};
			std::vector<int64_t> shape20{1,144,14,14};
			std::vector<int64_t> shape21{1,144,14,14};
			std::vector<int64_t> shape22{1,144,14,14};
			std::vector<int64_t> shape23{1,144,14,14};
			std::vector<int64_t> shape24{1,144,14,14};
			std::vector<int64_t> shape25{1,144,14,14};
			std::vector<int64_t> shape26{1,144,14,14};
			std::vector<int64_t> shape27{1,200,14,14};
			std::vector<int64_t> shape28{1,200,14,14};
			std::vector<int64_t> shape29{1,200,14,14};
			std::vector<int64_t> shape30{1,200,14,14};
			std::vector<int64_t> shape31{1,200,14,14};
			std::vector<int64_t> shape32{1,200,14,14};
			std::vector<int64_t> shape33{1,200,14,14};
			std::vector<int64_t> shape34{1,200,14,14};
			std::vector<int64_t> shape35{1,344,7,7};
			std::vector<int64_t> shape36{1,344,7,7};
			std::vector<int64_t> shape37{1,344,7,7};
			std::vector<int64_t> shape38{1,344,7,7};
			std::vector<int64_t> shape39{1,344,7,7};
			std::vector<int64_t> shape40{1,344,7,7};
			std::vector<int64_t> shape41{1,344,7,7};
			std::vector<int64_t> shape42{1,344,7,7};
			std::vector<int64_t> shape43{1,344,7,7};
			std::vector<int64_t> shape44{1,344,7,7};
			std::vector<int64_t> shape45{1,344,7,7};
			std::vector<int64_t> shape46{1,576,7,7};
			std::vector<int64_t> shape47{1,576,7,7};
			std::vector<int64_t> shape48{1,576,7,7};
			std::vector<int64_t> shape49{1,2304,7,7};
			std::vector<int64_t> shape50{1,2304,7,7};
			std::vector<int64_t> shape51{1,2304,1,1};
			std::vector<int64_t> shape52{1,2304};
			std::vector<int64_t> shape53{1,2304};
			std::vector<int64_t> shape54{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			shapes.push_back(shape41);
			shapes.push_back(shape42);
			shapes.push_back(shape43);
			shapes.push_back(shape44);
			shapes.push_back(shape45);
			shapes.push_back(shape46);
			shapes.push_back(shape47);
			shapes.push_back(shape48);
			shapes.push_back(shape49);
			shapes.push_back(shape50);
			shapes.push_back(shape51);
			shapes.push_back(shape52);
			shapes.push_back(shape53);
			shapes.push_back(shape54);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);
			index_set.push_back((float)40);
			index_set.push_back((float)41);
			index_set.push_back((float)42);
			index_set.push_back((float)43);
			index_set.push_back((float)44);
			index_set.push_back((float)45);
			index_set.push_back((float)46);
			index_set.push_back((float)47);
			index_set.push_back((float)48);
			index_set.push_back((float)49);
			index_set.push_back((float)50);
			index_set.push_back((float)51);
			index_set.push_back((float)52);
			index_set.push_back((float)53);
		}
		else if(model_name == "b7")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,80,28,28};
			std::vector<int64_t> shape3{1,160,14,14};
			std::vector<int64_t> shape4{1,224,14,14};
			std::vector<int64_t> shape5{1,384,7,7};
			std::vector<int64_t> shape6{1,640,7,7};
			std::vector<int64_t> shape7{1,2560,7,7};
			std::vector<int64_t> shape8{1,2560,1,1};
			std::vector<int64_t> shape9{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			index_set.push_back((float)0);
			index_set.push_back((float)14);
			index_set.push_back((float)21);
			index_set.push_back((float)31);
			index_set.push_back((float)41);
			index_set.push_back((float)54);
			index_set.push_back((float)58);
			index_set.push_back((float)60);
			index_set.push_back((float)64);
		}
		else if(model_name == "mobilenetv2")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,24,56,56};
			std::vector<int64_t> shape3{1,32,28,28};
			std::vector<int64_t> shape4{1,64,14,14};
			std::vector<int64_t> shape5{1,96,14,14};
			std::vector<int64_t> shape6{1,160,7,7};
			std::vector<int64_t> shape7{1,320,7,7};
			std::vector<int64_t> shape8{1,1280,7,7};
			std::vector<int64_t> shape9{1,1280,1,1};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			index_set.push_back((float)0);

			index_set.push_back((float)5);
			index_set.push_back((float)7);
			index_set.push_back((float)10);
			index_set.push_back((float)14);
			index_set.push_back((float)17);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)24);

		}

		else if(model_name =="alexnet")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,64,27,27};
			std::vector<int64_t> shape3{1,192,27,27};
			std::vector<int64_t> shape4{1,192,13,13};
			std::vector<int64_t> shape5{1,384,13,13};
			std::vector<int64_t> shape6{1,256,13,13};
			std::vector<int64_t> shape7{1,256,6,6};
			std::vector<int64_t> shape8{1,4096};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			index_set.push_back((float)0);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)9);
			index_set.push_back((float)13);
			index_set.push_back((float)16);
		}
		else if(model_name =="darknet53")
		{


			std::vector<int64_t> shape1{1,3,448,448};
			std::vector<int64_t> shape2{1,512,28,28};
			std::vector<int64_t> shape3{1,1024,14,14};
			std::vector<int64_t> shape4{1,1024,1,1};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			index_set.push_back((float)0);
			index_set.push_back((float)24);
			index_set.push_back((float)35);
			index_set.push_back((float)42);
		}

		else if(model_name == "resnet50")
		{
			std::vector<int64_t> shape1{1,3,224,224};
std::vector<int64_t> shape2{1,64,112,112};
std::vector<int64_t> shape3{1,64,112,112};
std::vector<int64_t> shape4{1,64,112,112};
std::vector<int64_t> shape5{1,64,56,56};
std::vector<int64_t> shape6{1,256,56,56};
std::vector<int64_t> shape7{1,256,56,56};
std::vector<int64_t> shape8{1,256,56,56};
std::vector<int64_t> shape9{1,512,28,28};
std::vector<int64_t> shape10{1,512,28,28};
std::vector<int64_t> shape11{1,512,28,28};
std::vector<int64_t> shape12{1,512,28,28};
std::vector<int64_t> shape13{1,1024,14,14};
std::vector<int64_t> shape14{1,1024,14,14};
std::vector<int64_t> shape15{1,1024,14,14};
std::vector<int64_t> shape16{1,1024,14,14};
std::vector<int64_t> shape17{1,1024,14,14};
std::vector<int64_t> shape18{1,1024,14,14};
std::vector<int64_t> shape19{1,2048,7,7};
std::vector<int64_t> shape20{1,2048,7,7};
std::vector<int64_t> shape21{1,2048,7,7};
std::vector<int64_t> shape22{1,2048,1,1};
std::vector<int64_t> shape23{1,2048};
shapes.push_back(shape1);
shapes.push_back(shape2);
shapes.push_back(shape3);
shapes.push_back(shape4);
shapes.push_back(shape5);
shapes.push_back(shape6);
shapes.push_back(shape7);
shapes.push_back(shape8);
shapes.push_back(shape9);
shapes.push_back(shape10);
shapes.push_back(shape11);
shapes.push_back(shape12);
shapes.push_back(shape13);
shapes.push_back(shape14);
shapes.push_back(shape15);
shapes.push_back(shape16);
shapes.push_back(shape17);
shapes.push_back(shape18);
shapes.push_back(shape19);
shapes.push_back(shape20);
shapes.push_back(shape21);
shapes.push_back(shape22);
shapes.push_back(shape23);
index_set.push_back((float)0);
index_set.push_back((float)1);
index_set.push_back((float)2);
index_set.push_back((float)3);
index_set.push_back((float)4);
index_set.push_back((float)5);
index_set.push_back((float)6);
index_set.push_back((float)7);
index_set.push_back((float)8);
index_set.push_back((float)9);
index_set.push_back((float)10);
index_set.push_back((float)11);
index_set.push_back((float)12);
index_set.push_back((float)13);
index_set.push_back((float)14);
index_set.push_back((float)15);
index_set.push_back((float)16);
index_set.push_back((float)17);
index_set.push_back((float)18);
index_set.push_back((float)19);
index_set.push_back((float)20);
index_set.push_back((float)21);
index_set.push_back((float)22);
		}
		else if(model_name == "resnet152")
		{
std::vector<int64_t> shape1{1,3,224,224};
std::vector<int64_t> shape2{1,64,112,112};
std::vector<int64_t> shape3{1,64,112,112};
std::vector<int64_t> shape4{1,64,112,112};
std::vector<int64_t> shape5{1,64,56,56};
std::vector<int64_t> shape6{1,256,56,56};
std::vector<int64_t> shape7{1,256,56,56};
std::vector<int64_t> shape8{1,256,56,56};
std::vector<int64_t> shape9{1,512,28,28};
std::vector<int64_t> shape10{1,512,28,28};
std::vector<int64_t> shape11{1,512,28,28};
std::vector<int64_t> shape12{1,512,28,28};
std::vector<int64_t> shape13{1,512,28,28};
std::vector<int64_t> shape14{1,512,28,28};
std::vector<int64_t> shape15{1,512,28,28};
std::vector<int64_t> shape16{1,512,28,28};
std::vector<int64_t> shape17{1,1024,14,14};
std::vector<int64_t> shape18{1,1024,14,14};
std::vector<int64_t> shape19{1,1024,14,14};
std::vector<int64_t> shape20{1,1024,14,14};
std::vector<int64_t> shape21{1,1024,14,14};
std::vector<int64_t> shape22{1,1024,14,14};
std::vector<int64_t> shape23{1,1024,14,14};
std::vector<int64_t> shape24{1,1024,14,14};
std::vector<int64_t> shape25{1,1024,14,14};
std::vector<int64_t> shape26{1,1024,14,14};
std::vector<int64_t> shape27{1,1024,14,14};
std::vector<int64_t> shape28{1,1024,14,14};
std::vector<int64_t> shape29{1,1024,14,14};
std::vector<int64_t> shape30{1,1024,14,14};
std::vector<int64_t> shape31{1,1024,14,14};
std::vector<int64_t> shape32{1,1024,14,14};
std::vector<int64_t> shape33{1,1024,14,14};
std::vector<int64_t> shape34{1,1024,14,14};
std::vector<int64_t> shape35{1,1024,14,14};
std::vector<int64_t> shape36{1,1024,14,14};
std::vector<int64_t> shape37{1,1024,14,14};
std::vector<int64_t> shape38{1,1024,14,14};
std::vector<int64_t> shape39{1,1024,14,14};
std::vector<int64_t> shape40{1,1024,14,14};
std::vector<int64_t> shape41{1,1024,14,14};
std::vector<int64_t> shape42{1,1024,14,14};
std::vector<int64_t> shape43{1,1024,14,14};
std::vector<int64_t> shape44{1,1024,14,14};
std::vector<int64_t> shape45{1,1024,14,14};
std::vector<int64_t> shape46{1,1024,14,14};
std::vector<int64_t> shape47{1,1024,14,14};
std::vector<int64_t> shape48{1,1024,14,14};
std::vector<int64_t> shape49{1,1024,14,14};
std::vector<int64_t> shape50{1,1024,14,14};
std::vector<int64_t> shape51{1,1024,14,14};
std::vector<int64_t> shape52{1,1024,14,14};
std::vector<int64_t> shape53{1,2048,7,7};
std::vector<int64_t> shape54{1,2048,7,7};
std::vector<int64_t> shape55{1,2048,7,7};
std::vector<int64_t> shape56{1,2048,1,1};
std::vector<int64_t> shape57{1,2048};
shapes.push_back(shape1);
shapes.push_back(shape2);
shapes.push_back(shape3);
shapes.push_back(shape4);
shapes.push_back(shape5);
shapes.push_back(shape6);
shapes.push_back(shape7);
shapes.push_back(shape8);
shapes.push_back(shape9);
shapes.push_back(shape10);
shapes.push_back(shape11);
shapes.push_back(shape12);
shapes.push_back(shape13);
shapes.push_back(shape14);
shapes.push_back(shape15);
shapes.push_back(shape16);
shapes.push_back(shape17);
shapes.push_back(shape18);
shapes.push_back(shape19);
shapes.push_back(shape20);
shapes.push_back(shape21);
shapes.push_back(shape22);
shapes.push_back(shape23);
shapes.push_back(shape24);
shapes.push_back(shape25);
shapes.push_back(shape26);
shapes.push_back(shape27);
shapes.push_back(shape28);
shapes.push_back(shape29);
shapes.push_back(shape30);
shapes.push_back(shape31);
shapes.push_back(shape32);
shapes.push_back(shape33);
shapes.push_back(shape34);
shapes.push_back(shape35);
shapes.push_back(shape36);
shapes.push_back(shape37);
shapes.push_back(shape38);
shapes.push_back(shape39);
shapes.push_back(shape40);
shapes.push_back(shape41);
shapes.push_back(shape42);
shapes.push_back(shape43);
shapes.push_back(shape44);
shapes.push_back(shape45);
shapes.push_back(shape46);
shapes.push_back(shape47);
shapes.push_back(shape48);
shapes.push_back(shape49);
shapes.push_back(shape50);
shapes.push_back(shape51);
shapes.push_back(shape52);
shapes.push_back(shape53);
shapes.push_back(shape54);
shapes.push_back(shape55);
shapes.push_back(shape56);
shapes.push_back(shape57);
index_set.push_back((float)0);
index_set.push_back((float)1);
index_set.push_back((float)2);
index_set.push_back((float)3);
index_set.push_back((float)4);
index_set.push_back((float)5);
index_set.push_back((float)6);
index_set.push_back((float)7);
index_set.push_back((float)8);
index_set.push_back((float)9);
index_set.push_back((float)10);
index_set.push_back((float)11);
index_set.push_back((float)12);
index_set.push_back((float)13);
index_set.push_back((float)14);
index_set.push_back((float)15);
index_set.push_back((float)16);
index_set.push_back((float)17);
index_set.push_back((float)18);
index_set.push_back((float)19);
index_set.push_back((float)20);
index_set.push_back((float)21);
index_set.push_back((float)22);
index_set.push_back((float)23);
index_set.push_back((float)24);
index_set.push_back((float)25);
index_set.push_back((float)26);
index_set.push_back((float)27);
index_set.push_back((float)28);
index_set.push_back((float)29);
index_set.push_back((float)30);
index_set.push_back((float)31);
index_set.push_back((float)32);
index_set.push_back((float)33);
index_set.push_back((float)34);
index_set.push_back((float)35);
index_set.push_back((float)36);
index_set.push_back((float)37);
index_set.push_back((float)38);
index_set.push_back((float)39);
index_set.push_back((float)40);
index_set.push_back((float)41);
index_set.push_back((float)42);
index_set.push_back((float)43);
index_set.push_back((float)44);
index_set.push_back((float)45);
index_set.push_back((float)46);
index_set.push_back((float)47);
index_set.push_back((float)48);
index_set.push_back((float)49);
index_set.push_back((float)50);
index_set.push_back((float)51);
index_set.push_back((float)52);
index_set.push_back((float)53);
index_set.push_back((float)54);
index_set.push_back((float)55);
index_set.push_back((float)56);


		}
		else if(model_name == "resnet152")
		{
			std::vector<int64_t> shape1{1,3,224,224};
std::vector<int64_t> shape2{1,64,112,112};
std::vector<int64_t> shape3{1,64,112,112};
std::vector<int64_t> shape4{1,64,112,112};
std::vector<int64_t> shape5{1,64,56,56};
std::vector<int64_t> shape6{1,256,56,56};
std::vector<int64_t> shape7{1,256,56,56};
std::vector<int64_t> shape8{1,256,56,56};
std::vector<int64_t> shape9{1,512,28,28};
std::vector<int64_t> shape10{1,512,28,28};
std::vector<int64_t> shape11{1,512,28,28};
std::vector<int64_t> shape12{1,512,28,28};
std::vector<int64_t> shape13{1,512,28,28};
std::vector<int64_t> shape14{1,512,28,28};
std::vector<int64_t> shape15{1,512,28,28};
std::vector<int64_t> shape16{1,512,28,28};
std::vector<int64_t> shape17{1,1024,14,14};
std::vector<int64_t> shape18{1,1024,14,14};
std::vector<int64_t> shape19{1,1024,14,14};
std::vector<int64_t> shape20{1,1024,14,14};
std::vector<int64_t> shape21{1,1024,14,14};
std::vector<int64_t> shape22{1,1024,14,14};
std::vector<int64_t> shape23{1,1024,14,14};
std::vector<int64_t> shape24{1,1024,14,14};
std::vector<int64_t> shape25{1,1024,14,14};
std::vector<int64_t> shape26{1,1024,14,14};
std::vector<int64_t> shape27{1,1024,14,14};
std::vector<int64_t> shape28{1,1024,14,14};
std::vector<int64_t> shape29{1,1024,14,14};
std::vector<int64_t> shape30{1,1024,14,14};
std::vector<int64_t> shape31{1,1024,14,14};
std::vector<int64_t> shape32{1,1024,14,14};
std::vector<int64_t> shape33{1,1024,14,14};
std::vector<int64_t> shape34{1,1024,14,14};
std::vector<int64_t> shape35{1,1024,14,14};
std::vector<int64_t> shape36{1,1024,14,14};
std::vector<int64_t> shape37{1,1024,14,14};
std::vector<int64_t> shape38{1,1024,14,14};
std::vector<int64_t> shape39{1,1024,14,14};
std::vector<int64_t> shape40{1,1024,14,14};
std::vector<int64_t> shape41{1,1024,14,14};
std::vector<int64_t> shape42{1,1024,14,14};
std::vector<int64_t> shape43{1,1024,14,14};
std::vector<int64_t> shape44{1,1024,14,14};
std::vector<int64_t> shape45{1,1024,14,14};
std::vector<int64_t> shape46{1,1024,14,14};
std::vector<int64_t> shape47{1,1024,14,14};
std::vector<int64_t> shape48{1,1024,14,14};
std::vector<int64_t> shape49{1,1024,14,14};
std::vector<int64_t> shape50{1,1024,14,14};
std::vector<int64_t> shape51{1,1024,14,14};
std::vector<int64_t> shape52{1,1024,14,14};
std::vector<int64_t> shape53{1,2048,7,7};
std::vector<int64_t> shape54{1,2048,7,7};
std::vector<int64_t> shape55{1,2048,7,7};
std::vector<int64_t> shape56{1,2048,1,1};
std::vector<int64_t> shape57{1,2048};
shapes.push_back(shape1);
shapes.push_back(shape2);
shapes.push_back(shape3);
shapes.push_back(shape4);
shapes.push_back(shape5);
shapes.push_back(shape6);
shapes.push_back(shape7);
shapes.push_back(shape8);
shapes.push_back(shape9);
shapes.push_back(shape10);
shapes.push_back(shape11);
shapes.push_back(shape12);
shapes.push_back(shape13);
shapes.push_back(shape14);
shapes.push_back(shape15);
shapes.push_back(shape16);
shapes.push_back(shape17);
shapes.push_back(shape18);
shapes.push_back(shape19);
shapes.push_back(shape20);
shapes.push_back(shape21);
shapes.push_back(shape22);
shapes.push_back(shape23);
shapes.push_back(shape24);
shapes.push_back(shape25);
shapes.push_back(shape26);
shapes.push_back(shape27);
shapes.push_back(shape28);
shapes.push_back(shape29);
shapes.push_back(shape30);
shapes.push_back(shape31);
shapes.push_back(shape32);
shapes.push_back(shape33);
shapes.push_back(shape34);
shapes.push_back(shape35);
shapes.push_back(shape36);
shapes.push_back(shape37);
shapes.push_back(shape38);
shapes.push_back(shape39);
shapes.push_back(shape40);
shapes.push_back(shape41);
shapes.push_back(shape42);
shapes.push_back(shape43);
shapes.push_back(shape44);
shapes.push_back(shape45);
shapes.push_back(shape46);
shapes.push_back(shape47);
shapes.push_back(shape48);
shapes.push_back(shape49);
shapes.push_back(shape50);
shapes.push_back(shape51);
shapes.push_back(shape52);
shapes.push_back(shape53);
shapes.push_back(shape54);
shapes.push_back(shape55);
shapes.push_back(shape56);
shapes.push_back(shape57);
index_set.push_back((float)0);
index_set.push_back((float)1);
index_set.push_back((float)2);
index_set.push_back((float)3);
index_set.push_back((float)4);
index_set.push_back((float)5);
index_set.push_back((float)6);
index_set.push_back((float)7);
index_set.push_back((float)8);
index_set.push_back((float)9);
index_set.push_back((float)10);
index_set.push_back((float)11);
index_set.push_back((float)12);
index_set.push_back((float)13);
index_set.push_back((float)14);
index_set.push_back((float)15);
index_set.push_back((float)16);
index_set.push_back((float)17);
index_set.push_back((float)18);
index_set.push_back((float)19);
index_set.push_back((float)20);
index_set.push_back((float)21);
index_set.push_back((float)22);
index_set.push_back((float)23);
index_set.push_back((float)24);
index_set.push_back((float)25);
index_set.push_back((float)26);
index_set.push_back((float)27);
index_set.push_back((float)28);
index_set.push_back((float)29);
index_set.push_back((float)30);
index_set.push_back((float)31);
index_set.push_back((float)32);
index_set.push_back((float)33);
index_set.push_back((float)34);
index_set.push_back((float)35);
index_set.push_back((float)36);
index_set.push_back((float)37);
index_set.push_back((float)38);
index_set.push_back((float)39);
index_set.push_back((float)40);
index_set.push_back((float)41);
index_set.push_back((float)42);
index_set.push_back((float)43);
index_set.push_back((float)44);
index_set.push_back((float)45);
index_set.push_back((float)46);
index_set.push_back((float)47);
index_set.push_back((float)48);
index_set.push_back((float)49);
index_set.push_back((float)50);
index_set.push_back((float)51);
index_set.push_back((float)52);
index_set.push_back((float)53);
index_set.push_back((float)54);
index_set.push_back((float)55);
index_set.push_back((float)56);

		}
		else if(model_name == "vgg16")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,64,224,224};
			std::vector<int64_t> shape3{1,64,224,224};
			std::vector<int64_t> shape4{1,64,224,224};
			std::vector<int64_t> shape5{1,64,224,224};
			std::vector<int64_t> shape6{1,64,112,112};
			std::vector<int64_t> shape7{1,128,112,112};
			std::vector<int64_t> shape8{1,128,112,112};
			std::vector<int64_t> shape9{1,128,112,112};
			std::vector<int64_t> shape10{1,128,112,112};
			std::vector<int64_t> shape11{1,128,56,56};
			std::vector<int64_t> shape12{1,256,56,56};
			std::vector<int64_t> shape13{1,256,56,56};
			std::vector<int64_t> shape14{1,256,56,56};
			std::vector<int64_t> shape15{1,256,56,56};
			std::vector<int64_t> shape16{1,256,56,56};
			std::vector<int64_t> shape17{1,256,56,56};
			std::vector<int64_t> shape18{1,256,28,28};
			std::vector<int64_t> shape19{1,512,28,28};
			std::vector<int64_t> shape20{1,512,28,28};
			std::vector<int64_t> shape21{1,512,28,28};
			std::vector<int64_t> shape22{1,512,28,28};
			std::vector<int64_t> shape23{1,512,28,28};
			std::vector<int64_t> shape24{1,512,28,28};
			std::vector<int64_t> shape25{1,512,14,14};
			std::vector<int64_t> shape26{1,512,14,14};
			std::vector<int64_t> shape27{1,512,14,14};
			std::vector<int64_t> shape28{1,512,14,14};
			std::vector<int64_t> shape29{1,512,14,14};
			std::vector<int64_t> shape30{1,512,14,14};
			std::vector<int64_t> shape31{1,512,14,14};
			std::vector<int64_t> shape32{1,512,7,7};
			std::vector<int64_t> shape33{1,512,7,7};
			std::vector<int64_t> shape34{1,25088};
			std::vector<int64_t> shape35{1,4096};
			std::vector<int64_t> shape36{1,4096};
			std::vector<int64_t> shape37{1,4096};
			std::vector<int64_t> shape38{1,4096};
			std::vector<int64_t> shape39{1,4096};
			std::vector<int64_t> shape40{1,4096};
			std::vector<int64_t> shape41{1,1000};
			
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			shapes.push_back(shape41);

			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);
			index_set.push_back((float)40);
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		//std::uniform_int_distribution<int> dis(0, shapes.size()-1);
		
		
		//This is real implementation!!XXX
		//std::uniform_int_distribution<int> dis(0, shapes.size()-1);
		//int randindex = dis(gen);	

		int randindex = 11;
		/*std::uniform_int_distribution<int> dis(0, 100);
		int dice = dis(gen);
		int randindex = 0;
		if (dice > 0)
		{
			randindex = 16;
		}*/
		
		
		//int randindex = dis(gen);  
		//int randindex = 2;
		//randindex = index_filtering[randindex];

		//  int randindex = rand() % (48-1-0+1) + 0;
		//  int randindex = rand() % (shapes.size()-1-0+1) + 0;
		int input0_datasize = 1;
		for(int i = 0; i <(int) shapes[randindex].size();i++)
		{
			input0_datasize = input0_datasize * shapes[randindex][i];
		}

		//std::vector<float> input0_data(1*40*28*28);
		std::vector<float> input0_data(input0_datasize);

		memset(&input0_data[0], 2.3, input0_data.size() * sizeof input0_data[0]);
		nic::InferInput* input0;
		nic::InferInput::Create(&input0, "INPUT__0", shapes[randindex], "FP32");
		std::shared_ptr<nic::InferInput> input0_ptr;
		input0_ptr.reset(input0);
		input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input0_data[0]),
				input0_data.size() * sizeof(float));

		//memset(&input1_data[0], 7, input1_data.size() * sizeof input1_data[0]);

		std::vector<nic::InferInput*> myinputs = {input0_ptr.get()};

		nic::InferOptions myoptions(model_name);
		myoptions.partitioning_point_ = index_set[randindex];

		RETURN_IF_ERROR(client_.http_client_->Infer(
					result, 
					//options, 
					myoptions,
					//inputs
					myinputs
					, outputs, *http_headers_));
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::AsyncInfer(
		nic::InferenceServerClient::OnCompleteFn callback,
		const nic::InferOptions& options,
		const std::vector<nic::InferInput*>& inputs,
		const std::vector<const nic::InferRequestedOutput*>& outputs)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->AsyncInfer(
					callback, options, inputs, outputs, *http_headers_));
	} else {

		// rand() % (last value - first value + 1) + first value	
		//int randindex = rand() % (3-0+1) + 0;
		std::vector<std::vector<int64_t>> shapes;
		std::vector<float> index_set;

		std::string model_name = options.model_name_;

		if(model_name == "b0")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,24,56,56};
			std::vector<int64_t> shape3{1,40,28,28};
			std::vector<int64_t> shape4{1,80,14,14};
			std::vector<int64_t> shape5{1,112,14,14};
			std::vector<int64_t> shape6{1,192,7,7};
			std::vector<int64_t> shape7{1,320,7,7};
			std::vector<int64_t> shape8{1,1280,7,7};
			std::vector<int64_t> shape9{1,1280,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)4);
			index_set.push_back((float)6);
			index_set.push_back((float)8);
			index_set.push_back((float)11);
			index_set.push_back((float)14);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)21);
			index_set.push_back((float)24);

		}
		else if(model_name == "b6")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,40,56,56};
			std::vector<int64_t> shape3{1,72,28,28};
			std::vector<int64_t> shape4{1,144,14,14};
			std::vector<int64_t> shape5{1,200,14,14};
			std::vector<int64_t> shape6{1,344,7,7};
			std::vector<int64_t> shape7{1,576,7,7};
			std::vector<int64_t> shape8{1,2304,7,7};
			std::vector<int64_t> shape9{1,2304,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)6);
			index_set.push_back((float)12);
			index_set.push_back((float)18);
			index_set.push_back((float)26);
			index_set.push_back((float)34);
			index_set.push_back((float)45);
			index_set.push_back((float)48);
			index_set.push_back((float)50);
			index_set.push_back((float)54);

		}
		else if(model_name == "b7")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,80,28,28};
			std::vector<int64_t> shape3{1,160,14,14};
			std::vector<int64_t> shape4{1,224,14,14};
			std::vector<int64_t> shape5{1,384,7,7};
			std::vector<int64_t> shape6{1,640,7,7};
			std::vector<int64_t> shape7{1,2560,7,7};
			std::vector<int64_t> shape8{1,2560,1,1};
			std::vector<int64_t> shape9{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			index_set.push_back((float)0);
			index_set.push_back((float)14);
			index_set.push_back((float)21);
			index_set.push_back((float)31);
			index_set.push_back((float)41);
			index_set.push_back((float)54);
			index_set.push_back((float)58);
			index_set.push_back((float)60);
			index_set.push_back((float)64);
		}
		else if(model_name == "b5")
		{

			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,48,112,112};
			std::vector<int64_t> shape3{1,48,112,112};
			std::vector<int64_t> shape4{1,24,112,112};
			std::vector<int64_t> shape5{1,24,112,112};
			std::vector<int64_t> shape6{1,24,112,112};
			std::vector<int64_t> shape7{1,40,56,56};
			std::vector<int64_t> shape8{1,40,56,56};
			std::vector<int64_t> shape9{1,40,56,56};
			std::vector<int64_t> shape10{1,40,56,56};
			std::vector<int64_t> shape11{1,40,56,56};
			std::vector<int64_t> shape12{1,64,28,28};
			std::vector<int64_t> shape13{1,64,28,28};
			std::vector<int64_t> shape14{1,64,28,28};
			std::vector<int64_t> shape15{1,64,28,28};
			std::vector<int64_t> shape16{1,64,28,28};
			std::vector<int64_t> shape17{1,128,14,14};
			std::vector<int64_t> shape18{1,128,14,14};
			std::vector<int64_t> shape19{1,128,14,14};
			std::vector<int64_t> shape20{1,128,14,14};
			std::vector<int64_t> shape21{1,128,14,14};
			std::vector<int64_t> shape22{1,128,14,14};
			std::vector<int64_t> shape23{1,128,14,14};
			std::vector<int64_t> shape24{1,176,14,14};
			std::vector<int64_t> shape25{1,176,14,14};
			std::vector<int64_t> shape26{1,176,14,14};
			std::vector<int64_t> shape27{1,176,14,14};
			std::vector<int64_t> shape28{1,176,14,14};
			std::vector<int64_t> shape29{1,176,14,14};
			std::vector<int64_t> shape30{1,176,14,14};
			std::vector<int64_t> shape31{1,304,7,7};
			std::vector<int64_t> shape32{1,304,7,7};
			std::vector<int64_t> shape33{1,304,7,7};
			std::vector<int64_t> shape34{1,304,7,7};
			std::vector<int64_t> shape35{1,304,7,7};
			std::vector<int64_t> shape36{1,304,7,7};
			std::vector<int64_t> shape37{1,304,7,7};
			std::vector<int64_t> shape38{1,304,7,7};
			std::vector<int64_t> shape39{1,304,7,7};
			std::vector<int64_t> shape40{1,512,7,7};
			std::vector<int64_t> shape41{1,512,7,7};
			std::vector<int64_t> shape42{1,512,7,7};
			std::vector<int64_t> shape43{1,2048,7,7};
			std::vector<int64_t> shape44{1,2048,7,7};
			std::vector<int64_t> shape45{1,2048,1,1};
			std::vector<int64_t> shape46{1,2048};
			std::vector<int64_t> shape47{1,2048};
			std::vector<int64_t> shape48{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			shapes.push_back(shape41);
			shapes.push_back(shape42);
			shapes.push_back(shape43);
			shapes.push_back(shape44);
			shapes.push_back(shape45);
			shapes.push_back(shape46);
			shapes.push_back(shape47);
			shapes.push_back(shape48);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);
			index_set.push_back((float)40);
			index_set.push_back((float)41);
			index_set.push_back((float)42);
			index_set.push_back((float)43);
			index_set.push_back((float)44);
			index_set.push_back((float)45);
			index_set.push_back((float)46);
			index_set.push_back((float)47);

		}
		else if(model_name == "mobilenetv2")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,24,56,56};
			std::vector<int64_t> shape3{1,32,28,28};
			std::vector<int64_t> shape4{1,64,14,14};
			std::vector<int64_t> shape5{1,96,14,14};
			std::vector<int64_t> shape6{1,160,7,7};
			std::vector<int64_t> shape7{1,320,7,7};
			std::vector<int64_t> shape8{1,1280,7,7};
			std::vector<int64_t> shape9{1,1280,1,1};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			index_set.push_back((float)0);

			index_set.push_back((float)5);
			index_set.push_back((float)7);
			index_set.push_back((float)10);
			index_set.push_back((float)14);
			index_set.push_back((float)17);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)24);

		}

		else if(model_name =="b3")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,32,56,56};
			std::vector<int64_t> shape3{1,48,28,28};
			std::vector<int64_t> shape4{1,96,14,14};
			std::vector<int64_t> shape5{1,136,14,14};
			std::vector<int64_t> shape6{1,232,7,7};
			std::vector<int64_t> shape7{1,384,7,7};
			std::vector<int64_t> shape8{1,1536,7,7};
			std::vector<int64_t> shape9{1,1536,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)5);
			index_set.push_back((float)8);
			index_set.push_back((float)11);
			index_set.push_back((float)16);
			index_set.push_back((float)21);
			index_set.push_back((float)27);
			index_set.push_back((float)29);
			index_set.push_back((float)31);
			index_set.push_back((float)35);
		}
		else if(model_name =="alexnet")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,64,27,27};
			std::vector<int64_t> shape3{1,192,27,27};
			std::vector<int64_t> shape4{1,192,13,13};
			std::vector<int64_t> shape5{1,384,13,13};
			std::vector<int64_t> shape6{1,256,13,13};
			std::vector<int64_t> shape7{1,256,6,6};
			std::vector<int64_t> shape8{1,4096};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			index_set.push_back((float)0);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)9);
			index_set.push_back((float)13);
			index_set.push_back((float)16);
		}
		else if(model_name =="b4")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,32,56,56};
			std::vector<int64_t> shape3{1,56,28,28};
			std::vector<int64_t> shape4{1,112,14,14};
			std::vector<int64_t> shape5{1,160,14,14};
			std::vector<int64_t> shape6{1,272,7,7};
			std::vector<int64_t> shape7{1,448,7,7};
			std::vector<int64_t> shape8{1,1792,7,7};
			std::vector<int64_t> shape9{1,1792,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)5);
			index_set.push_back((float)9);
			index_set.push_back((float)13);
			index_set.push_back((float)19);
			index_set.push_back((float)25);
			index_set.push_back((float)33);
			index_set.push_back((float)35);
			index_set.push_back((float)37);
			index_set.push_back((float)41);


		}
		else if(model_name =="b1")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,24,56,56};
			std::vector<int64_t> shape3{1,40,28,28};
			std::vector<int64_t> shape4{1,80,14,14};
			std::vector<int64_t> shape5{1,112,14,14};
			std::vector<int64_t> shape6{1,192,7,7};
			std::vector<int64_t> shape7{1,320,7,7};
			std::vector<int64_t> shape8{1,1280,7,7};
			std::vector<int64_t> shape9{1,1280,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)5);
			index_set.push_back((float)8);
			index_set.push_back((float)11);
			index_set.push_back((float)15);
			index_set.push_back((float)19);
			index_set.push_back((float)24);
			index_set.push_back((float)26);
			index_set.push_back((float)28);
			index_set.push_back((float)32);


		}


		else if(model_name =="b2")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,24,56,56};
			std::vector<int64_t> shape3{1,48,28,28};
			std::vector<int64_t> shape4{1,88,14,14};
			std::vector<int64_t> shape5{1,120,14,14};
			std::vector<int64_t> shape6{1,208,7,7};
			std::vector<int64_t> shape7{1,352,7,7};
			std::vector<int64_t> shape8{1,1408,7,7};
			std::vector<int64_t> shape9{1,1408,1,1};
			std::vector<int64_t> shape10{1,1000};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			index_set.push_back((float)0);
			index_set.push_back((float)5);
			index_set.push_back((float)8);
			index_set.push_back((float)11);
			index_set.push_back((float)15);
			index_set.push_back((float)19);
			index_set.push_back((float)24);
			index_set.push_back((float)26);
			index_set.push_back((float)28);
			index_set.push_back((float)32);
		}


		else if(model_name =="darknet53")
		{


			std::vector<int64_t> shape1{1,3,448,448};
			std::vector<int64_t> shape2{1,512,28,28};
			std::vector<int64_t> shape3{1,1024,14,14};
			std::vector<int64_t> shape4{1,1024,1,1};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			index_set.push_back((float)0);
			index_set.push_back((float)24);
			index_set.push_back((float)35);
			index_set.push_back((float)42);
		}


		else if(model_name == "resnet152")
		{

			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,512,28,28};
			std::vector<int64_t> shape3{1,1024,14,14};
			std::vector<int64_t> shape4{1,1024,1,1};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			index_set.push_back((float)0);
			index_set.push_back((float)24);
			index_set.push_back((float)35);
			index_set.push_back((float)42);
		}

		else if(model_name == "vgg16")
		{
			std::vector<int64_t> shape1{1,3,224,224};
			std::vector<int64_t> shape2{1,64,224,224};
			std::vector<int64_t> shape3{1,64,224,224};
			std::vector<int64_t> shape4{1,64,224,224};
			std::vector<int64_t> shape5{1,64,224,224};
			std::vector<int64_t> shape6{1,64,112,112};
			std::vector<int64_t> shape7{1,128,112,112};
			std::vector<int64_t> shape8{1,128,112,112};
			std::vector<int64_t> shape9{1,128,112,112};
			std::vector<int64_t> shape10{1,128,112,112};
			std::vector<int64_t> shape11{1,128,56,56};
			std::vector<int64_t> shape12{1,256,56,56};
			std::vector<int64_t> shape13{1,256,56,56};
			std::vector<int64_t> shape14{1,256,56,56};
			std::vector<int64_t> shape15{1,256,56,56};
			std::vector<int64_t> shape16{1,256,56,56};
			std::vector<int64_t> shape17{1,256,56,56};
			std::vector<int64_t> shape18{1,256,28,28};
			std::vector<int64_t> shape19{1,512,28,28};
			std::vector<int64_t> shape20{1,512,28,28};
			std::vector<int64_t> shape21{1,512,28,28};
			std::vector<int64_t> shape22{1,512,28,28};
			std::vector<int64_t> shape23{1,512,28,28};
			std::vector<int64_t> shape24{1,512,28,28};
			std::vector<int64_t> shape25{1,512,14,14};
			std::vector<int64_t> shape26{1,512,14,14};
			std::vector<int64_t> shape27{1,512,14,14};
			std::vector<int64_t> shape28{1,512,14,14};
			std::vector<int64_t> shape29{1,512,14,14};
			std::vector<int64_t> shape30{1,512,14,14};
			std::vector<int64_t> shape31{1,512,14,14};
			std::vector<int64_t> shape32{1,512,7,7};
			std::vector<int64_t> shape33{1,512,7,7};
			std::vector<int64_t> shape34{1,25088};
			std::vector<int64_t> shape35{1,4096};
			std::vector<int64_t> shape36{1,4096};
			std::vector<int64_t> shape37{1,4096};
			std::vector<int64_t> shape38{1,4096};
			std::vector<int64_t> shape39{1,4096};
			std::vector<int64_t> shape40{1,4096};
			shapes.push_back(shape1);
			shapes.push_back(shape2);
			shapes.push_back(shape3);
			shapes.push_back(shape4);
			shapes.push_back(shape5);
			shapes.push_back(shape6);
			shapes.push_back(shape7);
			shapes.push_back(shape8);
			shapes.push_back(shape9);
			shapes.push_back(shape10);
			shapes.push_back(shape11);
			shapes.push_back(shape12);
			shapes.push_back(shape13);
			shapes.push_back(shape14);
			shapes.push_back(shape15);
			shapes.push_back(shape16);
			shapes.push_back(shape17);
			shapes.push_back(shape18);
			shapes.push_back(shape19);
			shapes.push_back(shape20);
			shapes.push_back(shape21);
			shapes.push_back(shape22);
			shapes.push_back(shape23);
			shapes.push_back(shape24);
			shapes.push_back(shape25);
			shapes.push_back(shape26);
			shapes.push_back(shape27);
			shapes.push_back(shape28);
			shapes.push_back(shape29);
			shapes.push_back(shape30);
			shapes.push_back(shape31);
			shapes.push_back(shape32);
			shapes.push_back(shape33);
			shapes.push_back(shape34);
			shapes.push_back(shape35);
			shapes.push_back(shape36);
			shapes.push_back(shape37);
			shapes.push_back(shape38);
			shapes.push_back(shape39);
			shapes.push_back(shape40);
			index_set.push_back((float)0);
			index_set.push_back((float)1);
			index_set.push_back((float)2);
			index_set.push_back((float)3);
			index_set.push_back((float)4);
			index_set.push_back((float)5);
			index_set.push_back((float)6);
			index_set.push_back((float)7);
			index_set.push_back((float)8);
			index_set.push_back((float)9);
			index_set.push_back((float)10);
			index_set.push_back((float)11);
			index_set.push_back((float)12);
			index_set.push_back((float)13);
			index_set.push_back((float)14);
			index_set.push_back((float)15);
			index_set.push_back((float)16);
			index_set.push_back((float)17);
			index_set.push_back((float)18);
			index_set.push_back((float)19);
			index_set.push_back((float)20);
			index_set.push_back((float)21);
			index_set.push_back((float)22);
			index_set.push_back((float)23);
			index_set.push_back((float)24);
			index_set.push_back((float)25);
			index_set.push_back((float)26);
			index_set.push_back((float)27);
			index_set.push_back((float)28);
			index_set.push_back((float)29);
			index_set.push_back((float)30);
			index_set.push_back((float)31);
			index_set.push_back((float)32);
			index_set.push_back((float)33);
			index_set.push_back((float)34);
			index_set.push_back((float)35);
			index_set.push_back((float)36);
			index_set.push_back((float)37);
			index_set.push_back((float)38);
			index_set.push_back((float)39);

		}


		std::random_device rd;
		std::mt19937 gen(rd());
		//std::uniform_int_distribution<int> dis(0, shapes.size()-1);
		std::uniform_int_distribution<int> dis(0, shapes.size()-1);
		int randindex = dis(gen);	
		//int randindex = dis(gen);  
		//int randindex = 2;

		//  int randindex = rand() % (48-1-0+1) + 0;
		//  int randindex = rand() % (shapes.size()-1-0+1) + 0;
		//randindex = 0;
		int input0_datasize = 1;
		for(int i = 0; i <(int) shapes[randindex].size();i++)
		{
			input0_datasize = input0_datasize * shapes[randindex][i];
		}

		//std::vector<float> input0_data(1*40*28*28);
		std::vector<float> input0_data(input0_datasize);

		memset(&input0_data[0], 2.3, input0_data.size() * sizeof input0_data[0]);
		nic::InferInput* input0;
		nic::InferInput::Create(&input0, "INPUT__0", shapes[randindex], "FP32");
		std::shared_ptr<nic::InferInput> input0_ptr;
		input0_ptr.reset(input0);
		input0_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&input0_data[0]),
				input0_data.size() * sizeof(float));

		//memset(&input1_data[0], 7, input1_data.size() * sizeof input1_data[0]);

		std::vector<nic::InferInput*> myinputs = {input0_ptr.get()};

		nic::InferOptions myoptions(model_name);
		myoptions.partitioning_point_ = index_set[randindex];


		RETURN_IF_ERROR(client_.http_client_->AsyncInfer(
					callback, myoptions, myinputs, outputs, *http_headers_));
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::StartStream(
		nic::InferenceServerClient::OnCompleteFn callback)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->StartStream(
					callback, true /*enable_stats*/, *http_headers_));
	} else {
		return nic::Error("HTTP does not support starting streams");
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::AsyncStreamInfer(
		const nic::InferOptions& options,
		const std::vector<nic::InferInput*>& inputs,
		const std::vector<const nic::InferRequestedOutput*>& outputs)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(
				client_.grpc_client_->AsyncStreamInfer(options, inputs, outputs));
	} else {
		return nic::Error("HTTP does not support streaming inferences");
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::ClientInferStat(nic::InferStat* infer_stat)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->ClientInferStat(infer_stat));
	} else {
		RETURN_IF_ERROR(client_.http_client_->ClientInferStat(infer_stat));
	}
	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::ModelInferenceStatistics(
		std::map<ModelIdentifier, ModelStatistics>* model_stats,
		const std::string& model_name, const std::string& model_version)
{
	if (protocol_ == ProtocolType::GRPC) {
		ni::ModelStatisticsResponse infer_stat;
		RETURN_IF_ERROR(client_.grpc_client_->ModelInferenceStatistics(
					&infer_stat, model_name, model_version, *http_headers_));
		ParseStatistics(infer_stat, model_stats);
	} else {
		std::string infer_stat;
		RETURN_IF_ERROR(client_.http_client_->ModelInferenceStatistics(
					&infer_stat, model_name, model_version, *http_headers_));
		rapidjson::Document infer_stat_json;
		RETURN_IF_ERROR(nic::ParseJson(&infer_stat_json, infer_stat));
		ParseStatistics(infer_stat_json, model_stats);
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::UnregisterAllSharedMemory()
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(
				client_.grpc_client_->UnregisterSystemSharedMemory("", *http_headers_));
		RETURN_IF_ERROR(
				client_.grpc_client_->UnregisterCudaSharedMemory("", *http_headers_));
	} else {
		RETURN_IF_ERROR(
				client_.http_client_->UnregisterSystemSharedMemory("", *http_headers_));
		RETURN_IF_ERROR(
				client_.http_client_->UnregisterCudaSharedMemory("", *http_headers_));
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::RegisterSystemSharedMemory(
		const std::string& name, const std::string& key, const size_t byte_size)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->RegisterSystemSharedMemory(
					name, key, byte_size, 0 /* offset */, *http_headers_));

	} else {
		RETURN_IF_ERROR(client_.http_client_->RegisterSystemSharedMemory(
					name, key, byte_size, 0 /* offset */, *http_headers_));
	}

	return nic::Error::Success;
}

	nic::Error
TritonClientWrapper::RegisterCudaSharedMemory(
		const std::string& name, const cudaIpcMemHandle_t& handle,
		const size_t byte_size)
{
	if (protocol_ == ProtocolType::GRPC) {
		RETURN_IF_ERROR(client_.grpc_client_->RegisterCudaSharedMemory(
					name, handle, 0 /*device id*/, byte_size, *http_headers_));

	} else {
		RETURN_IF_ERROR(client_.http_client_->RegisterCudaSharedMemory(
					name, handle, 0 /*device id*/, byte_size, *http_headers_));
	}

	return nic::Error::Success;
}

	void
TritonClientWrapper::ParseStatistics(
		ni::ModelStatisticsResponse& infer_stat,
		std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
	model_stats->clear();
	for (const auto& this_stat : infer_stat.model_stats()) {
		auto it = model_stats
			->emplace(
					std::make_pair(this_stat.name(), this_stat.version()),
					ModelStatistics())
			.first;
		it->second.inference_count_ = this_stat.inference_count();
		it->second.execution_count_ = this_stat.execution_count();
		it->second.success_count_ = this_stat.inference_stats().success().count();
		it->second.cumm_time_ns_ = this_stat.inference_stats().success().ns();
		it->second.queue_time_ns_ = this_stat.inference_stats().queue().ns();
		it->second.compute_input_time_ns_ =
			this_stat.inference_stats().compute_input().ns();
		it->second.compute_infer_time_ns_ =
			this_stat.inference_stats().compute_infer().ns();
		it->second.compute_output_time_ns_ =
			this_stat.inference_stats().compute_output().ns();
	}
}

	void
TritonClientWrapper::ParseStatistics(
		rapidjson::Document& infer_stat,
		std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
	model_stats->clear();
	for (const auto& this_stat : infer_stat["model_stats"].GetArray()) {
		auto it = model_stats
			->emplace(
					std::make_pair(
						this_stat["name"].GetString(),
						this_stat["version"].GetString()),
					ModelStatistics())
			.first;
		it->second.inference_count_ = this_stat["inference_count"].GetUint64();
		it->second.execution_count_ = this_stat["execution_count"].GetUint64();
		it->second.success_count_ =
			this_stat["inference_stats"]["success"]["count"].GetUint64();
		it->second.cumm_time_ns_ =
			this_stat["inference_stats"]["success"]["ns"].GetUint64();
		it->second.queue_time_ns_ =
			this_stat["inference_stats"]["queue"]["ns"].GetUint64();
		it->second.compute_input_time_ns_ =
			this_stat["inference_stats"]["compute_input"]["ns"].GetUint64();
		it->second.compute_infer_time_ns_ =
			this_stat["inference_stats"]["compute_infer"]["ns"].GetUint64();
		it->second.compute_output_time_ns_ =
			this_stat["inference_stats"]["compute_output"]["ns"].GetUint64();
	}
}

//==============================================================================

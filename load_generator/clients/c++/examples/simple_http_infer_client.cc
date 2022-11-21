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

#include <fstream>
#include <unistd.h>
#include <iostream>
#include <string>
#include "src/clients/c++/library/http_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
{                                                                \
	nic::Error err = (X);                                          \
	if (!err.IsOk()) {                                             \
		std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
		exit(1);                                                     \
	}                                                              \
}

	int
main(int argc, char** argv)
{
	bool verbose = false;
	std::string url("210.107.197.107:8000");
	nic::Headers http_headers;

	// We use a simple model that takes 2 input tensors of 16 integers
	// each and returns 2 output tensors of 16 integers each. One output
	// tensor is the element-wise sum of the inputs and one output is
	// the element-wise difference.
	std::string model_name = argv[1];
	int mode = atoi(argv[2]);

	std::vector<std::vector<int64_t>> shapes;
	std::vector<float> index_set;
	std::vector<float> recored_time;
	if(model_name == "b5"){

		//b5
		//randindex = 0;

		std::vector<int64_t> shape1{1,3,224,224};
		std::vector<int64_t> shape2{1,40,56,56};
		std::vector<int64_t> shape3{1,64,28,28};
		std::vector<int64_t> shape4{1,128,14,14};
		std::vector<int64_t> shape5{1,176,14,14};
		std::vector<int64_t> shape6{1, 304, 7,7};
		std::vector<int64_t> shape7{1, 512, 7,7};
		std::vector<int64_t> shape8{1,2048,7,7};
		std::vector<int64_t> shape9{1,2048,1,1};

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
		index_set.push_back((float)6);
		index_set.push_back((float)11);
		index_set.push_back((float)16);
		index_set.push_back((float)23);
		index_set.push_back((float)30);
		index_set.push_back((float)39);
		index_set.push_back((float)42);
		index_set.push_back((float)44);

		recored_time.push_back(24.6795);
		recored_time.push_back(26.0826);
		recored_time.push_back(24.764);
		recored_time.push_back(18.1384);
		recored_time.push_back(13.5856);
		recored_time.push_back(9.50659);
		recored_time.push_back(4.96948);
		recored_time.push_back(3.04489);
		recored_time.push_back(3.13036);

	}
	std::string model_version = "";

	// Create a InferenceServerHttpClient instance to communicate with the
	// server using HTTP protocol.
	std::unique_ptr<nic::InferenceServerHttpClient> client;
	FAIL_IF_ERR(
			nic::InferenceServerHttpClient::Create(&client, url, verbose),
			"unable to create http client");




	if(mode == 0) // CHECK time for each partitioning point
	{
		std::vector<float> average_time;

		for(unsigned int point = 0; point < index_set.size(); point++)
		{
			// Create the data for the two input tensors. Initialize the first
			// to unique integers and the second to all ones.

			unsigned int N = shapes[point][0];
			unsigned int C = shapes[point][1];
			unsigned int H = shapes[point][2];
			unsigned int W = shapes[point][3];

			std::vector<float> input0_data(N*C*H*W);
			for (size_t i = 0; i < N*C*H*W; ++i) {
				input0_data[i] = (float)i;
			}
			std::vector<int64_t> shape{N,C,H,W};

			// Initialize the inputs with the data.
			nic::InferInput* input0;

			FAIL_IF_ERR(
					nic::InferInput::Create(&input0, "INPUT__0", shape, "FP32"),
					"unable to get INPUT0");
			std::shared_ptr<nic::InferInput> input0_ptr;
			input0_ptr.reset(input0);

			FAIL_IF_ERR(
					input0_ptr->AppendRaw(
						reinterpret_cast<uint8_t*>(&input0_data[0]),
						input0_data.size() * sizeof(int32_t)),
					"unable to set data for INPUT0");

			// Generate the outputs to be requested.
			nic::InferRequestedOutput* output0;

			FAIL_IF_ERR(
					nic::InferRequestedOutput::Create(&output0, "OUTPUT__0"),
					"unable to get 'OUTPUT0'");
			std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
			output0_ptr.reset(output0);

			// The inference settings. Will be using default for now.
			nic::InferOptions options(model_name);
			options.partitioning_point_ = index_set[point];

			std::vector<nic::InferInput*> inputs = {input0_ptr.get()};
			std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get()};

			float sum = 0;
			for(unsigned iter = 0; iter < 1000 ; iter++)
			{
				nic::InferResult* results;
				FAIL_IF_ERR(
						client->Infer(&results, options, inputs, outputs, http_headers),
						"unable to run model");
				std::shared_ptr<nic::InferResult> results_ptr;
				results_ptr.reset(results);

				std::string queue_ns;
				results_ptr->ModelQueueNs(&queue_ns);

				double queue_ms = stod(queue_ns)/1000/1000;
				std::cout << "queue " <<queue_ms << std::endl;

				std::string infer_ns;
				results_ptr->ModelInferNs(&infer_ns);
				double infer_ms = stod(infer_ns)/1000/1000;
				std::cout << "infer " <<infer_ms << std::endl;
				sum += queue_ms + infer_ms;
				// Get pointers to the result returned...
				float* output0_data;
				size_t output0_byte_size;
				FAIL_IF_ERR(
						results_ptr->RawData(
							"OUTPUT__0", (const uint8_t**)&output0_data, &output0_byte_size),
						"unable to get result data for 'OUTPUT0'");

				std::cout << output0_byte_size << std::endl;

				int max = 0;
				for(int i = 0; i < (int)output0_byte_size/4 ;i=i+4)
				{
					float value = 0;
					int index = i;
					unsigned char b[] = { (const uint8_t)output0_data[index], (const uint8_t)output0_data[index + 1], (const uint8_t)output0_data[index + 2], (const uint8_t) output0_data[index + 3] }; //4byte
					memcpy(&value, &b, sizeof(float));

					if(output0_data[i] >= output0_data[max])
					{
						max = i;
					}
				}




				std::cout << "MAX : " << max << std::endl;
				std::cout << "PASS : Infer" << std::endl;
			}//iter
			average_time.push_back(sum/(1000));
		}//point

		for(unsigned int i = 0; i < average_time.size(); i++)
		{
			std::cout << average_time[i] << std::endl;
		}



	}//mode0
	else if (mode == 1)
	{
		float SF = 1;
		int previous_point = 0;
		std::ofstream ofile(argv[3]);

		for(unsigned int iter = 0; iter< (unsigned int)atoi(argv[4]); iter++)
		{
			// rand() % (last value - first value + 1) + first value	
			int point = rand() % (8-0+1) + 0;

			unsigned int N = shapes[point][0];
			unsigned int C = shapes[point][1];
			unsigned int H = shapes[point][2];
			unsigned int W = shapes[point][3];

			std::vector<float> input0_data(N*C*H*W);
			for (size_t i = 0; i < N*C*H*W; ++i) {
				input0_data[i] = (float)i;
			}
			std::vector<int64_t> shape{N,C,H,W};

			// Initialize the inputs with the data.
			nic::InferInput* input0;

			FAIL_IF_ERR(
					nic::InferInput::Create(&input0, "INPUT__0", shape, "FP32"),
					"unable to get INPUT0");
			std::shared_ptr<nic::InferInput> input0_ptr;
			input0_ptr.reset(input0);

			FAIL_IF_ERR(
					input0_ptr->AppendRaw(
						reinterpret_cast<uint8_t*>(&input0_data[0]),
						input0_data.size() * sizeof(int32_t)),
					"unable to set data for INPUT0");

			// Generate the outputs to be requested.
			nic::InferRequestedOutput* output0;

			FAIL_IF_ERR(
					nic::InferRequestedOutput::Create(&output0, "OUTPUT__0"),
					"unable to get 'OUTPUT0'");
			std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
			output0_ptr.reset(output0);

			// The inference settings. Will be using default for now.
			nic::InferOptions options(model_name);
			options.partitioning_point_ = index_set[point];

			std::vector<nic::InferInput*> inputs = {input0_ptr.get()};
			std::vector<const nic::InferRequestedOutput*> outputs = {output0_ptr.get()};

			nic::InferResult* results;
			FAIL_IF_ERR(
					client->Infer(&results, options, inputs, outputs, http_headers),
					"unable to run model");
			std::shared_ptr<nic::InferResult> results_ptr;
			results_ptr.reset(results);

			std::string queue_ns;
			results_ptr->ModelQueueNs(&queue_ns);

			double queue_ms = stod(queue_ns)/1000/1000;

			std::string infer_ns;
			results_ptr->ModelInferNs(&infer_ns);
			double infer_ms = stod(infer_ns)/1000/1000;

			SF = (infer_ms+queue_ms)/recored_time[point];
			//sum += queue_ms + infer_ms;
			// Get pointers to the result returned...
			float* output0_data;
			size_t output0_byte_size;
			FAIL_IF_ERR(
					results_ptr->RawData(
						"OUTPUT__0", (const uint8_t**)&output0_data, &output0_byte_size),
					"unable to get result data for 'OUTPUT0'");

			int max = 0;
			for(int i = 0; i < (int)output0_byte_size/4 ;i=i+4)
			{
				float value = 0;
				int index = i;
				unsigned char b[] = { (const uint8_t)output0_data[index], (const uint8_t)output0_data[index + 1], (const uint8_t)output0_data[index + 2], (const uint8_t) output0_data[index + 3] }; //4byte
				memcpy(&value, &b, sizeof(float));

				if(output0_data[i] >= output0_data[max])
				{
					max = i;
				}
			}

			std::cout << "MAX : " << max << std::endl;
			float expected = recored_time[point] * SF;
			if (ofile.is_open()) {
				std::cout << previous_point << "," << point << "," << SF << "," << expected << "," << queue_ms<< "," <<infer_ms << "," << queue_ms+infer_ms << std::endl;
				ofile << previous_point << "," << point << "," << SF << "," << expected << "," << queue_ms<< "," <<infer_ms << "," << queue_ms+infer_ms << std::endl;
			}

			previous_point = point;
		}//iter
	
				ofile.close();
	}//mode1


	return 0;
}

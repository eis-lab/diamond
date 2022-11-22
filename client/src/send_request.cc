#include "send_request.h"
#include "Server.h"

struct tcp_info send_infer(std::string model_name, std::vector<float> serverside_input, int partitioning_point, 
		std::vector<int64_t> serverside_shape, struct diamond_results &diamond_result, ServerInfo &server_info)
{
	nic::Headers http_headers;
	std::unique_ptr<nic::InferenceServerHttpClient> client;
	FAIL_IF_ERR(
			nic::InferenceServerHttpClient::Create(&client, URL, false),
			"unable to create http client");
	nic::InferInput* input;

	FAIL_IF_ERR(nic::InferInput::Create(&input, "INPUT__0", serverside_shape, "FP32"),"unable to get INPUT0");
	std::shared_ptr<nic::InferInput> input_ptr;
	input_ptr.reset(input);

	//FAIL_IF_ERR(input_ptr->AppendRaw(serverside_input), "unable to set data for INPUT0"); // for uint8 vector

	FAIL_IF_ERR(
			input_ptr->AppendRaw(
				reinterpret_cast<uint8_t*>(&serverside_input[0]),
				serverside_input.size() * sizeof(int32_t)),
			"unable to set data for INPUT0");

	// Generate the outputs to be requested.
	nic::InferRequestedOutput* output;

	FAIL_IF_ERR(
			nic::InferRequestedOutput::Create(&output, "OUTPUT__0"),
			"unable to get 'OUTPUT0'");
	std::shared_ptr<nic::InferRequestedOutput> output_ptr;
	output_ptr.reset(output);

	// The inference settings. Will be using default for now.
	nic::InferOptions options(model_name);
	options.partitioning_point_ = partitioning_point;

	std::vector<nic::InferInput*> inputs = {input_ptr.get()};
	std::vector<const nic::InferRequestedOutput*> outputs = {output_ptr.get()};

	nic::InferResult* results;
	struct tcp_info ret = client->Infer_with_tcpinfo(&results, options, inputs, outputs, http_headers);

	uint64_t end = get_current_unixtime();

	std::shared_ptr<nic::InferResult> results_ptr;
	results_ptr.reset(results);

	std::string queue_ns;
	std::string infer_ns;
	results_ptr->ModelQueueNs(&queue_ns);
	results_ptr->ModelInferNs(&infer_ns);

/*
	results_ptr->ModelQueueContents(&queue_contents);
	results_ptr->ModelNumOfBatch(&num_of_batch);
	results_ptr->ModelArrivalRate(&arrival_rate);
	results_ptr->ModelLastInference(&last_batch_size, &last_partitioning_point);
*/

	double queue_ms;
	double infer_ms;
	if(stod(queue_ns) > 10000000)
		std::cout << queue_ns << "                  <<" << std::endl;
	else
		std::cout << queue_ns << std::endl;
	queue_ms = stod(queue_ns)/1000/1000;
	infer_ms = stod(infer_ns)/1000/1000;	
	diamond_result.queue_ms = queue_ms;
	diamond_result.infer_ms = infer_ms;
	std::string str_bw;
	results_ptr->ModelVersion(&diamond_result.server_capacity);
	results_ptr->ModelQueueContents(&diamond_result.queue_contents);
	results_ptr->ModelNumOfBatch(&diamond_result.num_of_batch);
	results_ptr->ModelArrivalRate(&diamond_result.arrival_rate);
	results_ptr->ModelLastInference(&diamond_result.last_batch_size, &diamond_result.last_partitioning_point, &diamond_result.last_inference_start, &diamond_result.current_inference_start, &diamond_result.request_enqueue_time);
	std::vector <double> percentile=  parseStrToDoubleVec(diamond_result.queue_contents);
	server_info.percentile.clear();
	server_info.percentile.assign(percentile.begin(), percentile.end());
	
	double a,b,c,d,e;
	results_ptr->ModelServerStatus(&a, &b, &c, &d, &e);
	
	server_info.SetServerInfo(a,b,c,d,e);
	
		server_info.last_batch = c;
	server_info.last_server_infertime = infer_ms;

	// Get pointers to the result returned...
	
	//struct tcp_info info;
        //memcpy(&info, &results_ptr->info, sizeof(struct tcp_info));	
	//std::cout << "rtt : " << info.tcpi_rtt << std::endl;
	
	float* output_data;
	size_t output_byte_size;
	FAIL_IF_ERR(
			results_ptr->RawData(
				"OUTPUT__0", (const uint8_t**)&output_data, &output_byte_size),
			"unable to get result data for 'OUTPUT0'");


	int max = 0;
	for(int i = 0; i < (int)output_byte_size/4 ;i=i+4)
	{
		float value = 0;
		int index = i;
		unsigned char b[] = { (const uint8_t)output_data[index], (const uint8_t)output_data[index + 1], (const uint8_t)output_data[index + 2], (const uint8_t) output_data[index + 3] }; //4byte
		memcpy(&value, &b, sizeof(float));

		if(output_data[i] >= output_data[max])
		{
			max = i;
		}
	}
	
	diamond_result.top1 = max;
	uint64_t before_return = get_current_unixtime();
	
	std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>result snd_cwnd " <<ret.tcpi_snd_cwnd << "rtt " << ret.tcpi_rtt<< "sstshresh" << ret.tcpi_snd_ssthresh<< std::endl;
	return ret;
}

void infer_result_analysis(struct diamond_results return_diamond_result, double *queue_ms, double *infer_ms, int *top1, std::string &queue_contents, std::string &num_of_batch, std::string &arrival_rate, std::string &last_batch_size, std::string &last_partitioning_point, std::string &last_inference_start, std::string &current_inference_start, std::string &request_enqueue_time, int *server_capacity)
{
	*queue_ms = return_diamond_result.queue_ms;
	*infer_ms = return_diamond_result.infer_ms;
	*top1 = return_diamond_result.top1;
	queue_contents = return_diamond_result.queue_contents;
	num_of_batch = return_diamond_result.num_of_batch;
	arrival_rate = return_diamond_result.arrival_rate;
	last_batch_size = return_diamond_result.last_batch_size;
	last_partitioning_point = return_diamond_result.last_partitioning_point;
	last_inference_start = return_diamond_result.last_inference_start;
	current_inference_start = return_diamond_result.current_inference_start;
	request_enqueue_time = return_diamond_result.request_enqueue_time;
	std::cout << "ss" << return_diamond_result.server_capacity << std::endl;
	*server_capacity = std::stoi(return_diamond_result.server_capacity);
}

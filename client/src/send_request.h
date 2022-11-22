#ifndef SEND_REQUEST_H
#define SEND_REQUEST_H

#include "../include/http_client.h"
#include <vector>
#include "util.h"
#include <curl/curl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <string>
#include "Server.h"
#define URL "210.107.197.107:8000"
namespace nic = nvidia::inferenceserver::client;
namespace ni = nvidia::inferenceserver;


struct diamond_results
{
	int top1;
	
	double queue_ms;
	double infer_ms;
	
	std::string queue_contents;
	std::string num_of_batch;
	std::string arrival_rate;

	std::string last_batch_size;
	std::string last_partitioning_point;
	std::string last_inference_start;
	std::string current_inference_start;	
	
	std::string request_enqueue_time;
	std::string server_capacity; 
};

struct tcp_info send_infer(std::string model_name, std::vector<float> serverside_input, int partitioning_point, std::vector<int64_t> serverside_shape, struct diamond_results &result, ServerInfo &server_info);
		
void infer_result_analysis(struct diamond_results return_diamond_result, double *queue_ms, double *infer_ms, int *top1, std::string &queue_contents, std::string &num_of_batch, std::string &arrival_rate, std::string &last_batch_size, std::string &last_partitioning_point, std::string &last_inference_start, std::string &current_inference_start, std::string &request_enqueue_time, int *server_capacity);

#endif

		//, double &queue_ms, double &infer_ms, int &top1, std::string &queue_contents, 
		//std::string &num_of_batch, std::string &arrival_rate, std::string &last_batch_size, std::string &last_partitioning_point);

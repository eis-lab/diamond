#include <iostream>
#include <fstream>
#include <unistd.h>
#include "Model.h"
#include "image_processing.h"
#include "util.h"
#include "local_execution.h"
#include "send_request.h"
#include "Server.h"
#include "comm_online_profiler.h"
#include "partitioner.h"
#include <mutex>
#include <thread>
#include "server_profiler.h"
extern std::string material_path;
int  layer_length;
std::vector<int> myhistory;
std::vector<int> point_history;
std::vector<uint64_t> send_history;
std::mutex mu;
std::condition_variable cv_;
std::mutex m_;
/*
void manage_history(uint64_t refresh_time)
{
	int old_data_index=0;
	for(int i = 0; i < send_history.size(); i++)
	{
		if(send_history[i] < refresh_time-1000)
		{
			old_data_index = i;
		}
	}
	point_history.erase(point_history.begin(), point_history.begin() + old_data_index);
	send_history.erase(send_history.begin(), send_history.begin() + (uint64_t)old_data_index);
	myhistory.clear();
	myhistory.resize(layer_length, 0);	

	for(int i = 0; i < point_history.size(); i++)
	{
		myhistory[point_history[i]] +=1;
	}
	for(int i = 0; i < myhistory.size(); i++)
	{
		myhistory[i] =(int) std::ceil( (double)myhistory[i] / 10.0);
	}
}
*/
bool done = false;
Communication comm(91);
bool wakeup = false;
void baseline_get_serverinfo(ModelInfo *model_info, ServerInfo *server_info, int run_policy)
{
	if(run_policy !=4 && run_policy !=5)
	{
		return;
	}
	std::unique_lock<std::mutex> lock(m_);
	while(1)
	{
	
		if(!wakeup)
		{
			cv_.wait(lock, [] {return wakeup;});
		}
			if(done)
		{
			return;
		}
		std::cout << "RUN BACKGROUND!!!!!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
		int partitioning_point = 0;
		std::vector<int64_t> serverside_shape = model_info->shapes[partitioning_point];
		std::vector<float> serverside_input(vector_mul(serverside_shape), 1.0);

		struct diamond_results return_diamond_result;

		auto remote_start = get_current_unixtime();
		struct tcp_info	ret_tcp_info = send_infer(model_info->model_name, serverside_input, partitioning_point, serverside_shape, return_diamond_result, *server_info);

		auto remote_end = get_current_unixtime();

		auto remote_elapsed_time = remote_end - remote_start;
		double	queue_ms = return_diamond_result.queue_ms;
		double	infer_ms = return_diamond_result.infer_ms;

		double comm_ms = remote_elapsed_time - queue_ms - infer_ms;	

		double expected_LINK =( (vector_mul(serverside_shape)*4*8)/(comm_ms/1000))/1024/1024;
		

		mu.lock();

		std::cout << "before sf " << server_info->sf << std::endl;
		std::cout << "before link " << server_info->link_capacity << std::endl;

		comm.LINK = expected_LINK;
		server_info->sf =  (infer_ms + queue_ms) / model_info->server_inference_time_ms[partitioning_point];

		server_info->link_capacity = expected_LINK;
		std::cout << "new link " << server_info->link_capacity << std::endl;
		std::cout << "new sf " << server_info->sf << std::endl;


		mu.unlock();
	
	}

}
double harmonic_mean(std::vector<double> v){
	double sum = 0;
	for (int i = 0; i <v.size(); i++)
		sum = sum + (double)1 / v[i];
	return (double)v.size()/sum;
}

int main(int argc, char** argv)
{
	std::cout << "pkill -9 perf_client" << std::endl;
	system("ssh LOADSERVERURL \"pkill -9 perf_client\"");
	std::cout << "sleep 10s" << std::endl;
	sleep(10);

	system("./clear_latency.sh");
	//argv[1] = model name
	//argv[2] = model_path
	//argv[3] = material_path	
	//argv[4] = output file path
	//argv[5] = SLO
	//std::string SLO_str = std::string(argv[5]);
	//double SLO = atof(argv[5]);
	
	int max_concurrency = atoi(argv[6]);
	int max_ping = atoi(argv[7]);
	int min_confidence = atoi(argv[8]);
	int min_SLO = atoi(argv[9]);
	int max_SLO = atoi(argv[10]);
	int run_policy = atoi(argv[5]);	
	material_path = std::string(argv[3]);
	std::string model_name = argv[1];	
	torch::jit::script::Module model = torch::jit::load(argv[2]);
	ServerInfo server_info;
	ModelInfo model_info(model_name); 
	std::thread t1(baseline_get_serverinfo, &model_info , &server_info , run_policy );		
	t1.detach();
	bool local_execution;
	server_info.GetServerInfo();
	comm.Init(server_info.rtt);

	layer_length =  model_info.layer_length;
	model.to(torch::kCUDA); model.eval();
	//Communication comm(atoi(argv[2]));
	loadimagenetlabel(material_path+"/imagenet_label", model_info.labels);

	torch::Tensor input_tensor;
	loadimage(material_path + "/zebra.jpg", input_tensor, 224);

	/*	std::chrono::steady_clock::time_point local_start = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point remote_start = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point remote_end = std::chrono::steady_clock::now();
		*/
	myhistory.resize(layer_length, 0);
	uint64_t task_start = 0;
	uint64_t local_start = 0;
	uint64_t remote_start = 0;
	uint64_t remote_end = 0;
	int previous_partitioning_point = 0;
	std::vector<double> LINK_history;
	for(int i = 0; i < model_info.shapes.size(); i++)
	{
		std::vector<int64_t> serverside_shape = model_info.shapes[i];
		at::Tensor local_output = execute_local_parts(model, input_tensor, i, serverside_shape);
		std::vector<float> serverside_input(local_output.data_ptr<float>(), local_output.data_ptr<float>() + local_output.numel());

	}
	for(int concurrency =0; concurrency <1+max_concurrency; concurrency = concurrency+50){
		std::cout << "pkill -9 perf_client" << std::endl;
		system("ssh LOADSERVERURL \"pkill -9 perf_client\"");
		std::cout << "sleep 10s" << std::endl;
		sleep(10);
		//std::vector<int> pings = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
		for(int ping=0; ping < 1 + max_ping; ping +=2){
		//for(int pingi=0; pingi < pings.size(); pingi +=1){
		//	int ping = pings[pingi];
			std::cout << "pkill -9 perf_client" << std::endl;
			system("ssh LOADSERVERURL \"pkill -9 perf_client\"");
			std::cout << "sleep 10s" << std::endl;
			sleep(10);

			system("./clear_latency.sh");
	

			std::string ping_command = "./set_latency.sh "+std::to_string(ping);
			if (ping !=0) system(ping_command.c_str()); // PING CHANGE
			/*if( ping == 0)
			{
				std::vector<int64_t> serverside_shape = model_info.shapes[1];
				std::vector<float> serverside_input(vector_mul(serverside_shape), 1.0);
				for(int i = 0; i < 200000; i++)
				{

					struct diamond_results return_diamond_result;
					struct tcp_info	ret_tcp_info = send_infer(model_name, serverside_input, 1, serverside_shape, return_diamond_result, server_info);
				}
			}
			*/
			if(concurrency != 0)
			{
				//			std::string command = " nohup ssh LOADSERVERURL \"/home/username/apps/client/server/src_batch/perf_client_all_rand -m "+ model_name+" -u 210.107.197.107:8000 --concurrency-range " + std::to_string(concurrency) +    " --measurement-interval 99999999\" &";
				std::string command;
			        if(run_policy != 3){
					std::cout << "server load all rand" << std::endl;
					command	= " nohup ssh LOADSERVERURL \"/home/username/apps/client/server/src_batch/perf_client_all_rand -m "+ model_name+" -u 210.107.197.107:8000 --request-distribution poisson --max-threads 256 --request-rate-range " + std::to_string(concurrency) +    " --measurement-interval 99999999\" &";
				}
				else if (run_policy == 3)
				{
					command	= " nohup ssh LOADSERVERURL \"/home/username/apps/client/server/src_batch/perf_client_only0 -m "+ model_name+" -u 210.107.197.107:8000 --request-distribution poisson --max-threads 256 --request-rate-range " + std::to_string(concurrency) +    " --measurement-interval 99999999\" &";
				}
				std::cout << command << std::endl;

				system(command.c_str());

				sleep(60);
			}

			for(int confidence_threshold = min_confidence; confidence_threshold < 81; confidence_threshold = confidence_threshold + 10){
				
					
				if(run_policy != 0 && run_policy !=1)
				{
					confidence_threshold = 100;
				}

				
				
				std::vector<int> target_policy = {run_policy};
				for(int policy_idx = 0; policy_idx < target_policy.size(); policy_idx++){
					int policy = target_policy[policy_idx];
					if(policy == 5 && confidence_threshold != 100)
					{
						continue;
					}
					for(double SLOs = min_SLO; SLOs <1 + max_SLO; SLOs=SLOs+20){
						std::string SLO_str = std::to_string((int)SLOs);
						double SLO = SLOs / 100.0;
						std::string path = std::string(argv[4]);
						//std::string path = std::string(argv[6]); //XXX
						std::ofstream fp;
						fp.open(path+"_logging_"+std::to_string(concurrency)+"_"+ std::to_string(policy) + "_" + SLO_str + "_"+ std::to_string(confidence_threshold)+"_"+ std::to_string(ping) +".csv");
						fp << "point, total, local, comm, queue, infer, slo, policy, interval, throughput, batch, avinfer, avqueue, bw, rtt, ex_comm, ex_queue, ex_infer, ex_time, slotime, ispolicyfailed, confidence_threshold , ping,  percentile\n";
						
						for(int i = 0 ; i < 1000 ; i++)
						{
							//		std::cout << "================================ iter " << i << "======================================" << std::endl;
							//double confidence_threshold = 90;
							
							struct partitioner_result r;
							task_start = get_current_unixtime(); 
							if(policy == 0 || policy == 1 || policy ==6 || policy == 7)
							{

								if(previous_partitioning_point == model_info.layer_length-1 || server_info.isMyInfoExpired()) // cold start
								{
									std::cout << "info update " << std::endl;
									server_info.GetServerInfo();
									comm.Init(server_info.rtt);
									LINK_history.clear();
									comm.LINK = std::min( (double)comm.MAX_LINK, (double)server_info.CURRENT_SERVER_CAPACITY );
									server_info.link_capacity = comm.LINK;
									server_info.last_batch = 0;
									server_info.last_server_infertime = 0;
								}
								local_execution = false;
								std::vector<double> estimated_comm = comm.expect_time_shapes(model_info.shapes, comm.LINK, server_info);


								r = get_partitioning_point(estimated_comm, model_info, server_info, policy, SLO, confidence_threshold);
							}
							else if(policy == 3)
							{

								local_execution = false;
								r.partitioning_point = 0;
							}
							else if (policy == 4 || policy == 5)
							{
								//if(previous_partitioning_point == model_info.layer_length-1)
								//{
								//	server_info.sf = 1;
								//	comm.LINK = 91;
								//}
								local_execution = false;
								
								wakeup = false;
								mu.lock();
								r = get_partitioning_point_baseline(comm.LINK, model_info, server_info, policy, SLO);
								mu.unlock();
								if(r.partitioning_point  == model_info.layer_length-1)
								{
//baseline_get_serverinfo(&model_info, &server_info);
									wakeup = true;
									cv_.notify_one();
									std::cout << "local executio n>>>>>>><<<<<<<<<<<<<<<<" << std::endl;
								}
								else
								{
									wakeup = false;
								}
															}	
							int partitioning_point = r.partitioning_point;
							/*
								if(i < 10) partitioning_point = 0;	
							*/
							
							if (partitioning_point == model_info.local_inference_time_ms.size()-1)
								local_execution = true;
							//local_start = std::chrono::steady_clock::now();
							local_start = get_current_unixtime();
							std::vector<int64_t> serverside_shape;
							at::Tensor local_output = execute_local_parts(model, input_tensor, partitioning_point, serverside_shape);
							std::vector<float> serverside_input(local_output.data_ptr<float>(), local_output.data_ptr<float>() + local_output.numel());
	/*	
							std::vector<int64_t> serverside_shape = model_info.shapes[partitioning_point];
							std::vector<float> serverside_input(vector_mul(serverside_shape), 1.0);
							std::cout << "sleep " << model_info.local_inference_time_ms[partitioning_point]*1000 << "us";
							usleep(model_info.local_inference_time_ms[partitioning_point]*1000);
	*/						
							double queue_ms = 0;	double infer_ms = 0;	double comm_ms = 0;
							uint64_t local_elapsed_time = 0; uint64_t remote_elapsed_time = 0; uint64_t total_elapsed_time = 0;

							if(!local_execution){
								struct diamond_results return_diamond_result;
								//remote_start = std::chrono::steady_clock::now();


								//RUN
									
								remote_start = get_current_unixtime();
								struct tcp_info	ret_tcp_info = send_infer(model_name, serverside_input, partitioning_point, serverside_shape, return_diamond_result, server_info);
								std::cout << "cwnd " << ret_tcp_info.tcpi_snd_cwnd << std::endl;
								server_info.isServerInfoExpiredResult = false;
								//std::cout << server_info.average_interval<< " " << server_info.average_throughput << " " <<
								//	server_info.average_batch << " " << server_info.average_infer_time << " " <<
								//	server_info.average_queue_time << std::endl;

								remote_end = get_current_unixtime();
								//server_info.server_queue_status = parseStrToIntVec(return_diamond_result.queue_contents);
								//server_info.server_arrival_rate = parseStrToIntVec(return_diamond_result.arrival_rate);
								server_info.server_information_refresh_time = remote_end;

								//std::cout << "server_queue_status size " << server_info.server_queue_status.size() << " " << "server arrival_rate size " <<  server_info.server_arrival_rate.size() << std::endl;
								//remote_end = std::chrono::steady_clock::now();	
							
									queue_ms = return_diamond_result.queue_ms;
								infer_ms = return_diamond_result.infer_ms;
								//	local_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(remote_start - local_start).count();
								//	remote_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(remote_end - remote_start).count();
								//	total_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(remote_end - local_start).count();
								local_elapsed_time = remote_start - local_start;
								remote_elapsed_time = remote_end - remote_start;
								total_elapsed_time = remote_end - local_start;

								comm_ms = remote_elapsed_time - queue_ms - infer_ms;	
								std::cout << "comm time " << comm_ms << std::endl;
								point_history.push_back( partitioning_point );
								send_history.push_back( remote_start + comm_ms );
							
								server_info.RTTrefresh((double)ret_tcp_info.tcpi_rtt/1000.0);
								ret_tcp_info.tcpi_rtt = server_info.rtt*1000;
								comm.set_tcp_info(ret_tcp_info);
								int expected_LINK;

								if(policy == 0 || policy == 1){
									expected_LINK = comm.get_LINK(vector_mul(serverside_shape)*4 , comm_ms);
									std::cout << "expected link " <<expected_LINK << std::endl;
								}
								else if(policy == 3 || policy == 4 || policy == 5 || policy == 6 || policy == 7)
								{
									expected_LINK =( (vector_mul(serverside_shape)*4*8)/(comm_ms/1000))/1024/1024;
									std::cout << "expected link " <<expected_LINK << std::endl;
								}


								if(LINK_history.size() == 10){
									LINK_history.erase(LINK_history.begin());
								}
								LINK_history.push_back((double)expected_LINK);


								double average_link_history = std::accumulate(LINK_history.begin(), LINK_history.end(), 0)/(double)LINK_history.size();
								//double average_link_history = harmonic_mean(LINK_history);
								comm.LINK = std::min( (double)average_link_history, std::stod(return_diamond_result.server_capacity));
																server_info.queueing = queue_ms;

								mu.lock();
								//manage_history(server_info.server_information_refresh_time);
								server_info.link_capacity = comm.LINK;
								server_info.sf = (infer_ms+queue_ms) / model_info.server_inference_time_ms[partitioning_point];


								mu.unlock();


							}
							else{
								uint64_t current_time = get_current_unixtime();
								local_elapsed_time = current_time - local_start;
								total_elapsed_time = local_elapsed_time ;
								//local_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(remote_start - local_start).count();
							}
							previous_partitioning_point = partitioning_point ;
							/*std::cout << "point " << partitioning_point << std::endl 
							  << " comm_ms (real / expected) " <<  comm_ms << " " << estimated_comm[partitioning_point] << " " << comm_ms / estimated_comm[partitioning_point] << std::endl
							  << "queue time " << queue_ms << " " << server_info.queueing << std::endl
							  << "infer time" <<   infer_ms << " " << server_info.sf * model_info.server_inference_time_ms[partitioning_point] << std::endl;

*/
							fp << partitioning_point << ", " << total_elapsed_time << "," <<  local_elapsed_time << "," << comm_ms << "," << queue_ms << "," << infer_ms << "," << SLO*model_info.local_only_time <<  "," << policy << "," << server_info.average_interval << " , "  << server_info.average_throughput << " ," << server_info.average_batch << ","<< server_info.average_infer_time << ", " << server_info.average_queue_time << "," << comm.LINK<< "," <<  comm.current_tcp_info.tcpi_rtt << "," << r.ex_comm << "," << r.ex_queue <<"," << r.ex_infer << "," << r.ex_time << "," << r.slo_time << "," << r.isPolicyFailed << "," << confidence_threshold << "," << ping << ",";

							for(int j = 0; j < server_info.percentile.size()-1 ; j++)
							{
								fp <<  server_info.percentile[j] << ",";
							}
							fp << server_info.percentile[server_info.percentile.size()-1]  << "," << server_info.sf << "," << task_start << "," << comm.current_tcp_info.tcpi_snd_ssthresh << ", " << comm.current_tcp_info.tcpi_snd_cwnd << "," << server_info.average_infer_time << "," << server_info.last_server_infertime << "," << model_info.server_inference_time_ms[partitioning_point] <<  "," << server_info.current_measured_rtt << std::endl;
							std::cout << "point " << partitioning_point << std::endl;

							bool isSLOViolation = total_elapsed_time > SLO*model_info.local_only_time; 	
							/*   
							     std::cout << "----------------- inference result --------------------" << std::endl;
							     std::cout << "partitioning point : " << partitioning_point<< std::endl;
							     std::cout << "total elapsed time (SLO) " << total_elapsed_time << "(" << SLO*model_info.local_only_time << ")" << std::endl;
							     std::cout << "SLO violation " << isSLOViolation <<std::endl;
							     std::cout << "local_elapsed_time " << local_elapsed_time << std::endl;
							     std::cout << "communication time " << comm_ms << std::endl;
							     std::cout << "queue time         " << queue_ms << std::endl;
							     std::cout << "infer time         " << infer_ms << std::endl;
							     std::cout << "Comm LINK          " << comm.LINK << std::endl;
							     std::cout << "Comm rtt           " << comm.current_tcp_info.tcpi_rtt << " ( " << comm.current_tcp_info.tcpi_rttvar << ")" << std::endl;
							     std::cout << "interval           " << server_info.average_interval << std::endl;	
							     std::cout << "---------------------------------------------------------" << std::endl;
							     std::cout << "==============================================================================" << std::endl;
							     */
						} // execution for loop
						fp.close();
					}//slo for loop
				} //policy for loop
			}//confidence
		}//concurrency for loop
	}
	std::cout << model_info.model_name << std::endl;
	done = true;
	wakeup=true;
	cv_.notify_one();
	return 0;
}

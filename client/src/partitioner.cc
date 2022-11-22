#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include "Model.h"
#include "util.h"
#include "Server.h"
#include <cmath>
#include "partitioner.h"

//policy 0 - fastest policy 1 - Min battery policy 2 - Min server computation
//

#define MAXVALUE 99999

double CDF(ServerInfo server_info, double x)
{
	if(x< 0)
		return 0;
	int start = 0;
	int end = 0;
	for(int i = 0; i < server_info.percentile.size(); i++)
	{
		if(server_info.percentile[i] <= x)
		{
			start = i;
			end = i+1;
		}
	}
	double ret = 0;
	if (start == end) return 0;
	else if(end == server_info.percentile.size()) return 100;
	else
	{
		ret = (start+1)*10 + (x-server_info.percentile[start])/((server_info.percentile[end]-server_info.percentile[start])/10);
	}
	return ret;
}
std::vector<std::vector<double>> gen_probs(std::vector<double> communication_time_ms, ModelInfo model_info, ServerInfo server_info, std::vector<double>& ex_infer)
{

	std::vector<std::vector<double>> probs;
	std::vector<double> target_times;
	for(double i = 5; i < 101; i=i+5)
		{
			target_times.push_back(model_info.local_only_time * (i/100));
		}
	for(int i = 0; i < target_times.size(); i++)
	{
		std::vector<double> prob;

		//std::cout << i << " ";
		for(int j = 0; j < model_info.local_inference_time_ms.size()-1; j++)
		{
			double withoutqueue;
		        if(!server_info.isServerInfoExpiredResult) // server is activated...
			{
				//withoutqueue = model_info.local_inference_time_ms[j] + communication_time_ms[j] + server_info.average_infer_time;
				if(server_info.last_batch == 1){
				//	std::cout << "last batch is 1" <<server_info.last_batch << std::endl;
					withoutqueue = model_info.local_inference_time_ms[j] + communication_time_ms[j] + model_info.server_inference_time_ms[j];
					if(i == 0) ex_infer.push_back(model_info.server_inference_time_ms[j]);
				}
				else if(server_info.last_batch == 0) // cold start
				{
				//	std::cout << "cold start use average infer " <<std::endl;
					withoutqueue = model_info.local_inference_time_ms[j] + communication_time_ms[j];
					if(server_info.average_infer_time < model_info.server_inference_time_ms[j])
					{
						withoutqueue += model_info.server_inference_time_ms[j];

						if (i == 0) ex_infer.push_back(model_info.server_inference_time_ms[j]);
					}
					else
					{
						withoutqueue += server_info.average_infer_time;
						if (i == 0) ex_infer.push_back(server_info.average_infer_time);
					}
				}
				else{
				//	std::cout << "use previous time" << server_info.last_batch << std::endl;
					withoutqueue = model_info.local_inference_time_ms[j] + communication_time_ms[j];
				       if(server_info.last_server_infertime < model_info.server_inference_time_ms[j])
				       {
					 withoutqueue += model_info.server_inference_time_ms[j];
				       
					if(i == 0) ex_infer.push_back(model_info.server_inference_time_ms[j]);
				       }
				       else
				       {
					       withoutqueue += server_info.last_server_infertime;
					       if(i == 0) ex_infer.push_back(server_info.last_server_infertime);
				       }
				
				}
			}
			else
			{
				withoutqueue = model_info.local_inference_time_ms[j] + communication_time_ms[j] + model_info.server_inference_time_ms[j];
				
				if(i == 0) ex_infer.push_back(model_info.server_inference_time_ms[j]);
			}	
			double remain_time = target_times[i] - withoutqueue;
			prob.push_back(CDF(server_info, remain_time));
//		std::cout << "<" << j << "," <<  CDF(server_info, remain_time) << " " << withoutqueue-communication_time_ms[j]-model_info.local_inference_time_ms[j]  << "," << communication_time_ms[j] << "> ";
		}
		if(target_times[i] < model_info.local_only_time)	prob.push_back(0);
		else	prob.push_back(100);
		probs.push_back(prob);
		//std::cout << std::endl;
	}
	return probs;
}
double comm_power(double datasize_bit, double time)
{
	if (time == 0 || datasize_bit==0)
		return 0;
	double th = datasize_bit/time/1000;
	
	return ((283.17*th+132.86)*time) / 1000000; //j
}

struct partitioner_result get_partitioning_point_baseline(int link, ModelInfo model_info, ServerInfo server_info, int policy, double SLO)
{
	server_info.GetServerInfoNoRefresh();
	std::vector<double> comm_time;
	std::vector<double> server_time;
	std::vector<double> expected_inference_time;
	std::vector<double> power_estimation;

	double deadline = model_info.local_only_time * (double)SLO;
	
	for(int i = 0; i < model_info.shapes.size(); i++)
	{
		double datasize_bits = vector_mul(model_info.shapes[i]) *4 *8;
		double expected_comm = datasize_bits / ((double)link * 1024 * 1024);
		comm_time.push_back(expected_comm * 1000);
	}

	for(int i = 0; i < model_info.server_inference_time_ms.size(); i++)
	{
		double expected_server = model_info.server_inference_time_ms[i] * server_info.sf;
		server_time.push_back(expected_server);
	}
	server_time.push_back(0); // local only

	if(comm_time.size() != server_time.size())
	{
		std::cout << " comm server size wrong " << comm_time.size() << " " << server_time.size() <<std::endl;
		exit(-1);
	}
	if(model_info.local_inference_time_ms.size() != server_time.size())
	{
		std::cout << "local server size wrong " << model_info.local_inference_time_ms.size() << " " << server_time.size() <<std::endl;
		exit(-1);
	}


	for(int i = 0; i < comm_time.size(); i++)
	{
		double expected_time = comm_time[i] + server_time[i] + model_info.local_inference_time_ms[i];
	//	std::cout << i << " " << comm_time[i] << "  " <<  server_time[i] << " " << model_info.local_inference_time_ms[i] << " " << expected_time <<  " " << link <<std::endl;
		double sum = comm_power(vector_mul(model_info.shapes[i])*4*8, comm_time[i]) + model_info.local_power_consumption_J[i];
		power_estimation.push_back(sum);
 
		expected_inference_time.push_back(expected_time);
	}
	

	struct partitioner_result r;
	if(policy == 4){
		r.partitioning_point = std::min_element(expected_inference_time.begin(), expected_inference_time.end())-expected_inference_time.begin();	
	}
	if(policy == 5)
	{
		std::vector<int> candidated_point;
		std::vector<double> candidated_point_power ;

		for(int i = 0; i < expected_inference_time.size();i++)
		{
			if (expected_inference_time[i] < deadline)
			{
				candidated_point.push_back(i);
				candidated_point_power.push_back(power_estimation[i]);
			}
		}
		if(candidated_point.size() != 0)
		{
			int low_power_point = std::min_element(candidated_point_power.begin(), candidated_point_power.end())-candidated_point_power.begin();	
			r.partitioning_point = candidated_point[low_power_point];
		}
		else
		{

			r.partitioning_point = std::min_element(expected_inference_time.begin(), expected_inference_time.end())-expected_inference_time.begin();	
			///r.partitioning_point = std::min_element(power_estimation.begin(), power_estimation.end())-power_estimation.begin();
			r.isPolicyFailed = 1;
		}
	}
	
	std::cout << "partitioning_point " << r.partitioning_point << std::endl; 

	return r;
}
struct partitioner_result get_partitioning_point(std::vector<double> communication_time_ms, ModelInfo model_info, ServerInfo server_info,  int policy, double SLO, double prob_threshold)
{
	if(server_info.isServerInfoExpiredResult)
	{
		server_info.ResetServerInfo();
	}
	
	std::vector<double> comm_time;	
	std::vector<std::vector<double>> probs;
	

	if (policy == 0 || policy == 1 || policy ==2)
	{
		comm_time.assign(communication_time_ms.begin(), communication_time_ms.end());
	}
	else
	{
		for(int i = 0; i < model_info.shapes.size(); i++)
		{
			double datasize_bits = vector_mul(model_info.shapes[i]) *4 *8;
			double expected_comm = datasize_bits / ((double)server_info.link_capacity * 1024 * 1024);
			comm_time.push_back(expected_comm * 1000);
		}

	}	
	std::vector<double> ex_infer;	
	probs 	= gen_probs(comm_time, model_info , server_info, ex_infer);
	
	std::cout << "ex_comm :";
	for(int i = 0; i < comm_time.size(); i++)
	{
		std::cout << comm_time[i] << ",";
	}	
	std::cout << std::endl;

	std::cout << "ex_infer :";
	for(int i = 0; i < ex_infer.size(); i++)
	{
		std::cout << ex_infer[i] << ",";
	}	
	std::cout << std::endl;
/*
	std::cout << "threshold " << prob_threshold << std::endl;
	std::cout << "local :";
	for(int i = 0; i < model_info.local_inference_time_ms.size(); i++)
	{
		std::cout << model_info.local_inference_time_ms[i] << ",";
	}	
	std::cout << std::endl;

	std::cout << "percentile :";
	for(int i = 0; i < server_info.percentile.size(); i++)
	{
		std::cout << server_info.percentile[i] << ",";
	}	
	std::cout << std::endl;

	std::cout << "server " << server_info.average_infer_time << std::endl;
	*/
	struct partitioner_result r;
	int point = -1;
	int time_index = -1;
	std::vector<double> target_times;	
	if (policy == 0 || policy == 6)
	{

		for(double i = 5; i < 100; i=i+5)
		{
			target_times.push_back(model_info.local_only_time * (i/100));
		}
		for(int i = 0; i < probs.size(); i++)
		{
			std::vector<double> prob;
			prob.assign(probs[i].begin(), probs[i].end());
			
			std::vector<int> satisfied_partitioning_point;
			std::vector<int> satisfied_partitioning_prob;
			for(int j = 0; j < prob.size(); j++)
			{
				if(prob[j] >= prob_threshold)
				{
					satisfied_partitioning_point.push_back(j);
					satisfied_partitioning_prob.push_back(prob[j]);
				}
			}
			if(satisfied_partitioning_point.size() > 0)
			{
				time_index = i;
				point = satisfied_partitioning_point[std::max_element(satisfied_partitioning_prob.begin(), satisfied_partitioning_prob.end()) - satisfied_partitioning_prob.begin()];
				break;
			}
		}
		

		double expected_time = target_times[time_index];
	
		//partitioning_point = 48;
			
		r.partitioning_point = point;
		r.ex_comm = comm_time[r.partitioning_point];
		/*if(server_info.last_batch == 1){
			//	std::cout << "last batch is 1" <<server_info.last_batch << std::endl;
			r.ex_infer =  model_info.server_inference_time_ms[r.partitioning_point];
		}
		else if(server_info.last_batch == 0) // cold start
		{
			//	std::cout << "cold start use average infer " <<std::endl;
			r.ex_infer =  server_info.average_infer_time;
		}
		else{
			//	std::cout << "use previous time" << server_info.last_batch << std::endl;
			r.ex_infer = server_info.last_server_infertime;
		}
*/
		r.ex_infer = ex_infer[r.partitioning_point];
		//r.ex_infer = server_info.average_infer_time;
		r.ex_time = target_times[time_index];
		r.ex_queue = r.ex_time - r.ex_infer - r.ex_comm;
		//r.slo_time = SLO_time;
		//r.isPolicyFailed = isPolicyFailed;
	}
	else if(policy == 1 || policy == 7)
	{
		double deadline = model_info.local_only_time * (double)SLO;
		std::cout << "deadline " << deadline << " " << model_info.local_only_time << " " << (double)SLO << std::endl;
		std::vector<double> power_estimation;	
		std::vector<int> satisfied_partitioning_point;
		std::vector<int> satisfied_partitioning_point_time;
		std::vector<int> satisfied_partitioning_prob;
		std::vector<double> satisfied_point_power;

		for(int i = 0 ; i < model_info.local_power_consumption_J.size() ; i++)
		{
			double sum = comm_power(vector_mul(model_info.shapes[i])*4*8, comm_time[i]) + model_info.local_power_consumption_J[i];
			power_estimation.push_back(sum);
		}

		for(double i = 5; i < 101; i=i+5)
		{
			target_times.push_back(model_info.local_only_time * (i/100)); 	
		}

		for(int i = probs.size() -1 ; i >= 0; i--)
		{
			if(target_times[i] > deadline)
			{
				continue;
			}
			std::vector<double> prob;
			prob.assign(probs[i].begin(), probs[i].end());
			for(int j = 0; j < prob.size(); j++)
			{
				if(prob[j] >= prob_threshold)
				{
					satisfied_partitioning_point.push_back(j);
					satisfied_partitioning_point_time.push_back(target_times[i]);
					satisfied_partitioning_prob.push_back(prob[j]);
				}
			}
		}
		for( int i = 0; i < satisfied_partitioning_point.size(); i++)
		{
			satisfied_point_power.push_back(power_estimation[satisfied_partitioning_point[i]]);
		}

		double expected_time;
		if(satisfied_point_power.size() != 0)
		{
			int minpower_point = std::min_element(satisfied_point_power.begin(), satisfied_point_power.end())-satisfied_point_power.begin();
			point = satisfied_partitioning_point[minpower_point];
			expected_time = satisfied_partitioning_point_time[minpower_point];
		}
		else
		{
			std::cout << "RUN POLICY 0" << std::endl;
			if(policy == 1){
			r = get_partitioning_point(communication_time_ms, model_info, server_info, 0, SLO, prob_threshold);
		        r.isPolicyFailed = 1;}
			else if(policy == 7)
			{
				
			r = get_partitioning_point(communication_time_ms, model_info, server_info, 6, SLO, prob_threshold);

		        r.isPolicyFailed = 1;}
			return r;
			//point  =   std::min_element(power_estimation.begin(), power_estimation.end())-power_estimation.begin();
		}


		//partitioning_point = 48;
		r.partitioning_point = point;
		r.ex_comm = comm_time[r.partitioning_point];
		r.ex_infer = server_info.average_infer_time;
		r.isPolicyFailed = 0;
		
		r.ex_time = expected_time;

		r.ex_queue = r.ex_time - r.ex_infer - r.ex_comm;

	}
	else if(policy == 3)
	{
		r.partitioning_point = 0;
	}

	return r;

}
/*
struct partitioner_result get_partitioning_point(std::vector<double> communication_time_ms, ModelInfo model_info, ServerInfo server_info,  int policy, double SLO, double queue_factor)
{
	std::vector<double> time_estimation;
	std::vector<double> power_estimation;
	std::vector<double> expected_server_inference_time;

	std::vector <double > expected_queue;
	std::vector <double> expected_infer;
	double R = 1/server_info.average_throughput * server_info.average_batch + server_info.average_queue_time/1000;
	double L = 1000 / server_info.average_interval * R;


	for(int i = 0 ; i < model_info.server_inference_time_ms.size() ; i++)
	{
		//double expect = model_info.server_inference_time_ms[i] * server_info.sf + server_info.queueing;
		//double expect = server_info.average_queue_time + server_info.average_infer_time;
		double expect;
		if(!server_info.isServerInfoExpiredResult){
		
			//double maxqueue = std::ceil(L/16) * server_info.average_infer_time;
			
			double maxqueue = std::ceil(L/server_info.average_batch) * server_info.average_infer_time;
			double expected_queue_point = 0;
			double expected_infer_point = 0;
			if (server_info.average_interval > server_info.average_infer_time)
			{
				maxqueue = 0;
				expected_queue_point = maxqueue*queue_factor;
				expected_infer_point = model_info.server_inference_time_ms[i];
				expect = expected_queue_point + expected_infer_point;

				expected_queue.push_back(expected_queue_point);
				expected_infer.push_back(expected_infer_point);
			}
			else{
				maxqueue = server_info.percentile[(int)queue_factor];
				expected_queue_point = maxqueue;
					//expected_queue_point = maxqueue*queue_factor;
			//	expected_queue_point = server_info.average_queue_time;

				expected_infer_point = server_info.average_infer_time;
				expect = expected_queue_point + expected_infer_point;
				expected_queue.push_back(expected_queue_point);
				expected_infer.push_back(expected_infer_point);
			}
		}
		else
		{
			expect = model_info.server_inference_time_ms[i] + 1; //noqueue
			expected_queue.push_back(1);
			expected_infer.push_back(model_info.server_inference_time_ms[i]);
		}
		expected_server_inference_time.push_back(expect);
	}
	//expected_server_inference_time[expected_server_inference_time.size()] = 0; // local only time 0?
	expected_queue.push_back(0);
	expected_infer.push_back(0);
	expected_server_inference_time.push_back( 0 ); // local only time 0?
	
	//std::cout << "point, comm, server, local, sum " << std::endl;
	for(int i = 0 ; i < communication_time_ms.size() ; i++)
	{
		double sum = communication_time_ms[i] + expected_server_inference_time[i] + model_info.local_inference_time_ms[i];

		std::cout << i << " " << communication_time_ms[i] << " " << expected_queue[i] << " " << expected_infer[i] << " "  << model_info.local_inference_time_ms[i] << " " << sum << std::endl;
		
		
		time_estimation.push_back(sum);
	}
	
	//std::cout << "len expected queue " << expected_queue.size() <<" len expected infer " << expected_infer.size() << " len server inference time ms " << model_info.server_inference_time_ms.size() << " comm " << communication_time_ms.size() << " time esti " << time_estimation.size()  << std::endl;

	for(int i = 0 ; i < model_info.local_power_consumption_J.size() ; i++)
	{
		double sum = comm_power(vector_mul(model_info.shapes[i])*4*8, communication_time_ms[i]) + model_info.local_power_consumption_J[i];
		power_estimation.push_back(sum);
	}
	

	int partitioning_point = 0;

	double SLO_time = model_info.local_inference_time_ms[model_info.local_inference_time_ms.size()-1] * SLO; 
	double min_time = *std::min_element(time_estimation.begin(), time_estimation.end());
	
	bool isPolicyFailed = false;	
	if (min_time > SLO_time && (policy == 1 || policy == 2))
	{
		policy = 0;
		isPolicyFailed = true;
	}

	if(policy == 0) // find min time point
	{
    		partitioning_point = std::min_element(time_estimation.begin(), time_estimation.end()) - time_estimation.begin();
	}	
	
	else if(policy == 1)
	{
		double min_power = MAXVALUE;
		int min_power_idx = MAXVALUE;

	//		std::cout << "p, time , slo , expected power, comm power, local power"<<std::endl;
		for(int i = 0; i < time_estimation.size(); i++)
		{

	//		std::cout.precision(9);
	//		std::cout << i << "\t" <<time_estimation[i] << "\t" << SLO_time << "\t" << power_estimation[i] << "\t" << power_estimation[i] - model_info.local_power_consumption_J[i] << "\t" <<model_info.local_power_consumption_J[i] << std::endl;
			if(time_estimation[i] <= SLO_time && min_power > power_estimation[i])
			{
				min_power = power_estimation[i];
				min_power_idx = i;
			}
		}
		if (min_power == MAXVALUE)
		{
			std::cout << "No point in SLO" << std::endl;
			isPolicyFailed = true;
    			partitioning_point = std::min_element(time_estimation.begin(), time_estimation.end()) - time_estimation.begin();
		}
		else
		{
			partitioning_point = min_power_idx;
		}
	}	
	else if(policy == 2)
	{
		double min_servertime = MAXVALUE;
		int min_servertime_idx = MAXVALUE;
		std::cout << "p, time , slo , min server, expected server"<<std::endl;
		for(int i = 0; i < time_estimation.size(); i++)
		{
			std::cout <<i << " " <<time_estimation[i] << " " << SLO_time << " " << min_servertime << " " << expected_server_inference_time[i] << std::endl;
			if(time_estimation[i] <= SLO_time && min_servertime > expected_server_inference_time[i])
			{
				min_servertime = expected_server_inference_time[i];
				min_servertime_idx = i;
			}
		}
		if (min_servertime == MAXVALUE)
		{
			isPolicyFailed = true;
    			partitioning_point = std::min_element(time_estimation.begin(), time_estimation.end()) - time_estimation.begin();
		}
		else
		{
			partitioning_point = min_servertime_idx;
		}
	}	
	else if(policy == 3)
	{
		partitioning_point = 0;
	}
	else if(policy == 4)
	{
		partitioning_point  = model_info.layer_length-1;
	}

	struct partitioner_result r;
	//partitioning_point = 48;
	r.partitioning_point = partitioning_point;
	r.ex_comm = communication_time_ms[partitioning_point];
	r.ex_queue = expected_queue[partitioning_point];
	r.ex_infer = expected_infer[partitioning_point];
	r.ex_time = time_estimation[partitioning_point];
	r.slo_time = SLO_time;
	r.isPolicyFailed = isPolicyFailed;
	std::cout << "----------- Partitioning result --------------" << std::endl;
	std::cout << "Partitioning p : " << partitioning_point << std::endl;
	std::cout << "Expected comm  : " << communication_time_ms[partitioning_point] << " ( " << server_info.link_capacity << ")"  << std::endl;
	std::cout << "Expected queue : " << expected_queue[partitioning_point] << std::endl;
	std::cout << "Expected infer : " << expected_infer[partitioning_point] << std::endl;
	std::cout << "Expcted time s : " << time_estimation[partitioning_point] << std::endl;
	std::cout << "SLO time       : " << SLO_time << std::endl;
	std::cout << "isPolicyFailed : " << isPolicyFailed << std::endl;
	std::cout << "isServerexpired: " << server_info.isServerInfoExpiredResult << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	return r;
}

*/

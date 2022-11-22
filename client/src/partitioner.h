#ifndef PARTITIONER_H_
#define PARTITIONER_H_

#include <vector>

struct partitioner_result{

	int partitioning_point;
	double ex_comm;
	double ex_queue;
	double ex_infer;
	double ex_time;
	double slo_time;
	bool isPolicyFailed;
};

struct partitioner_result get_partitioning_point(std::vector<double> communication_time_ms, ModelInfo model_info, ServerInfo server_info,  int policy, double SLO, double queue_factor);

struct partitioner_result get_partitioning_point_baseline(int link, ModelInfo model_info, ServerInfo server_info, int policy, double SLO);
#endif

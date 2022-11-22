#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>
#include "Model.h"
#include "Server.h"
#include <random>
#include "util.h"


double servertime_estimation(ServerInfo server_info, ModelInfo model_info)
{

	double R = 1/server_info.average_throughput * server_info.average_batch + server_info.average_queue_time/1000;
	double L = 1000 / server_info.average_interval * R;
return 0;
}

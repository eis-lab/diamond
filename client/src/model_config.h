#include <string>
#include <vector>

void set_model_config(std::string model_name, std::vector<std::vector<int64_t>> &shapes, std::vector<double> &index_set, std::vector<double> &inference_time_us, std::vector<double> &server_inference_time_us, std::vector<double> &local_power_consumption_J);

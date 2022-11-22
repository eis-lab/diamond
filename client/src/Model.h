#include <string>
#include <vector>

class ModelInfo{

	public:
		ModelInfo(std::string model_name);

		std::string model_name;
		void set_model_config();
		std::vector<std::vector<int64_t>> shapes;
		std::vector<double> local_inference_time_ms;
		std::vector<double> server_inference_time_ms;
		std::vector<double> local_power_consumption_J;
		std::vector<double> batch_factor;
		std::vector<std::string> labels;
		double local_only_time;
		double layer_length;
};

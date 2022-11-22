#include <vector>
#include <fstream>
#include <string>
#include "Model.h"
#include <sstream>
std::string material_path;
ModelInfo::ModelInfo(std::string target_model_name)
{
	model_name = target_model_name;
	set_model_config();
}

void ModelInfo::set_model_config()
{
 	std::ifstream local_power_fp(material_path + "/"+model_name+"/"+model_name + "_power");
 	std::ifstream local_time_fp(material_path + "/"+model_name+"/"+model_name + "_time");
 	std::ifstream server_time_fp(material_path + "/"+model_name+"/"+model_name + "_server");
 	std::ifstream layer_shape_fp(material_path + "/"+model_name+"/"+model_name + "_shape");
 	
	std::string str;
	int count = 0;
	while(std::getline(local_power_fp, str))
	{
		if(str.size() > 0)
		{
			double p = std::stod(str);
			local_power_consumption_J.push_back(p);
		}
		count ++;
	}
	while(std::getline(local_time_fp, str))
	{
		if(str.size() > 0)
		{
			double p = std::stod(str)/1000;
			local_inference_time_ms.push_back(p);
		}
	}
	while(std::getline(server_time_fp, str))
	{
		if(str.size() > 0)
		{
			double p = std::stod(str);
			server_inference_time_ms.push_back(p);
		}
	}
	while(std::getline(layer_shape_fp, str))
	{
		if(str.size() > 0)
		{
			std::vector<int64_t> shape;
			std::istringstream ss(str);

			int64_t v;
			while(ss >> v)
			{
				shape.push_back(v);
			}
			shapes.emplace_back(shape);
		}
	}
	/*while(std::getline(batch_fp, str))
	  {	
	  if(str.size() > 0)
	  {
	  double p = std::stod(str);
	  batch_factor.push_back(p);
	  }
	  }*/



	layer_length = local_inference_time_ms.size();
	local_only_time = local_inference_time_ms[layer_length-1];
}

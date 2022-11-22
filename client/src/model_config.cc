#include "model_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::string material_path;
///////////////////////////////////////////////////////////////////////////////
void set_model_config
(std::string model_name, std::vector<std::vector<int64_t>> &shapes, 
 std::vector<double> &index_set, std::vector<double> &inference_time_us, std::vector<double> server_inference_time_us, std::vector<double> &local_power_consumption_J)
{
 	std::ifstream local_power_fp(material_path + "/"+model_name+"/"+model_name + "_power");
 	std::ifstream local_time_fp(material_path + "/"+model_name+"/"+model_name + "_time");
 	std::ifstream server_time_fp(material_path + "/"+model_name+"/"+model_name + "_server");
 	std::ifstream layer_shape_fp(material_path + "/"+model_name+"/"+model_name + "_shape");
 	
	std::string str;
	int count = 0;
	while(std::getline(local_power_fp, str))
	{
		index_set.push_back((double)count);
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
			double p = std::stod(str);
			inference_time_us.push_back(p);
		}
	}
	while(std::getline(server_time_fp, str))
	{
		if(str.size() > 0)
		{
			double p = std::stod(str);
			server_inference_time_us.push_back(p);
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
			//shapes.push_back(shape);
		}
	}
}

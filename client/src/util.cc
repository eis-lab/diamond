#include "util.h"
#include <chrono>
#include <iostream>
uint64_t get_current_unixtime()
{

	std::chrono::system_clock::time_point current_time = std::chrono::system_clock::now();
	
       uint64_t ret = std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count();

       return ret;
	
}
std::vector<double> parseStrToDoubleVec(std::string src)
{
	std::vector<double> ret;
	size_t pos = 0;	
	std::string delimiter = ",";
	std::string token;
	while ((pos = src.find(delimiter)) != std::string::npos) {
            token = src.substr(0, pos);
            std::string strtoken(token);
            ret.push_back(std::stod(strtoken));
            src.erase(0, pos + delimiter.length());
        }
	if(src.length() !=0){
	ret.push_back(std::stod(src));
	}
	return ret;

}

std::vector<int> parseStrToIntVec(std::string src)
{
	std::vector<int> ret;
	size_t pos = 0;	
	std::string delimiter = ",";
	std::string token;
	while ((pos = src.find(delimiter)) != std::string::npos) {
            token = src.substr(0, pos);
            std::string strtoken(token);
            ret.push_back(std::stoi(strtoken));
            src.erase(0, pos + delimiter.length());
        }
	if(src.length() !=0){
	ret.push_back(std::stoi(src));
	}
	return ret;

}
uint64_t vector_mul(std::vector<int64_t> vector)
{
        uint64_t ret=1;
        for(int i =0;i < vector.size();i++)
        {
                ret =ret*vector[i];
        }

        return ret;
}
double vector_mean(std::vector<double> vector)
{
        double sum = 0;
        for(int i = 0; i < vector.size(); i++)
        {
                sum += vector[i];
        }

        return sum/vector.size();
}
bool loadimagenetlabel(std::string file_name, std::vector<std::string>& labels)
{
	std::ifstream ifs(file_name);
	if (!ifs)
	{
		return false;
	}
	std::string line;
	while (std::getline(ifs, line))
	{
		labels.push_back(line);
	}
	return true;
}
void print_longvector(std::vector<uint64_t> v)
{
	for(int i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}	
	std::cout << std::endl;
}
void print_doublevector(std::vector<double> v)
{
	for(int i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}	
	std::cout << std::endl;
}
void print_vector(std::vector<int> v)
{
	for(int i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}	
	std::cout << std::endl;
}

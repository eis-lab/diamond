#include <string>
#include <vector>
#include <fstream>
#define FAIL_IF_ERR(X, MSG)                                        \
{                                                                \
	nic::Error err = (X);                                          \
	if (!err.IsOk()) {                                             \
		std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
		exit(1);                                                     \
	}                                                              \
}

std::vector<int> parseStrToIntVec(std::string src);
std::vector<double> parseStrToDoubleVec(std::string src);
uint64_t vector_mul(std::vector<int64_t> vector);
double vector_mean(std::vector<double> vector);

bool loadimagenetlabel(std::string file_name, std::vector<std::string>& labels);

void print_vector(std::vector<int> v);
void print_doublevector(std::vector<double> v);
void print_longvector(std::vector<uint64_t> v);
uint64_t get_current_unixtime();

#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cmath>
#include <chrono>
#include <vector>
#include "Server.h"

extern "C" {
#include<curl/curl.h>
}


enum {
    ERROR_ARGS = 1,
    ERROR_CURL_INIT = 2
};

enum {
    OPTION_FALSE = 0,
    OPTION_TRUE = 1
};

enum {
    FLAG_DEFAULT = 0
};



class Communication
{
	public:
		double current_measured_rtt;
		const int init_cwnd = 10;
		const int MSS = 1448; // Byte
		int MAX_LINK;
		int LINK;
		double naive_link_bps;
		int MAX_SERVER_CAPACITY;
		int SERVER_CAPACITY;
		int MEAN_SERVER_CAPACITY;
		double expect_cwnd;
		struct tcp_info current_tcp_info;
		void print_info(bool verbose);
		void set_tcp_info(struct tcp_info tcpinfo);
		double expect_time(int64_t datasize_as_byte, bool verbose);
		void Init(double rtt_ms);	
		int get_LINK(int64_t datsize_as_byte, double comm_time);		
		double expect_time_with_given_link(int64_t datasize_as_byte, int link, bool *reach_to_max_rtt, double measured_rtt);
		
		std::vector<double> expect_time_shapes(std::vector<std::vector<int64_t>> shapes, double link, ServerInfo server_info);
		void get_server_capacity(std::vector<std::vector<int64_t>> shapes, std::vector<int> arrival_rate);
		
		Communication(int bottleneck_bw);
};

#ifndef SERVER_H
#define SERVER_H

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

class ServerInfo{

	public:
		const int init_cwnd = 10;
		const int MSS = 1448; // Byte

		int MAX_LINK;
		int LINK;
		double naive_link_bps;
		int MAX_SERVER_CAPACITY;
		int CURRENT_SERVER_CAPACITY;
		int MEAN_SERVER_CAPACITY;

		double expect_cwnd;
		struct tcp_info current_tcp_info;

		double sf;
		double queueing;
		std::vector<double> rtt_history;
		double rtt;
		int link_capacity;
		ServerInfo();
		void GetServerInfo();
		void init();					
		void RTTrefresh(double current_rtt);
		void SetServerInfo(double a, double b, double c, double d, double e);
		void ResetServerInfo();
		uint64_t server_information_refresh_time;
		uint64_t server_information_send_time;
		uint64_t server_information_update_time;

	//	std::vector<int> server_queue_status;
	//	std::vector<int> server_arrival_rate;
		
		bool isMyInfoExpired();
		bool isServerInfoExpired();
		double GetServerInfoNoRefresh();	
		bool isServerInfoExpiredResult;
		double current_measured_rtt;
		double average_interval; // interval
		double average_batch; //
		double average_throughput;
		double average_infer_time;
		double average_queue_time;
		double last_batch;
		double last_server_infertime;
		std::vector<double> percentile;
};
#endif
